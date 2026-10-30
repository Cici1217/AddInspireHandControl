"""Lossless depth sidecars, indexed by the existing LeRobot episode/frame clock."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ..contract import ContractError


@dataclass(frozen=True)
class DepthFrame:
    image: np.ndarray
    source_timestamp_ns: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if image.dtype != np.uint16 or image.ndim != 2 or not image.size:
            raise ContractError("ego_view_depth must be a nonempty 2D uint16 array")
        if self.source_timestamp_ns <= 0:
            raise ContractError("depth source timestamp must be positive")
        try:
            metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
            calibration = metadata["calibration"]
            scale = float(calibration["depth_scale_m"])
            intrinsics = calibration["depth_intrinsics"]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("depth scale must be positive")
            if (intrinsics["height"], intrinsics["width"]) != image.shape:
                raise ValueError("depth dimensions do not match intrinsics")
            if not all(np.isfinite(intrinsics[k]) for k in ("fx", "fy", "ppx", "ppy")):
                raise ValueError("non-finite depth intrinsics")
            if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
                raise ValueError("invalid depth focal length")
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid depth calibration: {exc}") from exc
        owned = image.copy()
        owned.flags.writeable = False
        object.__setattr__(self, "image", owned)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_camera_message(cls, message: dict[str, Any]) -> DepthFrame | None:
        image = message.get("images", {}).get("ego_view_depth")
        if image is None:
            return None
        try:
            timestamp = float(message["timestamps"]["ego_view_depth"])
            if not np.isfinite(timestamp) or timestamp <= 0:
                raise ValueError("invalid depth timestamp")
            metadata = message["metadata"]["ego_view"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                "depth needs source timestamp and camera metadata; update the camera server"
            ) from exc
        return cls(image, int(timestamp * 1_000_000_000), metadata)


class DepthEpisodeWriter:
    """Write only unique received depth frames; retain a reference for every action row."""

    def __init__(self, root: Path, episode_index: int) -> None:
        self.root = root
        self.episode_index = episode_index
        self.relative_dir = Path("depth") / f"episode_{episode_index:06d}"
        self.directory = root / self.relative_dir
        self.directory.mkdir(parents=True, exist_ok=False)
        self.rows: list[dict[str, Any]] = []
        self.calibration: dict[str, Any] | None = None
        self.unique_frames = 0
        self._last_timestamp_ns = -1
        self._last_file = ""

    def append(self, frame, dataset_frame_index: int) -> None:
        import cv2

        depth = frame.depth
        calibration = depth.metadata["calibration"]
        if self.calibration is None:
            self.calibration = calibration
        elif calibration != self.calibration:
            raise ValueError("depth camera calibration changed during an episode")
        timestamp = depth.source_timestamp_ns
        if timestamp < self._last_timestamp_ns:
            raise ValueError("depth source timestamp regressed")
        if timestamp != self._last_timestamp_ns:
            path = self.relative_dir / f"frame_{self.unique_frames:06d}.png"
            if not cv2.imwrite(str(self.root / path), depth.image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"failed to save depth frame: {path}")
            self._last_file = path.as_posix()
            self._last_timestamp_ns = timestamp
            self.unique_frames += 1
        self.rows.append(
            {
                "frame_index": len(self.rows),
                "dataset_frame_index": dataset_frame_index,
                "depth_file": self._last_file,
                "sample_monotonic_ns": frame.sample_monotonic_ns,
                "rgb_source_timestamp_ns": frame.camera_source_timestamp_ns,
                "depth_source_timestamp_ns": timestamp,
                "source": {k: v for k, v in depth.metadata.items() if k != "calibration"},
            }
        )

    def finish(self) -> dict[str, Any]:
        relative_manifest = Path("meta/depth") / f"episode_{self.episode_index:06d}.json"
        manifest = self.root / relative_manifest
        manifest.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "inspire.depth.v1",
            "episode_index": self.episode_index,
            "rgb_feature": "observation.images.ego_view",
            "encoding": "png_uint16_lossless",
            "distance_conversion": "depth_m = stored_uint16 * calibration.depth_scale_m",
            "invalid_depth_value": self.calibration.get("invalid_depth_value", 0),
            "calibration": self.calibration,
            "frames": self.rows,
            "unique_depth_frames": self.unique_frames,
            "alignment_note": (
                "RGB and depth retain their native pixel grids; use depth_to_color for registration."
            ),
            "clock_note": (
                "Source timestamps are camera-host receive times; sample_monotonic_ns is recorder-local. "
                "SDK timestamps/domains are retained in source.frames; no cross-host clock calibration is implied."
            ),
        }
        manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return {
            "episode_index": self.episode_index,
            "manifest": relative_manifest.as_posix(),
            "frames": len(self.rows),
            "unique_depth_frames": self.unique_frames,
        }

    def discard(self) -> None:
        shutil.rmtree(self.directory)


class DepthPolicyCamera:
    """Apply an explicit save policy without changing shared camera messages."""

    def __init__(self, camera, policy):
        if policy not in ("off", "required", "auto"):
            raise ValueError("invalid depth policy")
        self.camera, self.policy = camera, policy

    def read(self):
        message = self.camera.read()
        if message is None:
            return None
        if self.policy == "required":
            from .depth import DepthFrame

            if DepthFrame.from_camera_message(message) is None:
                raise RuntimeError("已选择保存深度，但相机只发送 RGB；启动 RealSense RGB-D 或仿真 RGB-D 发布器")
        if self.policy == "off":
            return message | {
                "images": {k: v for k, v in message.get("images", {}).items() if not k.endswith("_depth")}
            }
        return message
