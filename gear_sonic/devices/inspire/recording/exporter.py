"""Transactional adapter around the shared WBC ``Gr00tDataExporter``."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from ..config import DEFAULT_MAPPING_PATH

from .depth import DepthEpisodeWriter
from .frame import InspireLeRobotFrame
from .sonic_reference import SonicReferenceWriter
from .schema import (
    LEROBOT_TIMING_PATH,
    LEROBOT_TIMING_SCHEMA,
    SIM_VALIDATION_LEVEL,
    TIMING_FIELDS,
    capture_identity,
    lerobot_features,
    lerobot_modality_config,
)

MIN_EPISODE_FRAMES = 40


def _default_exporter_factory(**kwargs: Any) -> Any:
    # Keep LeRobot, datasets and PyAV out of normal Inspire imports.
    from gear_sonic.data.exporter import Gr00tDataExporter

    return Gr00tDataExporter.create(**kwargs)


class InspireLeRobotExporter:
    """Write one episode to staging and publish it with an atomic rename."""

    def __init__(
        self,
        output: str | Path,
        *,
        robot_model: Any,
        config_path=DEFAULT_MAPPING_PATH,
        fps: int,
        task_prompt: str,
        validation_level: str = SIM_VALIDATION_LEVEL,
        exporter_factory: Callable[..., Any] = _default_exporter_factory,
        deployment=None,
        outcome='unlabeled',
    ) -> None:
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"output already exists: {self.output}")
        if not task_prompt.strip():
            raise ValueError("task_prompt must not be empty")
        if fps <= 0:
            raise ValueError("fps must be positive")
        # Validate the level once before allocating staging or the shared writer.
        self.identity = capture_identity(fps=fps, validation_level=validation_level, config_path=config_path)
        self.device_metadata = None
        if deployment is not None:
            from gear_sonic.utils.data_collection.end_effector import recording_metadata
            self.device_metadata = recording_metadata(deployment, task=task_prompt, outcome=outcome,
                                                       source='sonic_read_only_streams')
        self.output.parent.mkdir(parents=True, exist_ok=True)
        staging = tempfile.mkdtemp(prefix=f".{self.output.name}.staging-", dir=self.output.parent)
        self.staging_root = Path(staging)
        self.dataset_root = self.staging_root / "dataset"
        self.fps = fps
        self.task_prompt = task_prompt.strip()
        self.validation_level = validation_level
        self.robot_model = robot_model
        self.config_path = config_path
        self.frame_count = 0
        self._depth_writer = None
        self._has_depth = None
        self._timing = {name: [] for name in TIMING_FIELDS}
        self._sonic_reference = SonicReferenceWriter()
        self._committed = False
        self._writer: Any | None = None
        try:
            self._writer = exporter_factory(
                save_root=self.dataset_root,
                fps=fps,
                features=lerobot_features(robot_model),
                modality_config=lerobot_modality_config(robot_model),
                task=self.task_prompt,
                script_config={
                    "collector": "gear_sonic.devices.inspire.recording",
                    "mode": "pico_read_only",
                },
                robot_type="unitree_g1_inspire_rh56dfx",
                overwrite_existing=False,
            )
        except Exception:
            self._mark_incomplete("exporter initialization failed")
            raise

    def append(self, frame: InspireLeRobotFrame) -> None:
        if self._committed or self._writer is None:
            raise RuntimeError("exporter is not writable")
        if not isinstance(frame, InspireLeRobotFrame):
            raise TypeError("writer accepts only validated InspireLeRobotFrame objects")
        has_depth = frame.depth is not None
        if self._has_depth is None:
            self._has_depth = has_depth
            if has_depth:
                self._depth_writer = DepthEpisodeWriter(self.dataset_root, 0)
        elif has_depth != self._has_depth:
            raise ValueError("depth appeared/disappeared during capture")
        if self._depth_writer is not None:
            self._depth_writer.append(frame, self.frame_count)
        self._writer.add_frame(frame.to_exporter_dict())
        self._sonic_reference.append(frame)
        for name in TIMING_FIELDS:
            self._timing[name].append(getattr(frame, name))
        self.frame_count += 1

    def _write_timing_trace(self) -> None:
        path = self.dataset_root / LEROBOT_TIMING_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema=np.asarray(LEROBOT_TIMING_SCHEMA),
            **{key: np.asarray(values, dtype=np.int64) for key, values in self._timing.items()},
        )

    def _cancel_open_video_writers(self) -> None:
        if self._writer is None:
            return
        for writer in getattr(self._writer, "video_writers", {}).values():
            try:
                writer.cancel()
            except Exception:
                pass
        self._writer.video_writers = {}

    def _mark_incomplete(self, reason: str) -> None:
        try:
            self.staging_root.mkdir(parents=True, exist_ok=True)
            marker = self.staging_root / "INCOMPLETE"
            marker.write_text(reason.rstrip() + "\n", encoding="utf-8")
        except OSError:
            pass

    def abort(self, reason: str) -> Path:
        """Close this run and leave an explicit staging marker for diagnosis."""
        if not self._committed:
            self._cancel_open_video_writers()
            self._mark_incomplete(reason)
            self._writer = None
        return self.staging_root

    def commit(self, capture_summary: dict[str, Any]) -> Path:
        if self._committed or self._writer is None:
            raise RuntimeError("exporter has already been closed")
        if self.frame_count < MIN_EPISODE_FRAMES:
            raise ValueError(f"episode requires at least {MIN_EPISODE_FRAMES} frames, got {self.frame_count}")
        try:
            self._writer.save_episode()
            # The shared exporter prepares another episode after every save.
            # This adapter is deliberately one-shot, so close that empty writer.
            self._cancel_open_video_writers()
            self._write_timing_trace()
            capture = self.identity | {
                "sonic_reference": self._sonic_reference.finish(self.dataset_root),
                "actual_frames": self.frame_count,
                "task_prompt": self.task_prompt,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "summary": capture_summary,
                "timing_trace": {
                    "schema": LEROBOT_TIMING_SCHEMA,
                    "path": LEROBOT_TIMING_PATH,
                    "frames": self.frame_count,
                },
            }
            if self._depth_writer is not None:
                capture["depth"] = self._depth_writer.finish()
            capture["source_metadata"] = {
                "control_writes": False,
                "hand_action": "submitted_target",
                "hand_intent": "optical_before_rate_limit",
            }
            capture_path = self.dataset_root / "meta" / "inspire_capture.json"
            capture_path.write_text(json.dumps(capture, indent=2) + "\n", encoding="utf-8")
            if self.device_metadata is not None:
                metadata = self.device_metadata | {'schema':'end_effector.recording.v1',
                    'source_schema':capture['capture_schema'], 'frames':self.frame_count,
                    'modality':lerobot_modality_config(self.robot_model),
                    'control_writes':False, 'timestamp_trace':capture['timing_trace']}
                (self.dataset_root/'meta/end_effector.json').write_text(json.dumps(metadata, indent=2))
                from gear_sonic.utils.data_collection.end_effector import preprocessing_config
                preprocessing = preprocessing_config(lerobot_features(self.robot_model),
                                                       metadata['modality'], metadata['deployment'])
                (self.dataset_root/'meta/preprocessing.json').write_text(json.dumps(preprocessing, indent=2))

            from .audit import audit_lerobot_dataset

            audit_lerobot_dataset(
                self.dataset_root,
                robot_model=self.robot_model,
                config_path=self.config_path,
            )
            if self.output.exists():
                raise FileExistsError(f"output appeared during capture: {self.output}")
            self.dataset_root.rename(self.output)
            shutil.rmtree(self.staging_root)
            self._committed = True
            return self.output
        except Exception as exc:
            self._cancel_open_video_writers()
            self._mark_incomplete(f"commit failed: {type(exc).__name__}: {exc}")
            raise

    def __enter__(self) -> "InspireLeRobotExporter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._committed:
            self.abort(
                f"capture failed: {exc_type.__name__}: {exc}" if exc is not None else "capture was not committed"
            )
