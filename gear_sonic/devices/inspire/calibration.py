"""Capture stable optical poses and save PICO hand calibration without robot control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .config import DEFAULT_MAPPING_PATH, load_mapping
from .contract import JOINT_ORDER
from .optical import (
    InspireOpticalCalibrationBundle,
    OpticalRetargetingError,
    extract_inspire_features,
)
from gear_sonic.utils.teleop.inspire import OpticalPicoReader


class CalibrationCaptureError(RuntimeError):
    """Raised when a requested optical calibration pose is not stable/valid."""


@dataclass(frozen=True)
class CapturedOpticalPose:
    name: str
    samples: int
    first_device_timestamp_ns: int
    last_device_timestamp_ns: int
    left_features: np.ndarray
    right_features: np.ndarray
    left_peak_to_peak: np.ndarray
    right_peak_to_peak: np.ndarray


def capture_stable_pose(
    reader: Any,
    *,
    name: str,
    samples: int = 30,
    timeout_s: float = 10.0,
    max_peak_to_peak: float = 0.15,
) -> CapturedOpticalPose:
    """Capture unique valid frames and return per-feature medians.

    Invalid or one-handed frames are skipped, never converted to zero. A pose
    with excessive motion is rejected instead of embedding noise into the
    calibration endpoints.
    """

    if samples < 5:
        raise ValueError("samples must be at least 5")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_peak_to_peak <= 0:
        raise ValueError("max_peak_to_peak must be positive")

    left_values: list[np.ndarray] = []
    right_values: list[np.ndarray] = []
    timestamps: list[int] = []
    last_timestamp = -1
    last_error = "no optical frame"
    deadline = time.monotonic() + timeout_s
    while len(left_values) < samples and time.monotonic() < deadline:
        tracked = reader.get_hand_tracking_data()
        if tracked is None or tracked.timestamp_ns <= last_timestamp:
            time.sleep(0.002)
            continue
        last_timestamp = tracked.timestamp_ns
        if tracked.left is None or tracked.right is None:
            last_error = "both optical hands are required"
            continue
        try:
            left = extract_inspire_features(tracked.left)
            right = extract_inspire_features(tracked.right)
        except OpticalRetargetingError as exc:
            last_error = str(exc)
            continue
        left_values.append(left)
        right_values.append(right)
        timestamps.append(tracked.timestamp_ns)

    if len(left_values) != samples:
        raise CalibrationCaptureError(
            f"{name}: captured {len(left_values)}/{samples} valid frames before timeout; last_error={last_error}"
        )

    left_array = np.stack(left_values)
    right_array = np.stack(right_values)
    left_span = np.ptp(left_array, axis=0)
    right_span = np.ptp(right_array, axis=0)
    unstable = max(float(np.max(left_span)), float(np.max(right_span)))
    if unstable > max_peak_to_peak:
        raise CalibrationCaptureError(
            f"{name}: pose moved during capture; peak_to_peak={unstable:.6f} > {max_peak_to_peak:.6f} rad"
        )
    return CapturedOpticalPose(
        name=name,
        samples=samples,
        first_device_timestamp_ns=timestamps[0],
        last_device_timestamp_ns=timestamps[-1],
        left_features=np.median(left_array, axis=0).astype(np.float32),
        right_features=np.median(right_array, axis=0).astype(np.float32),
        left_peak_to_peak=left_span.astype(np.float32),
        right_peak_to_peak=right_span.astype(np.float32),
    )


def _prompt_and_capture(reader, prompt: str, name: str) -> CapturedOpticalPose:
    input(f"\n{name}: {prompt}\nHold the pose still, then press Enter to capture: ")
    result = capture_stable_pose(
        reader,
        name=name,
        samples=30,
        timeout_s=15,
        max_peak_to_peak=0.15,
    )
    print(f"{name}: captured", flush=True)
    return result


def calibrate(reader, output, config_path=DEFAULT_MAPPING_PATH):
    """Capture the four required poses using the already-running reader."""
    calibration_id = f"pico-{time.strftime('%Y%m%d-%H%M%S')}"
    opened = _prompt_and_capture(
        reader,
        "Open all fingers naturally and keep both thumbs extended.",
        "OPEN",
    )
    fingers = _prompt_and_capture(
        reader,
        "Flex pinky/ring/middle/index fully; keep both thumbs in the OPEN pose.",
        "FOUR_FINGERS_CLOSED",
    )
    thumb_bend = _prompt_and_capture(
        reader,
        "Open the four fingers; bend both thumbs fully without rotating them across the palm.",
        "THUMB_BEND",
    )
    thumb_rotation = _prompt_and_capture(
        reader,
        "Open the four fingers; rotate both thumbs across the palm while minimizing thumb bend.",
        "THUMB_ROTATION",
    )
    bundle = InspireOpticalCalibrationBundle.from_independent_feature_poses(
        calibration_id=calibration_id,
        created_realtime_ns=time.time_ns(),
        samples_per_pose=30,
        tracker_source=reader.tracker_source,
        mapping=load_mapping(config_path),
        left_open=opened.left_features,
        right_open=opened.right_features,
        left_fingers_closed=fingers.left_features,
        right_fingers_closed=fingers.right_features,
        left_thumb_bent=thumb_bend.left_features,
        right_thumb_bent=thumb_bend.right_features,
        left_thumb_rotated=thumb_rotation.left_features,
        right_thumb_rotated=thumb_rotation.right_features,
    )
    left_span = np.abs(bundle.left.closed_features - bundle.left.open_features)
    right_span = np.abs(bundle.right.closed_features - bundle.right.open_features)
    invalid = np.argwhere(np.stack([left_span, right_span]) < float(0.15))
    if invalid.size:
        details = [f"{'left' if int(side) == 0 else 'right'}:{JOINT_ORDER[int(index)]}" for side, index in invalid]
        raise RuntimeError("calibration feature span is too small for: " + ", ".join(details))
    path = bundle.save(output)
    print(f"Calibration saved: {path}", flush=True)
    return bundle


def main():
    from gear_sonic.utils.hand_control.management import DeviceLease
    from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment

    parser = argparse.ArgumentParser(description="Recalibrate PICO optical hands; no robot commands")
    parser.add_argument("output", nargs="?", default="outputs/inspire_calibrations/pico_hands.json")
    parser.add_argument("--hand-config", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    args = parser.parse_args()
    config = deployment('inspire', registry_dir=args.registry_dir, options={'hand_config': args.hand_config})
    if Path(args.output).expanduser().resolve().is_relative_to(config.registration.source.parent):
        parser.error('calibration output must be outside the device package')
    with DeviceLease(config, 'hand_calibration'):
        reader = OpticalPicoReader(body=False)
        try:
            reader.start()
            calibrate(reader, args.output, args.hand_config)
        finally:
            reader.stop()


if __name__ == "__main__":
    main()
