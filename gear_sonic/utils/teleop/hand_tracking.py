"""Strict, device-neutral OpenXR hand-tracking snapshots.

IsaacTeleop exposes 26 joints in ``XrHandJointEXT`` order.  This module owns
the schema boundary so robot-specific retargeters never depend on pybind or
FlatBuffer object layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

HAND_JOINT_COUNT = 26


class HandTrackingError(ValueError):
    """Raised when a hand snapshot violates the OpenXR tracking contract."""


class OpenXRHandJoint(IntEnum):
    PALM = 0
    WRIST = 1
    THUMB_METACARPAL = 2
    THUMB_PROXIMAL = 3
    THUMB_DISTAL = 4
    THUMB_TIP = 5
    INDEX_METACARPAL = 6
    INDEX_PROXIMAL = 7
    INDEX_INTERMEDIATE = 8
    INDEX_DISTAL = 9
    INDEX_TIP = 10
    MIDDLE_METACARPAL = 11
    MIDDLE_PROXIMAL = 12
    MIDDLE_INTERMEDIATE = 13
    MIDDLE_DISTAL = 14
    MIDDLE_TIP = 15
    RING_METACARPAL = 16
    RING_PROXIMAL = 17
    RING_INTERMEDIATE = 18
    RING_DISTAL = 19
    RING_TIP = 20
    LITTLE_METACARPAL = 21
    LITTLE_PROXIMAL = 22
    LITTLE_INTERMEDIATE = 23
    LITTLE_DISTAL = 24
    LITTLE_TIP = 25


def _owned_array(value: Any, shape: tuple[int, ...], *, name: str, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise HandTrackingError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise HandTrackingError(f"{name} contains NaN or infinity")
    result = np.array(result, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrackedHandJoints:
    """One owned OpenXR hand snapshot in fixed 26-joint order."""

    positions: np.ndarray
    orientations: np.ndarray
    radii: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        positions = _owned_array(self.positions, (HAND_JOINT_COUNT, 3), name="positions", dtype=np.float32)
        orientations = _owned_array(
            self.orientations,
            (HAND_JOINT_COUNT, 4),
            name="orientations",
            dtype=np.float32,
        )
        radii = _owned_array(self.radii, (HAND_JOINT_COUNT,), name="radii", dtype=np.float32)
        valid_raw = np.asarray(self.valid)
        if valid_raw.shape != (HAND_JOINT_COUNT,):
            raise HandTrackingError(f"valid must have shape ({HAND_JOINT_COUNT},), got {valid_raw.shape}")
        valid = np.array(valid_raw, dtype=np.bool_, copy=True)
        valid.setflags(write=False)

        if np.any(radii < 0.0):
            raise HandTrackingError("radii must be non-negative")
        valid_quaternions = orientations[valid]
        if valid_quaternions.size:
            norms = np.linalg.norm(valid_quaternions, axis=1)
            if np.any(norms <= 1e-8):
                raise HandTrackingError("valid joint orientation has zero quaternion norm")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "orientations", orientations)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "valid", valid)

    def require_valid(self, indices: tuple[int, ...] | list[int] | np.ndarray) -> None:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested < 0) or np.any(requested >= HAND_JOINT_COUNT):
            raise HandTrackingError("required joint indices are outside OpenXR hand range")
        missing = requested[~self.valid[requested]]
        if missing.size:
            raise HandTrackingError(f"required OpenXR hand joints are invalid: {missing.tolist()}")


@dataclass(frozen=True)
class TrackedHandsFrame:
    """Synchronized left/right hand snapshots from one DeviceIO update."""

    timestamp_ns: int
    timestamp_monotonic: float
    timestamp_realtime: float
    left: TrackedHandJoints | None
    right: TrackedHandJoints | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, (bool, np.bool_))
            or not isinstance(self.timestamp_ns, (int, np.integer))
            or self.timestamp_ns < 0
        ):
            raise HandTrackingError("timestamp_ns must be a non-negative integer")
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        times = np.asarray([self.timestamp_monotonic, self.timestamp_realtime], dtype=np.float64)
        if not np.all(np.isfinite(times)):
            raise HandTrackingError("hand frame timestamps must be finite")


def _component(value: Any, name: str) -> float:
    attr = getattr(value, name, None)
    if attr is not None:
        return float(attr)
    index = {"x": 0, "y": 1, "z": 2, "w": 3}[name]
    try:
        return float(value[index])
    except Exception as exc:
        raise HandTrackingError(f"cannot extract {name} from {type(value).__name__}") from exc


def _from_joint_accessor(joints: Any) -> TrackedHandJoints:
    accessor = getattr(joints, "poses", None)
    if not callable(accessor):
        raise HandTrackingError("hand joints do not expose the official poses(index) accessor")

    positions = np.zeros((HAND_JOINT_COUNT, 3), dtype=np.float32)
    orientations = np.zeros((HAND_JOINT_COUNT, 4), dtype=np.float32)
    radii = np.zeros(HAND_JOINT_COUNT, dtype=np.float32)
    valid = np.zeros(HAND_JOINT_COUNT, dtype=np.bool_)
    # Invalid joints may contain unspecified pose fields.  Keep deterministic
    # finite placeholders, but never mark them valid.
    orientations[:, 3] = 1.0

    for index in range(HAND_JOINT_COUNT):
        try:
            joint = accessor(index)
        except Exception as exc:
            raise HandTrackingError(f"cannot read OpenXR hand joint {index}") from exc
        if joint is None:
            continue
        is_valid = bool(getattr(joint, "is_valid", False))
        valid[index] = is_valid
        radius = float(getattr(joint, "radius", 0.0))
        radii[index] = radius if np.isfinite(radius) and radius >= 0.0 else 0.0
        if not is_valid:
            continue
        pose = getattr(joint, "pose", None)
        if pose is None:
            raise HandTrackingError(f"valid OpenXR hand joint {index} has no pose")
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)
        if position is None or orientation is None:
            raise HandTrackingError(f"valid OpenXR hand joint {index} has incomplete pose")
        positions[index] = [_component(position, axis) for axis in ("x", "y", "z")]
        orientations[index] = [_component(orientation, axis) for axis in ("x", "y", "z", "w")]

    return TrackedHandJoints(positions, orientations, radii, valid)


def hand_data_to_tracked(hand_data: Any) -> TrackedHandJoints | None:
    """Convert one IsaacTeleop ``HandPose.data`` payload to an owned snapshot.

    Current IsaacTeleop bindings provide zero-copy bulk arrays.  The accessor
    fallback supports earlier builds with the same official FlatBuffer schema.
    ``None`` means tracking for that hand is inactive.
    """

    if hand_data is None:
        return None
    joints = getattr(hand_data, "joints", None)
    if joints is None:
        raise HandTrackingError(f"unrecognised hand schema {type(hand_data).__name__}: missing joints")

    bulk = tuple(getattr(joints, name, None) for name in ("positions", "orientations", "radii"))
    bulk_valid = getattr(joints, "is_valid", None)
    if all(value is not None for value in bulk) and bulk_valid is not None:
        return TrackedHandJoints(bulk[0], bulk[1], bulk[2], bulk_valid)
    return _from_joint_accessor(joints)


# XRoboToolkit acquisition is independent of the selected robot end effector.
import logging
import threading
import time

logger = logging.getLogger(__name__)

def xrt_hand_state_to_tracked(state, *, active):
    """Adapt XRoboToolkit's OpenXR 26x7 arrays; inactive joints stay invalid."""
    if not active:
        return None
    joints = np.asarray(state, dtype=np.float32)
    if joints.shape != (26, 7):
        raise HandTrackingError(f"XRoboToolkit hand state must be (26, 7), got {joints.shape}")
    positions, orientations = joints[:, :3].copy(), joints[:, 3:].copy()
    finite = np.all(np.isfinite(joints), axis=1)
    valid = finite & (np.linalg.norm(orientations, axis=1) > 1e-8)
    orientations[~valid] = [0, 0, 0, 1]
    positions[~finite] = 0
    return TrackedHandJoints(positions, orientations, np.zeros(26), valid)


class OpticalPicoReader:
    """Own one SDK on one thread and publish only the latest owned snapshot.

    XRT's device timestamp brackets the body and both hand reads. Calibration uses the
    same reader with body=False and never opens a robot connection.
    """

    def __init__(self, *, body=True, sdk=None):
        self.body = body
        self.tracker_source = "xrt.openxr26"
        self._sdk = sdk
        self._latest = None
        self._error = None
        self._last_stamp = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("reader already started")
        if self._sdk is None:
            try:
                import xrobotoolkit_sdk
            except ImportError as exc:
                raise RuntimeError("install PICO dependencies with install_scripts/install_pico.sh") from exc
            self._sdk = xrobotoolkit_sdk
        try:
            self._sdk.init()
            self._thread = threading.Thread(target=self._run, name="pico-reader", daemon=True)
            self._thread.start()
        except BaseException:
            self._sdk.close()
            raise

    def _read_atomic(self):
        stamp = int(self._sdk.get_time_stamp_ns())
        if stamp <= self._last_stamp or (self.body and not self._sdk.is_body_data_available()):
            return None
        body = np.array(self._sdk.get_body_joints_pose(), dtype=np.float32) if self.body else None
        left = right = None
        try:
            # Copy SDK buffers inside the timestamp boundary; validate afterward.
            if self._sdk.get_left_hand_is_active():
                left = np.array(self._sdk.get_left_hand_tracking_state(), dtype=np.float32)
            if self._sdk.get_right_hand_is_active():
                right = np.array(self._sdk.get_right_hand_tracking_state(), dtype=np.float32)
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.debug("PICO optical read unavailable: %s", exc)
            left = right = None
        if int(self._sdk.get_time_stamp_ns()) != stamp:
            return None
        if self.body and (body.shape != (24, 7) or not np.all(np.isfinite(body))):
            raise HandTrackingError("PICO body poses must be finite (24, 7)")
        try:
            left = xrt_hand_state_to_tracked(left, active=left is not None)
            right = xrt_hand_state_to_tracked(right, active=right is not None)
        except (HandTrackingError, ValueError) as exc:
            # As in the original manager, an invalid hand must not discard valid body data.
            logger.debug("PICO optical frame rejected: %s", exc)
            left = right = None
        now = time.monotonic()
        realtime = time.time()
        hands = TrackedHandsFrame(stamp, now, realtime, left, right)
        dt = (stamp - self._last_stamp) * 1e-9 if self._last_stamp else 0.0
        self._last_stamp = stamp
        return {
            "body_poses_np": body,
            "tracked_hands": hands,
            "timestamp_ns": stamp,
            "timestamp_monotonic": now,
            "timestamp_realtime": realtime,
            "dt": dt,
            "fps": 1.0 / dt if dt > 0 else 0.0,
        }

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    sample = self._read_atomic()
                except (HandTrackingError, ValueError) as exc:
                    logger.debug("PICO frame rejected: %s", exc)
                    sample = None
                if sample is not None:
                    with self._lock:
                        self._latest = sample
                self._stop.wait(0.001)
        except BaseException as exc:
            self._error = exc

    def get_latest(self):
        if self._error is not None:
            raise RuntimeError(f"PICO reader failed: {self._error}") from self._error
        with self._lock:
            return self._latest

    def get_hand_tracking_data(self):
        sample = self.get_latest()
        return sample["tracked_hands"] if sample else None

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("PICO SDK reader did not stop; SDK was not closed under a live read")
            self._sdk.close()
