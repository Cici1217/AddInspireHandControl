"""Calibrated OpenXR 26-joint tracking to Inspire RH56DFX native6.

This is deliberately separate from the legacy controller-trigger/Dex3 gesture
path.  It extracts six independent geometric features and refuses to produce a
command without an explicit per-hand open/closed calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import InspireMapping, load_mapping
from .contract import (
    HAND_DOF,
    JOINT_ORDER,
    hand_vector,
)
from gear_sonic.utils.teleop.hand_tracking import (
    HandTrackingError,
    OpenXRHandJoint as Joint,
    TrackedHandJoints,
)

OPTICAL_CALIBRATION_SCHEMA = "inspire.optical.calibration.v1"
OPTICAL_CALIBRATION_BUNDLE_SCHEMA = "inspire.optical.calibration.bundle.v1"
MIN_FEATURE_SPAN = 1e-3
# Existing RH56DFX combined closed-index/thumb pose from the hand example.
CANONICAL_THUMB_ROTATION_CLOSED_INDEX_RAD = 0.3
INDEX_CHANNEL = JOINT_ORDER.index("index")
THUMB_ROTATION_CHANNEL = JOINT_ORDER.index("thumb_rotation")

_FINGER_CHAINS = {
    "pinky": (
        Joint.LITTLE_METACARPAL,
        Joint.LITTLE_PROXIMAL,
        Joint.LITTLE_INTERMEDIATE,
        Joint.LITTLE_DISTAL,
        Joint.LITTLE_TIP,
    ),
    "ring": (
        Joint.RING_METACARPAL,
        Joint.RING_PROXIMAL,
        Joint.RING_INTERMEDIATE,
        Joint.RING_DISTAL,
        Joint.RING_TIP,
    ),
    "middle": (
        Joint.MIDDLE_METACARPAL,
        Joint.MIDDLE_PROXIMAL,
        Joint.MIDDLE_INTERMEDIATE,
        Joint.MIDDLE_DISTAL,
        Joint.MIDDLE_TIP,
    ),
    "index": (
        Joint.INDEX_METACARPAL,
        Joint.INDEX_PROXIMAL,
        Joint.INDEX_INTERMEDIATE,
        Joint.INDEX_DISTAL,
        Joint.INDEX_TIP,
    ),
}
_THUMB_CHAIN = (
    Joint.THUMB_METACARPAL,
    Joint.THUMB_PROXIMAL,
    Joint.THUMB_DISTAL,
    Joint.THUMB_TIP,
)
_REQUIRED_JOINTS = tuple(
    sorted(
        {
            int(Joint.WRIST),
            *(int(joint) for chain in _FINGER_CHAINS.values() for joint in chain),
            *(int(joint) for joint in _THUMB_CHAIN),
        }
    )
)


class OpticalRetargetingError(ValueError):
    """Raised when optical data cannot safely produce native6."""


def apply_thumb_index_collision_envelope(
    values: np.ndarray,
    *,
    mapping: InspireMapping,
    side: str,
) -> np.ndarray:
    """Keep full optical thumb yaw while preserving the TA's closed-fist limit.

    RH56DFX thumb rotation has a full 1.3 rad joint range.  The TA's hardware
    implementation uses 0.3 rad only for the pose where the index is fully
    closed because the two links otherwise interfere.  Interpolating those two
    documented endpoints gives continuous teleoperation without throwing away
    the sixth actuator.
    """

    if side not in ("left", "right"):
        raise OpticalRetargetingError("collision envelope side must be left or right")
    result = hand_vector(values, field_name=f"optical.{side}_collision_input")
    joints = mapping.left if side == "left" else mapping.right
    index_joint = joints[INDEX_CHANNEL]
    rotation_joint = joints[THUMB_ROTATION_CHANNEL]
    index_span = float(index_joint.upper - index_joint.lower)
    if index_span <= 0.0:
        raise OpticalRetargetingError("index mapping range must be positive")
    index_closed_fraction = float(
        np.clip(
            (result[INDEX_CHANNEL] - index_joint.lower) / index_span,
            0.0,
            1.0,
        )
    )
    full_rotation = float(rotation_joint.upper)
    closed_index_rotation = min(
        full_rotation,
        float(CANONICAL_THUMB_ROTATION_CLOSED_INDEX_RAD),
    )
    allowed_rotation = full_rotation + index_closed_fraction * (closed_index_rotation - full_rotation)
    result[THUMB_ROTATION_CHANNEL] = min(
        float(result[THUMB_ROTATION_CHANNEL]),
        allowed_rotation,
    )
    return result


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-7:
        raise OpticalRetargetingError(f"{name} is degenerate")
    return vector / norm


def _chain_curl(positions: np.ndarray, chain: tuple[Joint, ...], *, name: str) -> float:
    points = positions[np.asarray(chain, dtype=np.int64)]
    bones = np.diff(points, axis=0)
    units = np.stack([_unit(bone, name=f"{name} bone") for bone in bones])
    dots = np.sum(units[:-1] * units[1:], axis=1)
    return float(np.sum(np.arccos(np.clip(dots, -1.0, 1.0))))


def _thumb_rotation(positions: np.ndarray) -> float:
    wrist = positions[Joint.WRIST]
    middle = positions[Joint.MIDDLE_METACARPAL]
    index = positions[Joint.INDEX_METACARPAL]
    little = positions[Joint.LITTLE_METACARPAL]
    thumb_base = positions[Joint.THUMB_METACARPAL]
    thumb_tip = positions[Joint.THUMB_TIP]

    forward = _unit(middle - wrist, name="palm forward axis")
    lateral_raw = index - little
    lateral = _unit(
        lateral_raw - forward * float(np.dot(lateral_raw, forward)),
        name="palm lateral axis",
    )
    thumb = _unit(thumb_tip - thumb_base, name="thumb direction")
    forward_component = float(np.dot(thumb, forward))
    lateral_component = float(np.dot(thumb, lateral))
    if np.hypot(forward_component, lateral_component) <= 1e-7:
        raise OpticalRetargetingError("thumb direction has no component in the palm plane")
    return float(np.arctan2(lateral_component, forward_component))


def extract_inspire_features(hand: TrackedHandJoints) -> np.ndarray:
    """Return six geometry features in canonical Inspire joint order."""

    try:
        hand.require_valid(_REQUIRED_JOINTS)
    except HandTrackingError as exc:
        raise OpticalRetargetingError(str(exc)) from exc

    values = np.asarray(
        [_chain_curl(hand.positions, _FINGER_CHAINS[name], name=name) for name in JOINT_ORDER[:4]]
        + [
            _chain_curl(hand.positions, _THUMB_CHAIN, name="thumb_bend"),
            _thumb_rotation(hand.positions),
        ],
        dtype=np.float32,
    )
    if values.shape != (HAND_DOF,) or not np.all(np.isfinite(values)):
        raise OpticalRetargetingError("extracted hand features are not finite native6")
    return values


@dataclass(frozen=True)
class InspireOpticalCalibration:
    """One hand's explicit optical endpoints and canonical output endpoints."""

    calibration_id: str
    side: str
    open_features: np.ndarray
    closed_features: np.ndarray
    open_q: np.ndarray
    closed_q: np.ndarray

    def __post_init__(self) -> None:
        if not self.calibration_id.strip():
            raise OpticalRetargetingError("calibration_id must be non-empty")
        if self.side not in ("left", "right"):
            raise OpticalRetargetingError("calibration side must be left or right")
        open_features = hand_vector(self.open_features, field_name="open_features")
        closed_features = hand_vector(self.closed_features, field_name="closed_features")
        open_q = hand_vector(self.open_q, field_name="open_q")
        closed_q = hand_vector(self.closed_q, field_name="closed_q")
        spans = closed_features - open_features
        degenerate = np.flatnonzero(np.abs(spans) < MIN_FEATURE_SPAN)
        if degenerate.size:
            names = [JOINT_ORDER[index] for index in degenerate]
            raise OpticalRetargetingError(f"open/closed optical calibration has degenerate features: {names}")
        for field in (open_features, closed_features, open_q, closed_q):
            field.setflags(write=False)
        object.__setattr__(self, "open_features", open_features)
        object.__setattr__(self, "closed_features", closed_features)
        object.__setattr__(self, "open_q", open_q)
        object.__setattr__(self, "closed_q", closed_q)

    @classmethod
    def capture(
        cls,
        *,
        calibration_id: str,
        side: str,
        open_hand: TrackedHandJoints,
        closed_hand: TrackedHandJoints,
        open_q: Any,
        closed_q: Any,
        mapping: InspireMapping | None = None,
    ) -> "InspireOpticalCalibration":
        mapping = mapping or load_mapping()
        return cls(
            calibration_id=calibration_id,
            side=side,
            open_features=extract_inspire_features(open_hand),
            closed_features=extract_inspire_features(closed_hand),
            open_q=mapping.validate_limits(open_q, field_name="open_q", side=side),
            closed_q=mapping.validate_limits(closed_q, field_name="closed_q", side=side),
        )

    def retarget(self, hand: TrackedHandJoints, *, mapping: InspireMapping | None = None) -> np.ndarray:
        mapping = mapping or load_mapping()
        features = extract_inspire_features(hand)
        fraction = np.clip(
            (features - self.open_features) / (self.closed_features - self.open_features),
            0.0,
            1.0,
        )
        result = self.open_q + fraction * (self.closed_q - self.open_q)
        result = apply_thumb_index_collision_envelope(
            result,
            mapping=mapping,
            side=self.side,
        )
        return mapping.validate_limits(result, field_name=f"optical.{self.side}_hand_joints", side=self.side)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPTICAL_CALIBRATION_SCHEMA,
            "calibration_id": self.calibration_id,
            "side": self.side,
            "joint_order": list(JOINT_ORDER),
            "open_features": self.open_features.tolist(),
            "closed_features": self.closed_features.tolist(),
            "open_q": self.open_q.tolist(),
            "closed_q": self.closed_q.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InspireOpticalCalibration":
        if payload.get("schema") != OPTICAL_CALIBRATION_SCHEMA:
            raise OpticalRetargetingError("unknown optical calibration schema")
        if tuple(payload.get("joint_order", ())) != JOINT_ORDER:
            raise OpticalRetargetingError("optical calibration joint_order is not native6")
        return cls(
            calibration_id=str(payload.get("calibration_id", "")),
            side=str(payload.get("side", "")),
            open_features=payload.get("open_features"),
            closed_features=payload.get("closed_features"),
            open_q=payload.get("open_q"),
            closed_q=payload.get("closed_q"),
        )


@dataclass(frozen=True)
class InspireOpticalCalibrationBundle:
    """Versioned two-hand calibration used by one PICO native6 session."""

    calibration_id: str
    created_realtime_ns: int
    tracker_source: str
    samples_per_pose: int
    left: InspireOpticalCalibration
    right: InspireOpticalCalibration

    def __post_init__(self) -> None:
        if not self.calibration_id.strip():
            raise OpticalRetargetingError("bundle calibration_id must be non-empty")
        if self.created_realtime_ns < 0:
            raise OpticalRetargetingError("created_realtime_ns must be non-negative")
        if not self.tracker_source.strip():
            raise OpticalRetargetingError("tracker_source must be non-empty")
        if self.samples_per_pose <= 0:
            raise OpticalRetargetingError("samples_per_pose must be positive")
        if self.left.side != "left" or self.right.side != "right":
            raise OpticalRetargetingError("bundle calibrations are assigned to the wrong side")
        if self.left.calibration_id != self.calibration_id:
            raise OpticalRetargetingError("left calibration_id differs from bundle")
        if self.right.calibration_id != self.calibration_id:
            raise OpticalRetargetingError("right calibration_id differs from bundle")

    @classmethod
    def from_independent_feature_poses(
        cls,
        *,
        calibration_id: str,
        created_realtime_ns: int,
        samples_per_pose: int,
        left_open: Any,
        right_open: Any,
        left_fingers_closed: Any,
        right_fingers_closed: Any,
        left_thumb_bent: Any,
        right_thumb_bent: Any,
        left_thumb_rotated: Any,
        right_thumb_rotated: Any,
        tracker_source: str = "isaacteleop.hand_tracker.openxr26",
        mapping: InspireMapping | None = None,
    ) -> "InspireOpticalCalibrationBundle":
        """Assemble six independent feature spans from four explicit poses.

        The four finger curl endpoints come from ``fingers_closed``. Thumb bend
        and rotation each come from their own capture, preventing thumb motion
        from contaminating the other thumb channel.
        """

        mapping = mapping or load_mapping()
        left_open_vector = hand_vector(left_open, field_name="left_open")
        right_open_vector = hand_vector(right_open, field_name="right_open")
        left_closed = np.concatenate(
            [
                hand_vector(left_fingers_closed, field_name="left_fingers_closed")[:4],
                hand_vector(left_thumb_bent, field_name="left_thumb_bent")[4:5],
                hand_vector(left_thumb_rotated, field_name="left_thumb_rotated")[5:6],
            ]
        )
        right_closed = np.concatenate(
            [
                hand_vector(right_fingers_closed, field_name="right_fingers_closed")[:4],
                hand_vector(right_thumb_bent, field_name="right_thumb_bent")[4:5],
                hand_vector(right_thumb_rotated, field_name="right_thumb_rotated")[5:6],
            ]
        )
        left_q_open = np.asarray([joint.lower for joint in mapping.left], dtype=np.float32)
        right_q_open = np.asarray([joint.lower for joint in mapping.right], dtype=np.float32)
        # Each feature was captured independently, so each endpoint uses its
        # full actuator range.  In particular thumb rotation reaches 1.3 rad
        # while the index is open.  retarget() applies the TA-derived
        # index/thumb collision envelope for combined poses.
        left_q_closed = np.asarray([joint.upper for joint in mapping.left], dtype=np.float32)
        right_q_closed = np.asarray([joint.upper for joint in mapping.right], dtype=np.float32)
        return cls(
            calibration_id=calibration_id,
            created_realtime_ns=int(created_realtime_ns),
            tracker_source=tracker_source,
            samples_per_pose=int(samples_per_pose),
            left=InspireOpticalCalibration(
                calibration_id=calibration_id,
                side="left",
                open_features=left_open_vector,
                closed_features=left_closed,
                open_q=left_q_open,
                closed_q=left_q_closed,
            ),
            right=InspireOpticalCalibration(
                calibration_id=calibration_id,
                side="right",
                open_features=right_open_vector,
                closed_features=right_closed,
                open_q=right_q_open,
                closed_q=right_q_closed,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OPTICAL_CALIBRATION_BUNDLE_SCHEMA,
            "calibration_id": self.calibration_id,
            "created_realtime_ns": self.created_realtime_ns,
            "tracker_source": self.tracker_source,
            "samples_per_pose": self.samples_per_pose,
            "joint_order": list(JOINT_ORDER),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InspireOpticalCalibrationBundle":
        if payload.get("schema") != OPTICAL_CALIBRATION_BUNDLE_SCHEMA:
            raise OpticalRetargetingError("unknown optical calibration bundle schema")
        if tuple(payload.get("joint_order", ())) != JOINT_ORDER:
            raise OpticalRetargetingError("optical calibration bundle joint_order is not native6")
        return cls(
            calibration_id=str(payload.get("calibration_id", "")),
            created_realtime_ns=int(payload.get("created_realtime_ns", -1)),
            tracker_source=str(payload.get("tracker_source", "")),
            samples_per_pose=int(payload.get("samples_per_pose", 0)),
            left=InspireOpticalCalibration.from_dict(payload.get("left", {})),
            right=InspireOpticalCalibration.from_dict(payload.get("right", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "InspireOpticalCalibrationBundle":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, Mapping):
            raise OpticalRetargetingError("optical calibration bundle must be a JSON object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination


def safe_seed(values, mapping, side):
    """Seed from measured joints, tolerating only numeric round-off at joint limits.

    The optical envelope constrains requested targets, not the measured startup
    pose. Preserve that pose for the first hold, as in the original PICO path.
    """
    joints = getattr(mapping, side)
    lower = np.array([j.lower for j in joints])
    upper = np.array([j.upper for j in joints])
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("invalid measured hand seed")
    if np.any(values < lower - 1e-6) or np.any(values > upper + 1e-6):
        raise ValueError("measured seed is outside joint limits")
    return np.clip(values, lower, upper)


def slew_target(previous, desired, *, mapping, side, elapsed_s):
    """Slew toward the optical envelope without jumping from measured feedback.

    One alpha keeps an in-envelope pose inside and monotonically reduces any
    initial envelope excess while respecting every joint's velocity limit.
    """
    if not np.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("slew elapsed time must be positive and finite")
    desired = mapping.validate_limits(desired, field_name="optical intent", side=side)
    desired = apply_thumb_index_collision_envelope(desired, mapping=mapping, side=side)
    previous = safe_seed(previous, mapping, side)
    delta = desired - previous
    limits = np.array([j.velocity_limit for j in getattr(mapping, side)])
    moving = abs(delta) > 1e-12
    alpha = min(1.0, float(np.min(0.8 * limits[moving] * elapsed_s / abs(delta[moving])))) if moving.any() else 1.0
    target = previous + alpha * delta
    return mapping.validate_limits(target, field_name="submitted hand target", side=side)
