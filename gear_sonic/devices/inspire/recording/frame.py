"""Validated synchronized frame for the native Inspire LeRobot dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import InspireMapping
from ..contract import ContractError, HandStatus, InspireHandState, hand_vector

from .depth import DepthFrame
from .schema import BODY_DOF, EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, MOTION_TOKEN_DOF, STATE_DOF


def camera_frame_timestamp_ns(camera_message: dict[str, Any]) -> int:
    """Return the source timestamp used to reject a reused camera frame."""
    value = camera_message.get("timestamps", {}).get("ego_view")
    if isinstance(value, bool):
        raise ContractError("camera ego_view timestamp must be a finite positive number")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("camera ego_view timestamp must be a finite positive number") from exc
    if not np.isfinite(seconds) or seconds <= 0:
        raise ContractError("camera ego_view timestamp must be a finite positive number")
    return int(seconds * 1_000_000_000)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ContractError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ContractError(f"{name} contains NaN or infinity")
    return array.copy()


def _timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ContractError(f"{name} must be a non-negative integer timestamp")
    return int(value)


@dataclass(frozen=True)
class CaptureSyncPolicy:
    max_age_ns: int = 100_000_000
    max_skew_ns: int = 100_000_000

    def __post_init__(self) -> None:
        if self.max_age_ns <= 0 or self.max_skew_ns <= 0:
            raise ValueError("capture freshness limits must be positive")

    def validate(
        self, *, now_ns: int, body_ns: int, camera_ns: int, target_ns: int, hand_ns: int | None = None
    ) -> None:
        now_ns = _timestamp(now_ns, "now_ns")
        samples = {
            "body": _timestamp(body_ns, "body_received_ns"),
            "camera": _timestamp(camera_ns, "camera_received_ns"),
            "teleop_target": _timestamp(target_ns, "teleop_target_ns"),
        }
        if hand_ns is not None:
            samples["hand"] = _timestamp(hand_ns, "hand_received_ns")
        for name, sample_ns in samples.items():
            age_ns = now_ns - sample_ns
            if age_ns < 0 or age_ns > self.max_age_ns:
                raise ContractError(f"{name} sample is stale or from the future: age={age_ns} ns")
        if max(samples.values()) - min(samples.values()) > self.max_skew_ns:
            raise ContractError("body/camera/teleop target samples exceed capture skew policy")


@dataclass(frozen=True)
class InspireLeRobotFrame:
    observation_state: np.ndarray
    observation_left_hand: np.ndarray
    observation_right_hand: np.ndarray
    projected_gravity: np.ndarray
    motion_token: np.ndarray
    action_left_hand: np.ndarray
    action_right_hand: np.ndarray
    teleop_smpl_joints: np.ndarray
    teleop_smpl_pose: np.ndarray
    teleop_body_quat_w: np.ndarray
    teleop_vr_3pt_position: np.ndarray
    teleop_vr_3pt_orientation: np.ndarray
    hand_gateway_healthy: np.ndarray
    hand_target_valid: np.ndarray
    hand_command_sent: np.ndarray
    ego_view: np.ndarray
    sample_monotonic_ns: int
    camera_source_timestamp_ns: int
    hand_command_sequence: int
    teleop_target_frame_index: int
    teleop_device_timestamp_ns: int
    body_joint_positions: np.ndarray
    body_joint_velocities: np.ndarray
    body_quaternion: np.ndarray
    intent_left_hand: np.ndarray | None = None
    intent_right_hand: np.ndarray | None = None
    hand_source_device_timestamp_ns: int = 0
    hand_submitted_monotonic_ns: int = 0
    depth: DepthFrame | None = None

    def __post_init__(self) -> None:
        for side in ("left", "right"):
            value = getattr(self, f"intent_{side}_hand")
            if value is None:
                value = getattr(self, f"action_{side}_hand")
            object.__setattr__(
                self, f"intent_{side}_hand", hand_vector(value, field_name=f"teleop.{side}_hand_intent")
            )
        for side in ("left", "right"):
            for prefix, key in (
                ("observation", f"observation.{side}_hand"),
                ("action", f"action.{side}_hand_joints"),
            ):
                field_name = f"{prefix}_{side}_hand"
                object.__setattr__(self, field_name, hand_vector(getattr(self, field_name), field_name=key))
        for field_name, size, export_name in (
            ("body_joint_positions", BODY_DOF, "g1_debug.body_q"),
            ("body_joint_velocities", BODY_DOF, "g1_debug.body_dq"),
            ("body_quaternion", 4, "g1_debug.base_quat"),
            ("observation_state", STATE_DOF, "observation.state"),
            ("projected_gravity", 3, "observation.projected_gravity"),
            ("motion_token", MOTION_TOKEN_DOF, "action.motion_token"),
            ("teleop_smpl_joints", 72, "teleop.smpl_joints"),
            ("teleop_smpl_pose", 63, "teleop.smpl_pose"),
            ("teleop_body_quat_w", 4, "teleop.body_quat_w"),
            ("teleop_vr_3pt_position", 9, "teleop.vr_3pt_position"),
            ("teleop_vr_3pt_orientation", 18, "teleop.vr_3pt_orientation"),
            ("hand_gateway_healthy", 1, "observation.hand_gateway_healthy"),
            ("hand_target_valid", 1, "teleop.hand_target_valid"),
            ("hand_command_sent", 1, "teleop.hand_command_sent"),
        ):
            object.__setattr__(
                self,
                field_name,
                _vector(getattr(self, field_name), size, export_name),
            )
        image = np.asarray(self.ego_view)
        expected = (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3)
        if image.shape != expected or image.dtype != np.uint8:
            raise ContractError(
                f"ego_view must have dtype uint8 and shape {expected}, got {image.dtype} {image.shape}"
            )
        object.__setattr__(self, "ego_view", np.array(image, copy=True))
        for name in (
            "sample_monotonic_ns",
            "camera_source_timestamp_ns",
            "hand_command_sequence",
            "teleop_target_frame_index",
            "teleop_device_timestamp_ns",
            "hand_source_device_timestamp_ns",
            "hand_submitted_monotonic_ns",
        ):
            object.__setattr__(self, name, _timestamp(getattr(self, name), name))

    def to_exporter_dict(self) -> dict[str, np.ndarray]:
        return {
            "observation.images.ego_view": self.ego_view,
            "observation.state": self.observation_state,
            "observation.left_hand": self.observation_left_hand,
            "observation.right_hand": self.observation_right_hand,
            "observation.projected_gravity": self.projected_gravity,
            "action.motion_token": self.motion_token,
            "teleop.left_hand_intent": self.intent_left_hand,
            "teleop.right_hand_intent": self.intent_right_hand,
            "action.left_hand_joints": self.action_left_hand,
            "action.right_hand_joints": self.action_right_hand,
            "teleop.smpl_joints": self.teleop_smpl_joints,
            "teleop.smpl_pose": self.teleop_smpl_pose,
            "teleop.body_quat_w": self.teleop_body_quat_w,
            "teleop.vr_3pt_position": self.teleop_vr_3pt_position,
            "teleop.vr_3pt_orientation": self.teleop_vr_3pt_orientation,
            "observation.hand_gateway_healthy": self.hand_gateway_healthy,
            "teleop.hand_target_valid": self.hand_target_valid,
            "teleop.hand_command_sent": self.hand_command_sent,
        }


def build_lerobot_frame(
    *,
    now_ns: int,
    body_received_ns: int,
    camera_received_ns: int,
    hand_received_ns: int,
    target_received_ns: int,
    body_message: dict[str, Any],
    camera_message: dict[str, Any],
    hand_state: InspireHandState,
    teleop_target: Any,
    robot_model: Any,
    mapping: InspireMapping,
    sync_policy: CaptureSyncPolicy,
) -> InspireLeRobotFrame:
    """Merge live sources without padding, clipping, or substituting missing values."""
    sync_policy.validate(
        now_ns=now_ns,
        body_ns=body_received_ns,
        camera_ns=camera_received_ns,
        target_ns=target_received_ns,
        hand_ns=hand_received_ns,
    )
    if hand_state.sequence == 0:
        raise ContractError("gateway has no Inspire measurement")

    body_q = _vector(body_message.get("body_q"), BODY_DOF, "g1_debug.body_q")
    base_quat = _vector(body_message.get("base_quat"), 4, "g1_debug.base_quat")
    token = _vector(body_message.get("token_state"), MOTION_TOKEN_DOF, "g1_debug.token_state")
    left_measured = mapping.validate_limits(hand_state.left_q, field_name="gateway.left_q", side="left")
    right_measured = mapping.validate_limits(hand_state.right_q, field_name="gateway.right_q", side="right")
    left_action = mapping.validate_limits(
        teleop_target.left_hand_native6,
        field_name="teleop.left_hand_native6",
        side="left",
    )
    right_action = mapping.validate_limits(
        teleop_target.right_hand_native6,
        field_name="teleop.right_hand_native6",
        side="right",
    )
    images = camera_message.get("images", {})
    if "ego_view" not in images:
        raise ContractError("camera message is missing images.ego_view")
    source_timestamp_ns = camera_frame_timestamp_ns(camera_message)

    state = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=left_measured,
        right_hand_actuated_joint_values=right_measured,
    )
    from gear_sonic.utils.data_collection.transforms import compute_projected_gravity, quat_to_rot6d

    gravity = compute_projected_gravity(base_quat)
    hand_age_ns = now_ns - _timestamp(hand_received_ns, "hand_gateway_ns")
    hand_gateway_healthy = 0 <= hand_age_ns <= sync_policy.max_age_ns and hand_state.status is HandStatus.ARMED
    return InspireLeRobotFrame(
        observation_state=np.asarray(state, dtype=np.float32),
        observation_left_hand=left_measured,
        observation_right_hand=right_measured,
        projected_gravity=gravity,
        motion_token=token,
        action_left_hand=left_action,
        action_right_hand=right_action,
        teleop_smpl_joints=teleop_target.smpl_joints,
        teleop_smpl_pose=teleop_target.smpl_pose,
        teleop_body_quat_w=teleop_target.body_quat_w,
        teleop_vr_3pt_position=teleop_target.vr_position,
        teleop_vr_3pt_orientation=quat_to_rot6d(teleop_target.vr_orientation).astype(np.float32),
        hand_gateway_healthy=np.array([hand_gateway_healthy], dtype=np.float32),
        hand_target_valid=np.array([teleop_target.hand_target_valid], dtype=np.float32),
        hand_command_sent=np.array([teleop_target.hand_command_sent], dtype=np.float32),
        ego_view=images["ego_view"],
        depth=DepthFrame.from_camera_message(camera_message),
        sample_monotonic_ns=now_ns,
        camera_source_timestamp_ns=source_timestamp_ns,
        hand_command_sequence=(hand_state.accepted_command_sequence or 0),
        teleop_target_frame_index=teleop_target.frame_index,
        teleop_device_timestamp_ns=teleop_target.pico_device_timestamp_ns,
        body_joint_positions=body_q,
        body_joint_velocities=body_message.get("body_dq"),
        body_quaternion=base_quat,
        intent_left_hand=teleop_target.left_hand_intent,
        intent_right_hand=teleop_target.right_hand_intent,
        hand_source_device_timestamp_ns=teleop_target.hand_source_device_timestamp_ns,
        hand_submitted_monotonic_ns=teleop_target.hand_submitted_monotonic_ns,
    )
