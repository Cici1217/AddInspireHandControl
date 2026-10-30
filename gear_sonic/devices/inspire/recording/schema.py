"""GR00T LeRobot schema for the native 41-DoF G1 + Inspire embodiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..config import DEFAULT_MAPPING_PATH
from ..contract import HAND_DOF, JOINT_ORDER


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


BODY_DOF = 29
MOTION_TOKEN_DOF = 64
STATE_DOF = 41

EGO_VIEW_HEIGHT = 480
EGO_VIEW_WIDTH = 640
CONTROL_FPS = 50
LEROBOT_CAPTURE_SCHEMA = "inspire.lerobot.capture.v4"
LEROBOT_TIMING_SCHEMA = "inspire.lerobot.timing.v3"
LEROBOT_TIMING_PATH = "meta/inspire_timing.npz"
TIMING_FIELDS = (
    "sample_monotonic_ns",
    "camera_source_timestamp_ns",
    "hand_command_sequence",
    "teleop_target_frame_index",
    "teleop_device_timestamp_ns",
    "hand_source_device_timestamp_ns",
    "hand_submitted_monotonic_ns",
)
SIM_VALIDATION_LEVEL = "SIM_ONLY"
REAL_CAPTURE_VALIDATION_LEVEL = "REAL_CAPTURE_READ_ONLY"
CAPTURE_VALIDATION_LEVELS = frozenset((SIM_VALIDATION_LEVEL, REAL_CAPTURE_VALIDATION_LEVEL))

BODY_STATE_GROUPS = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
)
HAND_STATE_GROUPS = ("left_hand", "right_hand")


def _contiguous_group_slice(robot_model: Any, group: str) -> dict[str, int]:
    indices = list(robot_model.get_joint_group_indices(group))
    if not indices:
        raise ValueError(f"RobotModel group {group!r} is empty")
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(f"RobotModel group {group!r} is not contiguous: {indices}")
    return {"start": indices[0], "end": indices[-1] + 1}


def validate_robot_model(robot_model: Any) -> None:
    """Validate the exact model properties the LeRobot profile depends on."""
    if robot_model.num_joints != STATE_DOF:
        raise ValueError(f"Inspire LeRobot RobotModel must have {STATE_DOF} joints")
    if len(robot_model.joint_names) != STATE_DOF:
        raise ValueError("RobotModel joint_names does not match num_joints")
    for group in BODY_STATE_GROUPS:
        _contiguous_group_slice(robot_model, group)

    for side in ("left", "right"):
        indices = list(robot_model.get_hand_actuated_joint_indices(side))
        if len(indices) != HAND_DOF:
            raise ValueError(f"RobotModel {side}_hand must contain {HAND_DOF} actuated joints")
        names = [robot_model.joint_names[index] for index in indices]
        expected = [f"{side}_inspire_{joint}_joint" for joint in JOINT_ORDER]
        if names != expected:
            raise ValueError(
                f"RobotModel {side}_hand actuated order does not match Inspire domain order: {names} != {expected}"
            )


def lerobot_features(robot_model: Any) -> dict[str, dict[str, Any]]:
    """The current Inspire episode columns; full state follows RobotModel order."""
    validate_robot_model(robot_model)
    names = {
        "observation.state": list(robot_model.joint_names),
        "teleop.left_hand_intent": list(JOINT_ORDER),
        "teleop.right_hand_intent": list(JOINT_ORDER),
        "observation.left_hand": list(JOINT_ORDER),
        "observation.right_hand": list(JOINT_ORDER),
        "observation.projected_gravity": ["gravity_x", "gravity_y", "gravity_z"],
        "observation.hand_gateway_healthy": ["hand_gateway_healthy"],
        "action.motion_token": [f"token_{i}" for i in range(MOTION_TOKEN_DOF)],
        "action.left_hand_joints": list(JOINT_ORDER),
        "action.right_hand_joints": list(JOINT_ORDER),
        "teleop.smpl_joints": [f"smpl_joint_{i}" for i in range(72)],
        "teleop.smpl_pose": [f"smpl_pose_{i}" for i in range(63)],
        "teleop.body_quat_w": ["body_qw", "body_qx", "body_qy", "body_qz"],
        "teleop.vr_3pt_position": [f"{part}_{axis}" for part in ("lwrist", "rwrist", "neck") for axis in "xyz"],
        "teleop.vr_3pt_orientation": [f"vr_rotation_6d_{i}" for i in range(18)],
        "teleop.hand_target_valid": ["hand_target_valid"],
        "teleop.hand_command_sent": ["hand_command_sent"],
    }
    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        **{
            key: {"dtype": "float32", "shape": (len(columns),), "names": columns} for key, columns in names.items()
        },
    }


def lerobot_modality_config(robot_model: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Keep the existing UNITREE_G1_SONIC modality and transforms."""
    validate_robot_model(robot_model)
    state = {
        group: {**_contiguous_group_slice(robot_model, group), "original_key": "observation.state"}
        for group in BODY_STATE_GROUPS
    }
    for name, size in (("left_hand", HAND_DOF), ("right_hand", HAND_DOF), ("projected_gravity", 3)):
        state[name] = {"start": 0, "end": size, "original_key": f"observation.{name}"}
    action = {}
    for name, size, key in (
        ("motion_token", MOTION_TOKEN_DOF, "action.motion_token"),
        ("smpl_joints", 72, "teleop.smpl_joints"),
        ("smpl_pose", 63, "teleop.smpl_pose"),
        ("body_quat_w", 4, "teleop.body_quat_w"),
        ("vr_3pt_position", 9, "teleop.vr_3pt_position"),
        ("vr_3pt_orientation", 18, "teleop.vr_3pt_orientation"),
        ("left_hand_joints", HAND_DOF, "action.left_hand_joints"),
        ("right_hand_joints", HAND_DOF, "action.right_hand_joints"),
    ):
        action[name] = {"start": 0, "end": size, "original_key": key}
    action["body_quat_w"]["rotation_type"] = "quaternion"
    action["vr_3pt_orientation"]["rotation_type"] = "rotation_6d"
    return {
        "state": state,
        "action": action,
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def capture_identity(
    *, fps: int, validation_level: str = SIM_VALIDATION_LEVEL, config_path=DEFAULT_MAPPING_PATH
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if validation_level not in CAPTURE_VALIDATION_LEVELS:
        raise ValueError(f"unsupported capture validation level: {validation_level!r}")
    return {
        "capture_schema": LEROBOT_CAPTURE_SCHEMA,
        "validation_level": validation_level,
        "embodiment_tag": "UNITREE_G1_SONIC",
        "fps": fps,
        "state_dof": STATE_DOF,
        "body_dof": BODY_DOF,
        "hand_dof_per_side": HAND_DOF,
        "motion_token_dof": MOTION_TOKEN_DOF,
        "joint_order": list(JOINT_ORDER),
        "mapping_sha256": file_sha256(config_path),
    }


def validate_capture_identity(identity: Any, *, config_path=DEFAULT_MAPPING_PATH) -> None:
    """Verify this capture uses the selected Inspire mapping."""
    if not isinstance(identity, dict):
        raise ValueError("capture sidecar must be a JSON object")
    if identity.get("capture_schema") != LEROBOT_CAPTURE_SCHEMA:
        raise ValueError("capture sidecar has an unknown schema")
    if identity.get("validation_level") not in CAPTURE_VALIDATION_LEVELS:
        raise ValueError("capture sidecar has an unknown validation level")
    if identity.get("joint_order") != list(JOINT_ORDER):
        raise ValueError("capture sidecar native6 joint order is incompatible")
    if identity.get("hand_dof_per_side") != HAND_DOF:
        raise ValueError("capture sidecar hand DoF is incompatible")
    if identity.get("mapping_sha256") != file_sha256(config_path):
        raise ValueError("capture sidecar mapping_sha256 does not match the active native6 contract")
