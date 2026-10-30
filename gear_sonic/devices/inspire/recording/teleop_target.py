"""Read the exact PICO body and optical-hand targets mirrored by the manager."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import msgpack
import msgpack_numpy as mnp
import numpy as np

from ..contract import HAND_DOF, ContractError

TELEOP_TARGET_TOPIC = "teleop_target"


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,):
        raise ContractError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ContractError(f"{name} contains NaN or infinity")
    return result.copy()


def _scalar(value: Any, name: str) -> Any:
    result = np.asarray(value)
    if result.shape != (1,):
        raise ContractError(f"{name} must have shape (1,), got {result.shape}")
    return result[0]


@dataclass(frozen=True)
class TeleopTargetSample:
    frame_index: int
    pico_device_timestamp_ns: int
    smpl_pose: np.ndarray
    smpl_joints: np.ndarray
    body_quat_w: np.ndarray
    vr_position: np.ndarray
    vr_orientation: np.ndarray
    left_hand_native6: np.ndarray
    right_hand_native6: np.ndarray
    hand_target_valid: bool
    hand_command_sent: bool
    received_monotonic_ns: int
    left_hand_intent: np.ndarray | None = None
    right_hand_intent: np.ndarray | None = None
    hand_source_device_timestamp_ns: int = 0
    hand_submitted_monotonic_ns: int = 0

    @classmethod
    def from_wire(cls, message: dict[str, Any], *, received_monotonic_ns: int):
        frame_index = int(_scalar(message.get("frame_index"), "frame_index"))
        device_ns = int(
            _scalar(
                message.get("pico_device_timestamp_ns"),
                "pico_device_timestamp_ns",
            )
        )
        hand_device_ns = int(
            _scalar(message.get("hand_source_device_timestamp_ns"), "hand_source_device_timestamp_ns")
        )
        hand_submitted_ns = int(_scalar(message.get("hand_submitted_monotonic_ns"), "hand_submitted_monotonic_ns"))
        if min(frame_index, device_ns, received_monotonic_ns, hand_device_ns, hand_submitted_ns) < 0:
            raise ContractError("teleop target indices and timestamps must be non-negative")
        return cls(
            frame_index=frame_index,
            pico_device_timestamp_ns=device_ns,
            smpl_pose=_vector(message.get("smpl_pose"), 63, "teleop.smpl_pose"),
            smpl_joints=_vector(
                np.asarray(message.get("smpl_joints")).reshape(-1),
                72,
                "teleop.smpl_joints",
            ),
            body_quat_w=_vector(message.get("body_quat_w"), 4, "teleop.body_quat_w"),
            vr_position=_vector(message.get("vr_position"), 9, "teleop.vr_3pt_position"),
            vr_orientation=_vector(message.get("vr_orientation"), 12, "teleop.vr_3pt_orientation_quat"),
            left_hand_native6=_vector(message.get("left_hand_native6"), HAND_DOF, "teleop.left_hand_native6"),
            right_hand_native6=_vector(message.get("right_hand_native6"), HAND_DOF, "teleop.right_hand_native6"),
            hand_target_valid=bool(_scalar(message.get("native6_target_valid"), "native6_target_valid")),
            hand_command_sent=bool(_scalar(message.get("native6_command_sent"), "native6_command_sent")),
            received_monotonic_ns=int(received_monotonic_ns),
            left_hand_intent=_vector(message.get("left_hand_intent"), HAND_DOF, "left intent"),
            right_hand_intent=_vector(message.get("right_hand_intent"), HAND_DOF, "right intent"),
            hand_source_device_timestamp_ns=int(
                _scalar(message.get("hand_source_device_timestamp_ns"), "hand_source_device_timestamp_ns")
            ),
            hand_submitted_monotonic_ns=int(
                _scalar(message.get("hand_submitted_monotonic_ns"), "hand_submitted_monotonic_ns")
            ),
        )


class TeleopTargetClient:
    """Conflated SUB client for the manager's read-only target mirror."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556", *, timeout_ms: int = 2):
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        try:
            import zmq
        except ModuleNotFoundError as exc:
            raise RuntimeError("teleop target transport requires pyzmq") from exc
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, TELEOP_TARGET_TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._last_frame_index = -1

    def receive_timed(self) -> TeleopTargetSample:
        raw = self._socket.recv()
        received_ns = time.monotonic_ns()
        sample = TeleopTargetSample.from_wire(
            msgpack.unpackb(raw[len(TELEOP_TARGET_TOPIC) :], raw=False, object_hook=mnp.decode),
            received_monotonic_ns=received_ns,
        )
        if sample.frame_index < self._last_frame_index:
            raise ContractError(
                f"teleop target frame index regressed: previous={self._last_frame_index}, got={sample.frame_index}"
            )
        if sample.frame_index == self._last_frame_index:
            raise TimeoutError("repeated teleop target; local receipt was not refreshed")
        self._last_frame_index = sample.frame_index
        return sample

    def close(self) -> None:
        self._socket.close()
        self._context.term()
