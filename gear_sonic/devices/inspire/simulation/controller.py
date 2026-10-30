"""Resolve named MuJoCo hand joints and apply guarded Inspire PD control."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ..config import InspireMapping, load_mapping
from ..contract import (
    HAND_DOF,
    HandStatus,
    InspireHandCommand,
    InspireHandState,
)
from ..safety import CommandGuard, SafetyConfig


@dataclass(frozen=True)
class SideLayout:
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray


@dataclass(frozen=True)
class InspireMujocoLayout:
    left: SideLayout
    right: SideLayout

    @classmethod
    def resolve(cls, model: Any, mapping: InspireMapping) -> "InspireMujocoLayout":
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - exercised by CLI dependency checks
            raise RuntimeError("MuJoCo is required; install gear_sonic[sim]") from exc

        def resolve_side(joints) -> SideLayout:
            joint_ids = np.asarray(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, item.mujoco_joint) for item in joints],
                dtype=np.int32,
            )
            actuator_ids = np.asarray(
                [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, item.mujoco_actuator) for item in joints],
                dtype=np.int32,
            )
            if np.any(joint_ids < 0) or np.any(actuator_ids < 0):
                raise ValueError("mapping contains a MuJoCo joint or actuator absent from the model")
            if len(set(joint_ids.tolist())) != HAND_DOF or len(set(actuator_ids.tolist())) != HAND_DOF:
                raise ValueError("mapping must resolve to six unique joints and actuators per hand")
            return SideLayout(
                joint_ids=joint_ids,
                qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32),
                dof_addresses=np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32),
                actuator_ids=actuator_ids,
            )

        return cls(left=resolve_side(mapping.left), right=resolve_side(mapping.right))

    def read(self, data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(data.qpos[self.left.qpos_addresses], dtype=np.float32).copy(),
            np.asarray(data.qpos[self.right.qpos_addresses], dtype=np.float32).copy(),
            np.asarray(data.qvel[self.left.dof_addresses], dtype=np.float32).copy(),
            np.asarray(data.qvel[self.right.dof_addresses], dtype=np.float32).copy(),
        )


@dataclass(frozen=True)
class SimControlConfig:
    kp: float = 2.0
    kd: float = 0.08
    effort_limit: float = 1.4

    def __post_init__(self) -> None:
        if self.kp <= 0 or self.kd < 0 or self.effort_limit <= 0:
            raise ValueError("invalid hand PD/effort configuration")


class InspireSimController:
    """Applies guarded native-6D commands to named MuJoCo actuators."""

    def __init__(
        self,
        model: Any,
        data: Any,
        *,
        mapping: InspireMapping | None = None,
        control: SimControlConfig | None = None,
        safety: SafetyConfig | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.mapping = mapping or load_mapping()
        self.layout = InspireMujocoLayout.resolve(model, self.mapping)
        self.control = control or SimControlConfig()
        self.guard = CommandGuard(self.mapping, safety)
        self._state_sequence = 0

    def arm(self) -> None:
        self.guard.arm()

    def submit(self, command: InspireHandCommand, *, now_ns: int | None = None) -> None:
        self.guard.accept(command, now_ns=time.monotonic_ns() if now_ns is None else now_ns)

    def step(self, *, now_ns: int | None = None) -> InspireHandState:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        status = self.guard.poll(now_ns=now_ns)
        left_q, right_q, left_dq, right_dq = self.layout.read(self.data)
        if status is HandStatus.ARMED and self.guard.last_command is not None:
            command = self.guard.last_command
            self._apply_pd(self.layout.left, command.left_q, left_q, left_dq)
            self._apply_pd(self.layout.right, command.right_q, right_q, right_dq)
        else:
            self.data.ctrl[self.layout.left.actuator_ids] = 0.0
            self.data.ctrl[self.layout.right.actuator_ids] = 0.0

        self._state_sequence += 1
        source_ns = self.guard.last_command.source_monotonic_ns if self.guard.last_command else 0
        return InspireHandState(
            sequence=self._state_sequence,
            source_monotonic_ns=source_ns,
            gateway_monotonic_ns=now_ns,
            status=status,
            left_q=left_q,
            right_q=right_q,
            left_dq=left_dq,
            right_dq=right_dq,
        )

    def _apply_pd(self, layout, target, q, dq) -> None:
        torque = self.control.kp * (target - q) - self.control.kd * dq
        self.data.ctrl[layout.actuator_ids] = np.clip(
            torque, -self.control.effort_limit, self.control.effort_limit
        )
