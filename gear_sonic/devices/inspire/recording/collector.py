"""Three-source synchronized collector for native Inspire LeRobot frames."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import time
from typing import Any

from ..contract import ContractError, InspireHandState

from .frame import CaptureSyncPolicy, build_lerobot_frame, camera_frame_timestamp_ns


@dataclass
class CaptureStats:
    requested_frames: int
    written_frames: int = 0
    camera_unique_frames: int = 0
    hand_sequence_repeats: int = 0
    teleop_target_unique_frames: int = 0
    hand_target_invalid_frames: int = 0
    hand_command_not_sent_frames: int = 0
    hand_gateway_unhealthy_frames: int = 0
    drops: Counter[str] = field(default_factory=Counter)
    started_monotonic_ns: int = 0
    completed_monotonic_ns: int = 0
    stop_reason: str = "frame_limit"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["drops"] = dict(sorted(self.drops.items()))
        value["duration_seconds"] = (
            (self.completed_monotonic_ns - self.started_monotonic_ns) / 1_000_000_000
            if self.completed_monotonic_ns
            else None
        )
        return value


class InspireLeRobotCollector:
    """Poll camera, official body debug, and Gateway state at a fixed rate."""

    def __init__(
        self,
        *,
        camera: Any,
        body_state: Any,
        hand_state: Any,
        teleop_target: Any,
        writer: Any,
        robot_model: Any,
        mapping: Any,
        fps: int = 50,
        sync_policy: CaptureSyncPolicy | None = None,
        clock_ns: Any = time.monotonic_ns,
        sleep: Any = time.sleep,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.camera = camera
        self.body_state = body_state
        self.hand_state = hand_state
        self.teleop_target = teleop_target
        self.writer = writer
        self.robot_model = robot_model
        self.mapping = mapping
        self.fps = fps
        self.sync_policy = sync_policy or CaptureSyncPolicy()
        self.clock_ns = clock_ns
        self.sleep = sleep

        self._camera_message: dict[str, Any] | None = None
        self._camera_received_ns = 0
        self._camera_source_ns = -1
        self._body_message: dict[str, Any] | None = None
        self._body_received_ns = 0
        self._hand_sample: InspireHandState | None = None
        self._hand_received_ns = 0
        self._last_seen_hand_command_sequence = -1
        self._last_written_hand_command_sequence = -1
        self._teleop_target_sample: Any | None = None
        self._last_seen_teleop_target_frame = -1

    @staticmethod
    def _receive_timed_out(exc: Exception) -> bool:
        return type(exc).__name__ in {"Again", "TimeoutError", "WouldBlock"}

    def _poll_sources(self, stats: CaptureStats) -> None:
        camera_message = self.camera.read()
        if camera_message is not None:
            source_ns = camera_frame_timestamp_ns(camera_message)
            if source_ns < self._camera_source_ns:
                raise ContractError(
                    f"camera source timestamp regressed: previous={self._camera_source_ns}, got={source_ns}"
                )
            if source_ns > self._camera_source_ns:
                self._camera_message = camera_message
                self._camera_source_ns = source_ns
                self._camera_received_ns = self.clock_ns()
                stats.camera_unique_frames += 1

        body_message = self.body_state.get_msg()
        if body_message is not None:
            self._body_message = body_message
            self._body_received_ns = self.clock_ns()

        try:
            timed_hand = self.hand_state.receive_timed()
        except Exception as exc:
            if not self._receive_timed_out(exc):
                raise
        else:
            hand_sample = timed_hand.state
            accepted = hand_sample.accepted_command_sequence
            if accepted is not None and accepted < self._last_seen_hand_command_sequence:
                raise ContractError(
                    "accepted hand command sequence regressed: "
                    f"previous={self._last_seen_hand_command_sequence}, got={accepted}"
                )
            if accepted is not None:
                self._last_seen_hand_command_sequence = accepted
            self._hand_sample = hand_sample
            self._hand_received_ns = timed_hand.received_monotonic_ns

        try:
            target_sample = self.teleop_target.receive_timed()
        except Exception as exc:
            if not self._receive_timed_out(exc):
                raise
        else:
            if target_sample.frame_index > self._last_seen_teleop_target_frame:
                stats.teleop_target_unique_frames += 1
                self._last_seen_teleop_target_frame = target_sample.frame_index
            self._teleop_target_sample = target_sample

    def _missing_source(self) -> str | None:
        if self._camera_message is None:
            return "missing_camera"
        if self._body_message is None:
            return "missing_body"
        if self._hand_sample is None:
            return "missing_hand"
        if self._teleop_target_sample is None:
            return "missing_teleop_target"
        return None

    def _append_current_frame(self, stats: CaptureStats) -> None:
        missing = self._missing_source()
        if missing is not None:
            stats.drops[missing] += 1
            return
        try:
            frame = build_lerobot_frame(
                now_ns=self.clock_ns(),
                body_received_ns=self._body_received_ns,
                camera_received_ns=self._camera_received_ns,
                hand_received_ns=self._hand_received_ns,
                target_received_ns=self._teleop_target_sample.received_monotonic_ns,
                body_message=self._body_message,
                camera_message=self._camera_message,
                hand_state=self._hand_sample,
                teleop_target=self._teleop_target_sample,
                robot_model=self.robot_model,
                mapping=self.mapping,
                sync_policy=self.sync_policy,
            )
        except ContractError as exc:
            stats.drops[f"contract:{str(exc).split(':', 1)[0]}"] += 1
            return
        if frame.hand_command_sequence == self._last_written_hand_command_sequence:
            stats.hand_sequence_repeats += 1
        self.writer.append(frame)
        if not bool(frame.hand_target_valid[0]):
            stats.hand_target_invalid_frames += 1
        if not bool(frame.hand_command_sent[0]):
            stats.hand_command_not_sent_frames += 1
        if not bool(frame.hand_gateway_healthy[0]):
            stats.hand_gateway_unhealthy_frames += 1
        self._last_written_hand_command_sequence = frame.hand_command_sequence
        stats.written_frames += 1

    def _sleep_until_next_tick(self, next_tick_ns: int, period_ns: int) -> int:
        next_tick_ns += period_ns
        remaining_ns = next_tick_ns - self.clock_ns()
        if remaining_ns > 0:
            self.sleep(remaining_ns / 1_000_000_000)
        elif remaining_ns < -period_ns:
            next_tick_ns = self.clock_ns()
        return next_tick_ns

    def collect(self, *, frames: int, timeout_s: float) -> CaptureStats:
        if frames < 1:
            raise ValueError("frames must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        stats = CaptureStats(requested_frames=frames, started_monotonic_ns=self.clock_ns())
        deadline_ns = stats.started_monotonic_ns + int(timeout_s * 1_000_000_000)
        period_ns = int(1_000_000_000 / self.fps)
        next_tick_ns = stats.started_monotonic_ns

        while stats.written_frames < frames:
            now_ns = self.clock_ns()
            if now_ns >= deadline_ns:
                raise TimeoutError(
                    f"captured {stats.written_frames}/{frames} frames before timeout; drops={dict(stats.drops)}"
                )
            self._poll_sources(stats)
            self._append_current_frame(stats)
            next_tick_ns = self._sleep_until_next_tick(next_tick_ns, period_ns)

        stats.completed_monotonic_ns = self.clock_ns()
        return stats

    def collect_for(self, seconds: float) -> CaptureStats:
        """Capture a wall-clock window, without extending it to replace dropped samples."""
        import math

        if not math.isfinite(seconds) or not 1 <= seconds <= 1800:
            raise ValueError("seconds must be finite and between 1 and 1800")
        stats = CaptureStats(requested_frames=math.ceil(seconds * self.fps), started_monotonic_ns=self.clock_ns())
        deadline = stats.started_monotonic_ns + int(seconds * 1_000_000_000)
        period = int(1_000_000_000 / self.fps)
        tick = stats.started_monotonic_ns
        while self.clock_ns() < deadline:
            self._poll_sources(stats)
            if self.clock_ns() >= deadline:
                break
            self._append_current_frame(stats)
            tick = self._sleep_until_next_tick(tick, period)
        stats.completed_monotonic_ns = self.clock_ns()
        stats.stop_reason = "duration_limit"
        if stats.written_frames < 40:
            raise RuntimeError(f"duration ended with only {stats.written_frames} valid frames; drops={dict(stats.drops)}")
        return stats


def record_episode(args, config):
    """Registered compatibility source: original read-only G1/Inspire RGB-D streams."""
    from contextlib import ExitStack
    from pathlib import Path
    from gear_sonic.utils.hand_control.registry import asset_path
    from ..config import hardware_settings, load_mapping
    from gear_sonic.devices.inspire.adapter import InspireAdapter
    # Validate registered dimensions/units against the retained native contract.
    # Construction opens no control session; this recorder remains read-only.
    InspireAdapter(config)
    config_path = config.path(config.options['hand_config'])
    output = Path(args.output).expanduser().resolve()
    settings = hardware_settings(config_path)
    mapping = load_mapping(config_path)
    CONTROL_FPS = config.registration.description.command_hz
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    from ..model import instantiate_g1_rh56dfx_robot_model
    from ..recording.collector import InspireLeRobotCollector
    from ..recording.depth import DepthPolicyCamera
    from ..recording.exporter import InspireLeRobotExporter
    from ..recording.hand_state import StateClient
    from ..recording.schema import REAL_CAPTURE_VALIDATION_LEVEL, SIM_VALIDATION_LEVEL
    from ..recording.teleop_target import TeleopTargetClient
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

    if args.wait_seconds:
        print(f"Waiting {args.wait_seconds:g} seconds before capture.", flush=True)
        time.sleep(args.wait_seconds)
    with ExitStack() as resources:
        camera = ComposedCameraClientSensor(
            server_ip=args.camera_host, port=args.camera_port, include_depth=args.depth != "off"
        )
        resources.callback(camera.close)
        camera = DepthPolicyCamera(camera, args.depth)
        body = ZMQStateSubscriber("127.0.0.1", 5557)
        resources.callback(body.close)
        hand = StateClient(settings["state_endpoint"])
        resources.callback(hand.close)
        target = TeleopTargetClient("tcp://127.0.0.1:5556")
        resources.callback(target.close)
        model = instantiate_g1_rh56dfx_robot_model(config_path=config_path)
        writer = InspireLeRobotExporter(
            output,
            robot_model=model,
            fps=CONTROL_FPS,
            task_prompt=args.task_prompt,
            config_path=config_path,
            validation_level=SIM_VALIDATION_LEVEL if args.sim else REAL_CAPTURE_VALIDATION_LEVEL,
            deployment=config, outcome=getattr(args, "outcome", "unlabeled"),
        )
        try:
            print("Recording started. Ctrl+C cancels this episode.", flush=True)
            collector = InspireLeRobotCollector(
                camera=camera,
                body_state=body,
                hand_state=hand,
                teleop_target=target,
                writer=writer,
                robot_model=model,
                mapping=mapping,
                fps=CONTROL_FPS,
            )
            stats = (collector.collect_for(args.seconds) if args.seconds is not None
                     else collector.collect(frames=args.frames, timeout_s=args.timeout))
            path = writer.commit(stats.as_dict())
            print(f"Saved {stats.written_frames} frames: {path}", flush=True)
        except BaseException as exc:
            staging = writer.abort(f"recording incomplete: {type(exc).__name__}: {exc}")
            print(f"Recording not published; diagnostic staging: {staging}", flush=True)
            raise
