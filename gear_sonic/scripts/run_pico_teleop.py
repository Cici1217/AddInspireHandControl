"""Registered end-effector PICO teleoperation with the original SONIC body streamer."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np

from gear_sonic.utils.hand_control.interface import EffectorError
from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment, registrations
from gear_sonic.utils.teleop.end_effector import OpticalSession
from gear_sonic.utils.teleop.hand_tracking import OpticalPicoReader

DEFAULT_CALIBRATION = "outputs/inspire_calibrations/pico_hands.json"
# Separate source disconnect from the hand gateway's 0.3 s feedback freshness.
# The shared upstream PicoReader uses the same disconnect interval.
BODY_TRACKING_TIMEOUT_S = 5.0


class TargetMirror:
    """Mirror the body sample, optical intent and latest submitted hand target.

    The hand loop runs independently. Its source device timestamp and actual
    local submission timestamp explicitly identify which hand sample was used.
    The Pose packet keeps seven zero legacy fields; Inspire never reads them.
    """

    def __init__(self, session, wire_adapter=None):
        self.session = session
        self.wire_adapter = wire_adapter
        self.sent_frames = 0
        self._hand_error = None

    def __call__(self, sample, data):
        for key in ("smpl_pose", "smpl_joints", "body_quat_w", "vr_position", "vr_orientation"):
            if not np.all(np.isfinite(data[key])):
                raise ValueError(f"non-finite body target: {key}")
        self.sent_frames += 1
        try:
            submitted = self.session.latest()
        except RuntimeError as exc:
            # Match the original PICO manager: hand failure must not stop body poses.
            # The hand worker closes its backend; no stale target is mirrored as live.
            error = str(exc)
            if error != self._hand_error:
                print(f"PICO hands stopped; body continues: {error}", flush=True)
                self._hand_error = error
            return None
        if submitted is None:
            raise RuntimeError("no submitted hand target")
        valid = True
        try:
            intent = self.session.retarget(sample)
        except EffectorError as exc:
            if exc.category not in ("tracking_unavailable", "retargeting"):
                raise
            # A missing intent does not become a fabricated zero action.
            intent = submitted.receipt.action
            valid = False
        payload = {
            "frame_index": np.array([data["frame_index"][-1]], dtype=np.int64),
            "pico_device_timestamp_ns": np.array([sample["timestamp_ns"]], dtype=np.int64),
            "hand_source_device_timestamp_ns": np.array([submitted.receipt.action.source_timestamp_ns], dtype=np.int64),
            "hand_submitted_monotonic_ns": np.array([submitted.receipt.submitted_monotonic_ns], dtype=np.int64),
            "smpl_pose": data["smpl_pose"][-1].reshape(-1),
            "smpl_joints": data["smpl_joints"][-1].reshape(-1),
            "body_quat_w": data["body_quat_w"][-1],
            "vr_position": data["vr_position"],
            "vr_orientation": data["vr_orientation"],
            "end_effector_action": {mount: np.asarray(v) for mount, v in submitted.receipt.action.values.items()},
            "end_effector_intent": {mount: np.asarray(v) for mount, v in intent.values.items()},
            "end_effector_tracking_valid": bool(valid and submitted.tracking_valid),
            "end_effector_sequence": submitted.receipt.sequence,
        }
        if self.wire_adapter is not None:
            payload.update(self.wire_adapter.legacy_target_fields(submitted, intent, valid))

        return b"teleop_target" + msgpack.packb(payload, default=mnp.encode, use_bin_type=True)


class BodyStartup:
    """Follow the existing manager: planner Start, then pose after control output.

    g1_debug is published from the C++ CONTROL loop. A fresh, complete state
    after the Start request is the available startup receipt, not a pose ACK.
    """

    def __init__(self, socket, feedback, *, clock=time.monotonic, ready_pose=None):
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message, build_planner_message

        self.socket, self.feedback, self.clock = socket, feedback, clock
        self.command = build_command_message
        self.idle = build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0)
        if ready_pose is not None:
            from gear_sonic.utils.inference.ready_pose import UPPER_ORDER
            self.idle = build_planner_message(0, [0.,0.,0.], [1.,0.,0.],
                upper_body_position=np.asarray(ready_pose)[list(UPPER_ORDER)], upper_body_velocity=np.zeros(17))
        self.last_feedback = None
        self.state = None
        self.started_at = None
        self.pose_selected = False
        self.last_idle = float("-inf")

    def step(self, *, select_pose=True):
        if self.pose_selected:
            return
        now = self.clock()
        if self.started_at is None:
            # Discard state queued before this Start request.
            while self.feedback() is not None:
                pass
            self.socket.send(self.idle)
            self.socket.send(self.command(start=True, stop=False, planner=True))
            self.started_at = now
            self.last_idle = now
            return
        if self.last_feedback is None and now - self.started_at > 10:
            raise TimeoutError("no fresh body CONTROL output after planner Start")
        if now - self.last_idle >= 0.02:
            self.socket.send(self.idle)
            self.last_idle = now
        state = self.feedback()
        if state is None and self.last_feedback is not None and now-self.last_feedback > .5:
            raise TimeoutError('body feedback stopped while holding preparation pose')
        if state is not None:
            body = np.asarray(state.get("body_q"))
            token = np.asarray(state.get("token_state"))
            if (
                body.shape != (29,)
                or token.shape != (64,)
                or not np.all(np.isfinite(body))
                or not np.all(np.isfinite(token))
            ):
                raise ValueError("incomplete body CONTROL output during PICO startup")
            self.state, self.last_feedback = state, now
            if select_pose:
                self.socket.send(self.command(start=True, stop=False, planner=False))
                self.pose_selected = True

    def stop(self):
        # A mode switch on Stop re-enters planner initialization in ZMQManager.
        # Keep the current mode so shutdown does not request a new planner motion.
        for _ in range(3):
            self.socket.send(self.command(start=False, stop=True, planner=not self.pose_selected))
            time.sleep(0.02)


def build_parser():
    parser = argparse.ArgumentParser(description="PICO teleoperation for a registered end effector")
    parser.add_argument("--device", default="inspire")
    parser.add_argument("--scene", help="Custom standalone MuJoCo scene XML")
    parser.add_argument("--view", action="store_true", help="Open a standalone end-effector simulation window")
    parser.add_argument("--deployment", help="Local deployment JSON; overrides device/backend")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--sim", action="store_true", help="Connect to the shared simulation hand gateway")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--hand-config", help="Legacy Inspire configuration override")
    parser.add_argument("--ready-pose-values", help="29 quoted body angles; wait in preparation before teleop")
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def run(args):
    devices = registrations(args.registry_dir)
    if args.device not in devices and not args.deployment:
        raise EffectorError("configuration", f"unregistered device: {args.device}")
    mode = "sim" if args.sim else "real"
    integration_options = devices[args.device].integration if args.device in devices else {}
    backend = integration_options.get(f"default_{mode}_backend", mode)
    options = {"calibration": args.calibration}
    if args.hand_config:
        options["hand_config"] = args.hand_config
    config = deployment(args.device, backend, config_path=args.deployment,
                        registry_dir=args.registry_dir, options=options)
    if getattr(args, 'scene', None):
        from gear_sonic.utils.mujoco_sim.scenes import standalone_scene
        config = standalone_scene(config, args.scene)
    spec = config.registration.check_dependencies(config.backend)
    if "teleop" not in spec["capabilities"]:
        raise EffectorError("capability", "selected backend does not support teleoperation")
    ready_values = getattr(args, 'ready_pose_values', None)
    if ready_values:
        from gear_sonic.utils.inference.ready_pose import parse_ready_pose
        parse_ready_pose(ready_values)
        if 'body' not in spec['capabilities']:
            raise ValueError('ready pose requires a G1 body')
    if args.dry_run:
        print(json.dumps(config.snapshot() | {"control_connections_opened": False}))
        return
    from gear_sonic.utils.hand_control.management import DeviceLease
    with DeviceLease(config, 'pico_teleoperation'):
        return _run_session(args, config, spec)


def _run_session(args, config, spec):
    mapper = config.retargeter()
    if "body" not in spec["capabilities"]:
        from types import SimpleNamespace
        from gear_sonic.scripts.run_end_effector import run_optical
        return run_optical(SimpleNamespace(record=None, seconds=None, view=args.view), config, retargeter=mapper)
    integration = config.body_integration()
    # Keep optional SDK, Pinocchio and body processing imports out of --help/dry-run.
    import zmq

    from gear_sonic.scripts.pico_manager_thread_server import PoseStreamer, ThreePointPose

    reader = OpticalPicoReader()
    body_started = False
    with ExitStack() as resources:
        context = zmq.Context()
        resources.callback(context.term)
        socket = context.socket(zmq.PUB)
        resources.callback(socket.close, 0)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 3)
        socket.bind("tcp://*:5556")
        state_socket = context.socket(zmq.SUB)
        resources.callback(state_socket.close, 0)
        state_socket.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
        state_socket.setsockopt(zmq.CONFLATE, 1)
        state_socket.connect("tcp://127.0.0.1:5557")

        def feedback():
            try:
                raw = state_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return None
            return msgpack.unpackb(raw[len(b"g1_debug") :], raw=False, object_hook=mnp.decode)

        from gear_sonic.utils.inference.ready_pose import parse_ready_pose, upper_body_error
        ready_values = getattr(args, 'ready_pose_values', None)
        ready_pose = parse_ready_pose(ready_values) if ready_values else None
        startup = BodyStartup(socket, feedback, ready_pose=ready_pose)

        def stop_body():
            if body_started:
                startup.stop()

        # Register before starting so partial SDK startup also has a cleanup path.
        resources.callback(reader.stop)
        reader.start()
        deadline = time.monotonic() + 30
        while reader.get_latest() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("no PICO body frame; check headset body tracking and PC Service")
            time.sleep(0.01)
        mapper.prepare(reader)
        session = OpticalSession(reader, config, retargeter=mapper)
        model = integration.body_model()
        three_point = ThreePointPose(robot_model=model)
        resources.callback(three_point.close)
        resources.callback(stop_body)
        if ready_pose is not None:
            from gear_sonic.devices.inspire.inference import enter_ready
            print("PICO_PREPARE_READY 保留支撑；按 Enter 启动 SONIC 准备姿态。", flush=True)
            input()
            print("PICO_PREPARING 正在进入准备姿态。", flush=True)
            body_started = True
            began = time.monotonic()
            confirmed = False
            prompted = False
            stable_since = None
            deadline = None
            next_report = 0.
            while True:
                startup.step(select_pose=False)
                now = time.monotonic()
                if startup.state is not None and now-began >= 3:
                    if not prompted:
                        print("PICO_PREPARED 确认站稳，按现场方式释放支撑；摆好人体标定姿势，按 Enter 开始遥操。", flush=True)
                        prompted = True
                    if not confirmed and enter_ready():
                        confirmed = True
                        deadline = now+60
                        print("PICO_STARTING 检查准备姿态到位，再开始人体标定。", flush=True)
                    if confirmed:
                        error = upper_body_error(startup.state, ready_pose)
                        stable_since = (stable_since or now) if error <= .15 else None
                        if stable_since is not None and now-stable_since >= .5:
                            break
                        if now >= deadline:
                            raise TimeoutError(f'preparation pose did not settle: {error:.3f} rad')
                        if now >= next_report:
                            print(f'PICO_POSE_WAIT error_rad={error:.4f}', flush=True)
                            next_report = now+1
                time.sleep(.02)
        else:
            print("PICO ready. Stand in the calibration pose; press Enter to start. Ctrl+C stops.", flush=True)
            input()
        if ready_pose is not None:
            print("PICO ready. Preparation accepted; calibrating teleoperation.", flush=True)
        sample = reader.get_latest()
        if not three_point.calibrate_now(sample["body_poses_np"]):
            raise RuntimeError("body three-point calibration failed")

        def close_hands(exc_type, exc, traceback):
            try:
                session.close()
            except Exception as cleanup_error:
                if exc is None:
                    raise
                print(f"PICO hand cleanup failed: {cleanup_error}", flush=True)
            # Preserve the original body/SDK failure or Ctrl+C during cleanup.
            return False

        resources.callback(lambda: print(json.dumps(session.summary()), flush=True))
        resources.push(close_hands)
        resources.callback(stop_body)
        session.start()
        mirror = TargetMirror(session, integration)
        # Match the old PICO entry: this small FK workload is faster on the CPU.
        streamer = PoseStreamer(socket, reader, three_point, 5, 50, False, "", "npz", pose_adapter=mirror)
        streamer.reset_yaw()
        last_stamp = None
        waiting_for_body = False
        while True:
            sample = reader.get_latest()
            age = time.monotonic() - float(sample["timestamp_monotonic"])
            if not 0 <= age <= BODY_TRACKING_TIMEOUT_S:
                raise RuntimeError(f"PICO body tracking is stale ({age:.2f}s without a new frame)")
            stamp = int(sample["timestamp_ns"])
            if last_stamp is not None and stamp < last_stamp:
                raise RuntimeError("PICO body timestamp regressed")
            if stamp == last_stamp:
                if age > 0.3 and not waiting_for_body:
                    print("PICO body data paused; waiting for a new frame.", flush=True)
                    waiting_for_body = True
                time.sleep(0.005)
                continue
            if waiting_for_body:
                print("PICO body data restored.", flush=True)
                waiting_for_body = False
            streamer.run_once()
            last_stamp = stamp
            if mirror.sent_frames and not body_started:
                # Mark first so any partial Start failure still sends Stop on exit.
                body_started = True
            if body_started:
                startup.step()
            time.sleep(0.001)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("PICO stopped.")
    except (OSError, ValueError, RuntimeError, TimeoutError, EOFError) as exc:
        parser.exit(1, f"PICO stopped: {exc}\n")


if __name__ == "__main__":
    main()
