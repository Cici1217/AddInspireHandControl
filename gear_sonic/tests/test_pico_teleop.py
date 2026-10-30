"""Offline PICO control timing, safety and original body wire integration."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.tests.test_inspire_optical import calibration_bundle, synthetic_hand
from gear_sonic.devices.inspire.driver.client import FeedbackUnavailable
from gear_sonic.devices.inspire.config import DEFAULT_MAPPING_PATH, load_mapping
from gear_sonic.utils.hand_control.interface import HandState, EffectorAction, EffectorState, ActionReceipt
from gear_sonic.utils.hand_control.registry import deployment
from gear_sonic.utils.teleop.end_effector import SubmittedEffectors
from gear_sonic.utils.teleop.hand_tracking import TrackedHandsFrame
from gear_sonic.utils.teleop.inspire import (
    OpticalHandSession,
    OpticalPicoReader,
    safe_seed,
    slew_target,
    xrt_hand_state_to_tracked,
    InspireSonicIntegration,
)


def sample(*, closed=(), stamp=None, receipt=None, missing=False):
    now = time.monotonic()
    stamp = time.monotonic_ns() if stamp is None else stamp
    receipt = now if receipt is None else receipt
    hands = TrackedHandsFrame(
        stamp,
        receipt,
        time.time(),
        synthetic_hand(closed_channels=closed),
        None if missing else synthetic_hand(closed_channels=closed, mirror=True),
    )
    body = np.zeros((24, 7), dtype=np.float32)
    body[:, 6] = 1
    return {
        "timestamp_ns": stamp,
        "timestamp_monotonic": receipt,
        "timestamp_realtime": time.time(),
        "tracked_hands": hands,
        "body_poses_np": body,
        "dt": 0.02,
        "fps": 50.0,
    }


class FakeReader:
    def __init__(self, *, missing=False, frozen=False):
        self.missing = missing
        self.frozen = frozen
        self.frame = sample()

    def get_latest(self):
        return self.frame if self.frozen else sample(closed={"index", "thumb_rotation"}, missing=self.missing)


class FakeBackend:
    def __init__(self, initial=None):
        self.targets = []
        self.initial = tuple([0.0] * 6 if initial is None else initial)
        self.thread_ids = []
        self.closed = False
        self.error = None
        self.feedback_paused = False

    def connect(self):
        import threading

        self.thread_ids.append(threading.get_ident())

    def read_state(self):
        return HandState(self.initial, self.initial, time.monotonic_ns(), self.error)

    def set_target(self, left, right):
        import threading

        self.thread_ids.append(threading.get_ident())
        if self.error:
            raise RuntimeError(self.error)
        if self.feedback_paused:
            raise FeedbackUnavailable("stale measured feedback")
        self.targets.append((time.monotonic_ns(), left.copy(), right.copy()))

    def close(self):
        import threading

        self.thread_ids.append(threading.get_ident())
        self.closed = True


def test_uniform_slew_preserves_coupled_envelope_and_speed():
    mapping = load_mapping()
    previous = np.array([0, 0, 0, 0, 0, 1.3])
    desired = np.array([1.7, 1.7, 1.7, 1.7, 0.6, 0.3])
    for _ in range(150):
        result = slew_target(previous, desired, mapping=mapping, side="left", elapsed_s=0.02)
        assert np.all(abs(result - previous) <= np.array([j.velocity_limit for j in mapping.left]) * 0.02 + 1e-7)
        assert result[5] <= 1.3 - result[3] / 1.7 + 1e-6
        previous = result
    np.testing.assert_allclose(previous, desired, atol=1e-6)


@pytest.mark.parametrize("side,rotation", [("left", 0.3172), ("right", 0.3224)])
def test_measured_seed_is_preserved_outside_optical_envelope(side, rotation):
    measured = np.array([1.7, 1.7, 1.7, 1.7, 0.6, rotation])
    np.testing.assert_array_equal(safe_seed(measured, load_mapping(), side), measured)


def test_measured_seed_still_rejects_invalid_feedback():
    with pytest.raises(ValueError, match="outside"):
        safe_seed([-0.1, 0, 0, 0, 0, 0], load_mapping(), "left")
    with pytest.raises(ValueError, match="invalid"):
        safe_seed([0, 0, 0, np.nan, 0, 0], load_mapping(), "right")


@pytest.mark.parametrize("side", ["left", "right"])
def test_slew_from_measured_pose_reduces_envelope_excess_without_jump(side):
    mapping = load_mapping()
    previous = np.array([1.7, 1.7, 1.7, 1.7, 0.6, 0.3224])
    limits = np.array([j.velocity_limit for j in getattr(mapping, side)])
    # Even an unconstrained optical input must not move back outside the envelope.
    desired = np.array([1.7, 1.7, 1.7, 1.7, 0.6, 1.3])
    for _ in range(10):
        result = slew_target(previous, desired, mapping=mapping, side=side, elapsed_s=0.02)
        old_excess = max(0, previous[5] - (1.3 - previous[3] / 1.7))
        new_excess = max(0, result[5] - (1.3 - result[3] / 1.7))
        assert new_excess <= old_excess + 1e-7
        assert np.all(abs(result - previous) <= 0.8 * limits * 0.02 + 1e-7)
        previous = result
    np.testing.assert_allclose(previous, [1.7, 1.7, 1.7, 1.7, 0.6, 0.3], atol=1e-6)


@pytest.mark.parametrize("initial", [[0.0] * 6, [1.7, 1.7, 1.7, 1.7, 0.6, 0.3224]])
def test_hand_thread_submits_at_50hz_and_owns_all_backend_calls(initial):
    reader = FakeReader()
    backend = FakeBackend(initial)
    session = OpticalHandSession(
        reader, calibration_bundle(), DEFAULT_MAPPING_PATH, backend_factory=lambda: backend
    )
    session.start()
    time.sleep(0.30)
    session.close()
    assert backend.closed
    assert len(set(backend.thread_ids)) == 1
    assert len(backend.targets) >= 12
    np.testing.assert_array_equal(backend.targets[0][1], initial)
    np.testing.assert_array_equal(backend.targets[0][2], initial)
    gaps = np.diff([row[0] for row in backend.targets]) / 1e9
    assert 0.017 < np.median(gaps) < 0.028
    assert backend.targets[-1][1][3] > 0
    limits = np.array([j.velocity_limit for j in load_mapping().left])
    assert np.all(abs(np.diff([row[1] for row in backend.targets], axis=0)) <= limits * 0.02 + 1e-6)


@pytest.mark.parametrize("dropout", ["missing", "frozen"])
def test_optical_dropout_holds_last_target_and_resumes_without_restarting(dropout):
    reader = FakeReader()
    backend = FakeBackend()
    session = OpticalHandSession(
        reader, calibration_bundle(), DEFAULT_MAPPING_PATH, backend_factory=lambda: backend
    )
    session.start()
    try:
        time.sleep(0.06)
        reader.frame = reader.get_latest()
        setattr(reader, dropout, True)
        time.sleep(session.timeout_s + 0.06)
        session.check()
        held = session.latest()
        assert not held.tracking_valid
        count = session.submissions
        time.sleep(0.06)
        current = session.latest()
        assert session.submissions > count and not backend.closed
        np.testing.assert_array_equal(current.left, held.left)
        assert current.device_timestamp_ns == held.device_timestamp_ns
        # Recovery consumes a fresh device frame, never re-dates the old one.
        setattr(reader, dropout, False)
        time.sleep(0.06)
        resumed = session.latest()
        assert resumed.tracking_valid
        assert resumed.device_timestamp_ns > held.device_timestamp_ns
        assert np.any(resumed.left != held.left)
    finally:
        session.close()
    assert backend.closed


def test_missing_hand_opens_no_control_connection():
    backend = FakeBackend()
    session = OpticalHandSession(
        FakeReader(missing=True), calibration_bundle(), DEFAULT_MAPPING_PATH, backend_factory=lambda: backend
    )
    with pytest.raises(ValueError, match="both"):
        session.start()
    assert not backend.targets and not backend.thread_ids


@pytest.mark.parametrize("recovery", ["fresh", "optical_missing", "fault", "stop"])
def test_hand_feedback_pause_preserves_last_submission_and_can_resume(recovery, capsys):
    class PausingReader(FakeReader):
        def __init__(self):
            super().__init__()
            self.block_read = threading.Event()
            self.waiting = threading.Event()
            self.resume = threading.Event()

        def get_latest(self):
            if self.block_read.is_set():
                self.waiting.set()
                if not self.resume.wait(2):
                    raise TimeoutError("test did not release the reader")
            return super().get_latest()

    class PausingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.pause_seen = threading.Event()

        def set_target(self, left, right):
            try:
                super().set_target(left, right)
            except FeedbackUnavailable:
                self.pause_seen.set()
                raise

    reader, backend = PausingReader(), PausingBackend()
    session = OpticalHandSession(
        reader, calibration_bundle(), DEFAULT_MAPPING_PATH, backend_factory=lambda: backend
    )
    session.start()
    try:
        backend.feedback_paused = True
        assert backend.pause_seen.wait(2)
        held = session.latest()
        count = len(backend.targets)
        time.sleep(0.4)
        session.check()
        assert not backend.closed and len(backend.targets) == count
        assert session.latest() is held  # No invented submission or refreshed source time.
        if recovery == "stop":
            return
        # Change both inputs before the next snapshot, not halfway through a
        # target already computed from a valid frame on the worker thread.
        reader.block_read.set()
        assert reader.waiting.wait(2)
        reader.missing = recovery == "optical_missing"
        backend.error = "gateway FAULT" if recovery == "fault" else None
        backend.feedback_paused = False
        reader.block_read.clear()
        reader.resume.set()
        deadline = time.monotonic() + 2
        while (
            not backend.closed
            and (recovery == "fault" or session.latest() is held)
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        if recovery == "fault":
            with pytest.raises(RuntimeError, match="gateway FAULT"):
                session.check()
            assert backend.closed and len(backend.targets) == count
            return
        resumed = session.latest()
        assert len(backend.targets) > count
        first_resumed = backend.targets[count][1]
        limits = np.array([j.velocity_limit for j in load_mapping().left])
        assert np.all(abs(first_resumed - held.left) <= 0.8 * limits * 0.02 + 1e-7)
        if reader.missing:
            np.testing.assert_array_equal(resumed.left, held.left)
            assert not resumed.tracking_valid
        else:
            assert resumed.device_timestamp_ns > held.device_timestamp_ns
            assert np.any(resumed.left != held.left)
        assert "hand feedback restored; control resumed" in capsys.readouterr().out
    finally:
        reader.resume.set()
        if recovery == "fault":
            with pytest.raises(RuntimeError, match="gateway FAULT"):
                session.close()
        else:
            session.close()
        assert backend.closed


def test_gateway_fault_stops_and_is_not_reset():
    backend = FakeBackend()
    session = OpticalHandSession(
        FakeReader(), calibration_bundle(), DEFAULT_MAPPING_PATH, backend_factory=lambda: backend
    )
    session.start()
    backend.error = "ownership lost"
    time.sleep(0.04)
    with pytest.raises(RuntimeError, match="ownership lost"):
        session.close()
    assert backend.closed


def test_atomic_xrt_discards_mixed_device_frame():
    body = sample()["body_poses_np"]
    hand = np.zeros((26, 7))
    hand[:, 6] = 1

    class SDK:
        stamp = 10

        def get_time_stamp_ns(self):
            return self.stamp

        def is_body_data_available(self):
            return True

        def get_body_joints_pose(self):
            return body

        def get_left_hand_is_active(self):
            return True

        get_right_hand_is_active = get_left_hand_is_active

        def get_left_hand_tracking_state(self):
            return hand

        def get_right_hand_tracking_state(self):
            self.stamp += 1
            return hand

    reader = OpticalPicoReader(sdk=SDK())
    assert reader._read_atomic() is None
    reader._sdk.get_right_hand_tracking_state = lambda: hand
    frame = reader._read_atomic()
    assert frame["timestamp_ns"] == frame["tracked_hands"].timestamp_ns
    assert frame["tracked_hands"].left.positions.flags.writeable is False
    assert reader._read_atomic() is None


@pytest.mark.parametrize("hand_error", ["shape", "sdk"])
def test_hand_read_failure_preserves_fresh_body_frame(hand_error):
    body = sample()["body_poses_np"]

    def hand_state():
        if hand_error == "sdk":
            raise RuntimeError("optical hand unavailable")
        return np.zeros((0, 7))

    sdk = SimpleNamespace(
        get_time_stamp_ns=lambda: 10,
        is_body_data_available=lambda: True,
        get_body_joints_pose=lambda: body,
        get_left_hand_is_active=lambda: True,
        get_right_hand_is_active=lambda: True,
        get_left_hand_tracking_state=hand_state,
        get_right_hand_tracking_state=hand_state,
    )
    reader = OpticalPicoReader(sdk=sdk)
    frame = reader._read_atomic()
    np.testing.assert_array_equal(frame["body_poses_np"], body)
    assert frame["tracked_hands"].left is frame["tracked_hands"].right is None
    assert frame["timestamp_ns"] == 10
    assert reader._read_atomic() is None


def test_xrt_invalid_quaternion_is_not_valid_tracking():
    hand = xrt_hand_state_to_tracked(np.zeros((26, 7)), active=True)
    assert not np.any(hand.valid)
    assert xrt_hand_state_to_tracked(None, active=False) is None


@pytest.mark.parametrize("angle", [0.0, 0.01])
@pytest.mark.parametrize("hand_fault", [False, True])
def test_body_streamer_optional_hook_uses_real_packer_and_no_controller(monkeypatch, angle, hand_fault, capsys):
    import json

    import torch

    from gear_sonic.scripts import pico_manager_thread_server as manager
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

    def unpack_pose_message(raw, topic):
        offset = len(topic)
        header = json.loads(raw[offset : offset + HEADER_SIZE].split(bytes([0]))[0])
        pos = offset + HEADER_SIZE
        values = {}
        for field in header["fields"]:
            dtype = {"f32": "<f4", "f64": "<f8", "i64": "<i8", "i32": "<i4", "bool": "?"}[field["dtype"]]
            count = int(np.prod(field["shape"]))
            array = np.frombuffer(raw, dtype=dtype, count=count, offset=pos).reshape(field["shape"])
            values[field["name"]] = array
            pos += array.nbytes
        return values

    messages = []
    socket = SimpleNamespace(send=messages.append)
    frame = sample()
    reader = SimpleNamespace(get_latest=lambda: frame)
    three = SimpleNamespace(
        enable_smpl_vis=False, process_smpl_pose=lambda *a, **k: np.tile([0, 0, 0, 1, 0, 0, 0], (3, 1))
    )
    latest = {
        "smpl_pose": torch.full((1, 63), angle),
        "smpl_joints_local": torch.zeros((1, 24, 3)),
        "global_orient_quat": torch.tensor([[1.0, 0, 0, 0]]),
    }
    monkeypatch.setattr(manager, "compute_from_body_poses", lambda *args: latest)
    monkeypatch.setattr(manager, "get_controller_inputs", lambda *a: pytest.fail("controller fallback"))
    monkeypatch.setattr(manager, "get_controller_axes", lambda *a: pytest.fail("controller axes"))
    monkeypatch.setattr(manager, "init_hand_ik_solvers", lambda: pytest.fail("Dex3 IK"))

    import msgpack
    import msgpack_numpy as mnp

    from gear_sonic.scripts.run_pico_teleop import TargetMirror
    action = EffectorAction({'left': np.ones(6), 'right': np.ones(6)})

    def latest_hand():
        if hand_fault:
            raise RuntimeError("gateway FAULT (fault latched=True)")
        stamped = EffectorAction(action.values, frame['timestamp_ns'])
        return SubmittedEffectors(ActionReceipt(stamped, time.monotonic_ns(), 1), stamped,
                                  EffectorState(action.values, time.monotonic_ns(), {'left': True, 'right': True}), True)

    mirror_adapter = TargetMirror(
        SimpleNamespace(latest=latest_hand, retarget=lambda source: action),
        InspireSonicIntegration(deployment('inspire')),
    )

    def adapter(source, data):
        assert source["tracked_hands"].timestamp_ns == source["timestamp_ns"]
        return mirror_adapter(source, data)

    streamer = manager.PoseStreamer(socket, reader, three, 5, 50, False, "", "npz", pose_adapter=adapter)
    for _ in range(10):
        frame = sample(stamp=frame["timestamp_ns"] + 20_000_000)
        streamer.run_once()
    assert len(messages) >= 2
    poses = [unpack_pose_message(message, topic="pose") for message in messages if message.startswith(b"pose")]
    assert len(poses) >= 2
    assert poses[-1]["frame_index"][-1] > poses[0]["frame_index"][-1]
    pose = poses[-1]
    np.testing.assert_array_equal(pose["left_hand_joints"], np.zeros(7))
    if hand_fault:
        assert len(poses) == len(messages)  # No failed hand command is mirrored as sent.
        assert capsys.readouterr().out.count("PICO hands stopped; body continues:") == 1
    else:
        mirror = msgpack.unpackb(messages[-1][len("teleop_target") :], raw=False, object_hook=mnp.decode)
        assert mirror["left_hand_native6"].shape == (6,)
        from gear_sonic.devices.inspire.recording.teleop_target import TeleopTargetSample
        legacy = TeleopTargetSample.from_wire(mirror, received_monotonic_ns=time.monotonic_ns())
        np.testing.assert_array_equal(legacy.left_hand_native6, mirror['end_effector_action']['left'])
        assert legacy.hand_command_sent and legacy.hand_target_valid
    assert all(np.all(np.isfinite(value)) for value in pose.values())


def test_nonfinite_body_is_rejected_even_when_hands_have_failed():
    from gear_sonic.scripts.run_pico_teleop import TargetMirror

    def failed_hand():
        raise RuntimeError("gateway FAULT")

    mirror = TargetMirror(SimpleNamespace(latest=failed_hand))
    with pytest.raises(ValueError, match="non-finite body target"):
        mirror(sample(), {"smpl_pose": np.full((1, 63), np.nan)})
    assert mirror.sent_frames == 0


def test_body_start_waits_for_control_before_selecting_pose():
    from gear_sonic.scripts.run_pico_teleop import BodyStartup
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    now = [0.0]
    states = []
    sent = []
    startup = BodyStartup(
        SimpleNamespace(send=sent.append), lambda: states.pop(0) if states else None, clock=lambda: now[0]
    )
    startup.step()
    assert sent[-1] == build_command_message(start=True, stop=False, planner=True)
    now[0] = 0.1
    startup.step()
    assert not startup.pose_selected
    states.append({"body_q": np.zeros(29), "token_state": np.zeros(64)})
    startup.step()
    assert sent[-1] == build_command_message(start=True, stop=False, planner=False)
    assert startup.pose_selected


def test_body_start_missing_control_times_out():
    from gear_sonic.scripts.run_pico_teleop import BodyStartup

    now = [0.0]
    startup = BodyStartup(SimpleNamespace(send=lambda data: None), lambda: None, clock=lambda: now[0])
    startup.step()
    now[0] = 10.01
    with pytest.raises(TimeoutError, match="CONTROL output"):
        startup.step()
    assert not startup.pose_selected


@pytest.mark.parametrize("pose_selected", [False, True])
def test_body_stop_keeps_current_mode(pose_selected):
    from gear_sonic.scripts.run_pico_teleop import BodyStartup
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    messages = []
    startup = BodyStartup(SimpleNamespace(send=messages.append), lambda: None)
    startup.pose_selected = pose_selected
    startup.stop()
    assert messages == [build_command_message(start=False, stop=True, planner=not pose_selected)] * 3


@pytest.mark.parametrize("hand_fault", [False, True])
@pytest.mark.parametrize("source_mode", ["live", "paused", "disconnected", "regressed"])
def test_entry_hand_fault_keeps_body_running_until_interrupt_or_stale_body(
    monkeypatch, tmp_path, hand_fault, source_mode, capsys
):
    import msgpack
    import msgpack_numpy as mnp
    import zmq

    from gear_sonic.data.robot_model.instantiation import g1_inspire
    from gear_sonic.scripts import pico_manager_thread_server as manager, run_pico_teleop as entry
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

    sent, feedback, closed = [], [], []
    poses = []
    options = []
    paused_polls = []
    start = build_command_message(start=True, stop=False, planner=True)

    def send(message):
        sent.append(message)
        if message == start:
            feedback.append(
                b"g1_debug"
                + msgpack.packb(
                    {"body_q": np.zeros(29), "token_state": np.zeros(64)}, default=mnp.encode, use_bin_type=True
                )
            )

    def receive(flags):
        if feedback:
            return feedback.pop(0)
        raise zmq.Again()

    socket = SimpleNamespace(
        send=send,
        recv=receive,
        bind=lambda endpoint: None,
        connect=lambda endpoint: None,
        setsockopt=lambda *args: options.append(args),
        close=lambda *args: closed.append("socket"),
    )
    context = SimpleNamespace(socket=lambda kind: socket, term=lambda: closed.append("context"))
    monkeypatch.setattr(zmq, "Context", lambda: context)

    def reader_factory():
        last = sample()

        def latest():
            nonlocal last
            if source_mode == "paused" and len(poses) == 3 and len(paused_polls) < 4:
                paused_polls.append(1)
                return {**last, "timestamp_monotonic": time.monotonic() - 0.4}
            if source_mode == "disconnected" and len(poses) == 5:
                return sample(receipt=time.monotonic() - entry.BODY_TRACKING_TIMEOUT_S - 0.1)
            if source_mode == "regressed" and len(poses) == 5:
                return sample(stamp=1)
            last = sample()
            return last

        return SimpleNamespace(start=lambda: None, stop=lambda: closed.append("reader"), get_latest=latest)

    class Session:
        checks = 0
        timeout_s = 0.3

        def check(self):
            self.checks += 1
            if hand_fault and self.checks >= 3:
                raise RuntimeError("gateway FAULT (fault latched=True)")

        def latest(self):
            self.check()
            action = EffectorAction({'left': np.zeros(6), 'right': np.zeros(6)}, 1)
            return SubmittedEffectors(ActionReceipt(action, 1, 1), action,
                                      EffectorState(action.values, 1, {'left': True, 'right': True}), True)

        def start(self):
            pass

        def close(self):
            closed.append("hands")
            if hand_fault:
                raise RuntimeError("hand DISARM failed")

        def summary(self):
            return {}

        def retarget(self, sample):
            return EffectorAction({'left': np.zeros(6), 'right': np.zeros(6)})

    def session_factory(*args, **kwargs):
        return Session()

    class Streamer:
        def __init__(self, *args, pose_adapter, **kwargs):
            self.adapter = pose_adapter
            self.frames = 0

        def reset_yaw(self):
            pass

        def run_once(self):
            self.frames += 1
            if self.frames == 6:
                raise KeyboardInterrupt
            data = {
                "smpl_pose": np.zeros((1, 63)),
                "smpl_joints": np.zeros((1, 24, 3)),
                "body_quat_w": np.zeros((1, 4)),
                "vr_position": np.zeros(9),
                "vr_orientation": np.zeros(12),
                "frame_index": np.array([self.frames]),
            }
            mirrored = self.adapter(sample(), data)
            assert (mirrored is None) == (hand_fault and self.frames >= 3)
            assert build_command_message(start=False, stop=True, planner=False) not in sent
            poses.append(self.frames)
            send(b"pose")

    monkeypatch.setattr(entry, "OpticalPicoReader", reader_factory)
    monkeypatch.setattr(entry, "OpticalSession", session_factory)
    monkeypatch.setattr(manager, "PoseStreamer", Streamer)
    monkeypatch.setattr(
        manager,
        "ThreePointPose",
        lambda **kwargs: SimpleNamespace(
            close=lambda: closed.append("three_point"),
            calibrate_now=lambda body: True,
        ),
    )
    from gear_sonic.utils.hand_control.registry import Deployment
    monkeypatch.setattr(Deployment, "robot_model", lambda self: object())
    monkeypatch.setattr("builtins.input", lambda: "")
    calibration = tmp_path / "hands.json"
    calibration_bundle().save(calibration)
    args = entry.build_parser().parse_args(["--calibration", str(calibration)])
    with pytest.raises(
        RuntimeError if source_mode in ("disconnected", "regressed") else KeyboardInterrupt
    ) as error:
        entry.run(args)
    if source_mode == "disconnected":
        assert "PICO body tracking is stale" in str(error.value)
    if source_mode == "regressed":
        assert "PICO body timestamp regressed" in str(error.value)
    output = capsys.readouterr().out
    if source_mode == "paused":
        assert len(paused_polls) == 4
        assert "body data paused" in output and "body data restored" in output
    if hand_fault:
        assert "PICO hand cleanup failed: hand DISARM failed" in output
    assert poses == [1, 2, 3, 4, 5]  # Body packets continue after the frame-three hand fault.
    assert sent[-3:] == [build_command_message(start=False, stop=True, planner=False)] * 3
    assert (zmq.SNDHWM, 3) in options and (zmq.LINGER, 0) in options
    assert "reader" in closed and "context" in closed
    assert "hands" in closed


def test_shared_calibration_captures_four_poses_and_saves_bundle(monkeypatch, tmp_path):
    from gear_sonic.scripts import calibrate_pico_hands as entry
    from gear_sonic.tests.test_inspire_optical import extract_inspire_features, synthetic_hand
    from gear_sonic.devices.inspire.optical import InspireOpticalCalibrationBundle

    names = []
    channels = {
        "OPEN": set(),
        "FOUR_FINGERS_CLOSED": {"pinky", "ring", "middle", "index"},
        "THUMB_BEND": {"thumb_bend"},
        "THUMB_ROTATION": {"thumb_rotation"},
    }
    prompts = []
    monkeypatch.setattr("builtins.input", lambda text: prompts.append(text))

    def capture(reader, *, name, samples, timeout_s, max_peak_to_peak):
        names.append(name)
        assert samples == 30 and timeout_s == 15 and max_peak_to_peak == 0.15
        return SimpleNamespace(
            left_features=extract_inspire_features(synthetic_hand(closed_channels=channels[name])),
            right_features=extract_inspire_features(synthetic_hand(closed_channels=channels[name], mirror=True)),
        )

    monkeypatch.setattr(entry, "capture_stable_pose", capture)
    output = tmp_path / "hands.json"
    bundle = entry.calibrate(SimpleNamespace(tracker_source="xrt.openxr26"), output)
    assert names == list(channels) and len(prompts) == 4
    saved = InspireOpticalCalibrationBundle.load(output)
    assert saved.calibration_id == bundle.calibration_id
    assert saved.samples_per_pose == 30
    np.testing.assert_array_equal(saved.left.closed_features, bundle.left.closed_features)


def test_preparation_waits_for_explicit_handoff_and_requires_fresh_feedback():
    from gear_sonic.scripts.run_pico_teleop import BodyStartup
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message
    from gear_sonic.utils.inference.ready_pose import load_ready_pose
    from gear_sonic.utils.hand_control.registry import ROOT
    pose=load_ready_pose(ROOT/'gear_sonic/data/poses/g1/box_ready.txt')
    now=[0.];states=[];sent=[]
    startup=BodyStartup(SimpleNamespace(send=sent.append),lambda:states.pop(0) if states else None,
                        clock=lambda:now[0],ready_pose=pose)
    startup.step(select_pose=False)
    handoff=build_command_message(start=True,stop=False,planner=False)
    for i in range(1,602):
        now[0]=i*.02
        states.append({'body_q':np.array(pose),'token_state':np.zeros(64)})
        startup.step(select_pose=False)
    assert not startup.pose_selected and handoff not in sent
    now[0]+=.6
    with pytest.raises(TimeoutError,match='feedback stopped'):startup.step(select_pose=False)
    states.append({'body_q':np.array(pose),'token_state':np.zeros(64)})
    startup.step()
    assert startup.pose_selected and sent[-1]==handoff
