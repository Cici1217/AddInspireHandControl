from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import msgpack
import msgpack_numpy as mnp
import numpy as np
import pytest

from gear_sonic.devices.inspire.recording.frame import (
    CaptureSyncPolicy,
    build_lerobot_frame,
)
from gear_sonic.devices.inspire.recording.teleop_target import TeleopTargetClient, TeleopTargetSample
from gear_sonic.devices.inspire.config import load_mapping
from gear_sonic.devices.inspire.contract import ContractError, HandStatus, InspireHandState


class FakeRobotModel:
    def get_configuration_from_actuated_joints(
        self,
        body_actuated_joint_values,
        left_hand_actuated_joint_values,
        right_hand_actuated_joint_values,
    ):
        return np.concatenate(
            (
                body_actuated_joint_values,
                left_hand_actuated_joint_values,
                right_hand_actuated_joint_values,
            )
        )


def make_sources(now_ns=1_000_000_000):
    hand = np.full(6, 0.1, dtype=np.float32)
    state = InspireHandState(
        sequence=10,
        source_monotonic_ns=now_ns - 3_000_000,
        gateway_monotonic_ns=now_ns - 1_000_000,
        status=HandStatus.ARMED,
        left_q=hand,
        right_q=hand,
        accepted_command_sequence=9,
        left_command_q=hand,
        right_command_q=hand,
    )
    body = {
        "body_q": np.zeros(29, dtype=np.float32),
        "body_dq": np.zeros(29, dtype=np.float32),
        "base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "token_state": np.arange(64, dtype=np.float32),
    }
    camera = {
        "images": {"ego_view": np.zeros((480, 640, 3), dtype=np.uint8)},
        "timestamps": {"ego_view": 123.0},
    }
    target = TeleopTargetSample(
        frame_index=7,
        pico_device_timestamp_ns=123456,
        smpl_pose=np.arange(63, dtype=np.float32),
        smpl_joints=np.arange(72, dtype=np.float32),
        body_quat_w=np.array([1, 0, 0, 0], dtype=np.float32),
        vr_position=np.arange(9, dtype=np.float32),
        vr_orientation=np.tile([1, 0, 0, 0], 3).astype(np.float32),
        left_hand_native6=np.full(6, 0.2, dtype=np.float32),
        right_hand_native6=np.full(6, 0.3, dtype=np.float32),
        hand_target_valid=True,
        hand_command_sent=True,
        received_monotonic_ns=now_ns - 2_000_000,
    )
    return body, camera, state, target


def build(now_ns=1_000_000_000, **updates):
    body, camera, state, target = make_sources(now_ns)
    body.update(updates.pop("body", {}))
    camera.update(updates.pop("camera", {}))
    state = updates.pop("state", state)
    target = updates.pop("target", target)
    assert not updates
    return build_lerobot_frame(
        now_ns=now_ns,
        body_received_ns=now_ns - 2_000_000,
        camera_received_ns=now_ns - 4_000_000,
        hand_received_ns=now_ns - 1_000_000,
        target_received_ns=target.received_monotonic_ns,
        body_message=body,
        camera_message=camera,
        hand_state=state,
        teleop_target=target,
        robot_model=FakeRobotModel(),
        mapping=load_mapping(),
        sync_policy=CaptureSyncPolicy(),
    )


def test_frame_uses_measured_state_and_exact_teleop_targets() -> None:
    frame = build()
    assert frame.observation_state.shape == (41,)
    assert frame.observation_left_hand.shape == (6,)
    assert frame.motion_token.shape == (64,)
    assert frame.action_left_hand.shape == (6,)
    assert frame.ego_view.shape == (480, 640, 3)
    assert frame.camera_source_timestamp_ns == 123_000_000_000
    assert frame.hand_command_sequence == 9
    np.testing.assert_array_equal(frame.action_left_hand, np.full(6, 0.2, dtype=np.float32))
    np.testing.assert_array_equal(frame.action_right_hand, np.full(6, 0.3, dtype=np.float32))
    np.testing.assert_array_equal(frame.teleop_smpl_pose, np.arange(63, dtype=np.float32))
    assert set(frame.to_exporter_dict()) == {
        "observation.images.ego_view",
        "observation.state",
        "observation.left_hand",
        "observation.right_hand",
        "observation.projected_gravity",
        "action.motion_token",
        "teleop.left_hand_intent",
        "teleop.right_hand_intent",
        "action.left_hand_joints",
        "action.right_hand_joints",
        "teleop.smpl_joints",
        "teleop.smpl_pose",
        "teleop.body_quat_w",
        "teleop.vr_3pt_position",
        "teleop.vr_3pt_orientation",
        "observation.hand_gateway_healthy",
        "teleop.hand_target_valid",
        "teleop.hand_command_sent",
    }


def test_frame_uses_local_receipt_not_remote_gateway_clock_for_freshness() -> None:
    now_ns = 9_000_000_000
    _, _, state, _ = make_sources(now_ns)
    remote_clock_state = InspireHandState(
        sequence=state.sequence,
        source_monotonic_ns=state.source_monotonic_ns,
        gateway_monotonic_ns=123,
        status=state.status,
        left_q=state.left_q,
        right_q=state.right_q,
        accepted_command_sequence=state.accepted_command_sequence,
        left_command_q=state.left_command_q,
        right_command_q=state.right_command_q,
    )
    frame = build(now_ns=now_ns, state=remote_clock_state)
    assert frame.observation_left_hand.shape == (6,)
    assert frame.observation_right_hand.shape == (6,)


@pytest.mark.parametrize(
    "body,match",
    [
        ({"token_state": np.zeros(63)}, "token_state"),
        ({"token_state": np.full(64, np.nan)}, "NaN"),
        ({"body_q": np.zeros(28)}, "body_q"),
    ],
)
def test_frame_rejects_missing_or_invalid_body_contract(body, match) -> None:
    with pytest.raises(ContractError, match=match):
        build(body=body)


def test_frame_records_non_armed_hand_state_as_unhealthy_without_dropping_body() -> None:
    now_ns = 1_000_000_000
    _, _, state, target = make_sources(now_ns)
    stale = InspireHandState(
        sequence=state.sequence,
        source_monotonic_ns=state.source_monotonic_ns,
        gateway_monotonic_ns=state.gateway_monotonic_ns,
        status=HandStatus.STALE,
        left_q=state.left_q,
        right_q=state.right_q,
        accepted_command_sequence=state.accepted_command_sequence,
        left_command_q=state.left_command_q,
        right_command_q=state.right_command_q,
    )
    frame = build(state=stale)
    assert frame.hand_gateway_healthy.tolist() == [0.0]
    np.testing.assert_array_equal(frame.action_left_hand, target.left_hand_native6)


def test_frame_rejects_stale_body_camera_or_teleop_target() -> None:
    now_ns = 1_000_000_000
    body, camera, state, target = make_sources(now_ns)
    with pytest.raises(ContractError, match="stale"):
        build_lerobot_frame(
            now_ns=now_ns,
            body_received_ns=now_ns - 200_000_000,
            camera_received_ns=now_ns,
            hand_received_ns=now_ns,
            target_received_ns=now_ns,
            body_message=body,
            camera_message=camera,
            hand_state=state,
            teleop_target=target,
            robot_model=FakeRobotModel(),
            mapping=load_mapping(),
            sync_policy=CaptureSyncPolicy(),
        )


def test_frame_rejects_bad_image_and_out_of_range_hand() -> None:
    _, _, state, _ = make_sources()
    with pytest.raises(ContractError, match="ego_view"):
        build(camera={"images": {"ego_view": np.zeros((10, 10, 3), dtype=np.uint8)}})
    bad = InspireHandState(
        sequence=state.sequence,
        source_monotonic_ns=state.source_monotonic_ns,
        gateway_monotonic_ns=state.gateway_monotonic_ns,
        status=state.status,
        left_q=np.full(6, 9.0),
        right_q=state.right_q,
        accepted_command_sequence=state.accepted_command_sequence,
        left_command_q=state.left_command_q,
        right_command_q=state.right_command_q,
    )
    with pytest.raises(ContractError, match="outside"):
        build(state=bad)


def wire_target(index=7):
    _, _, _, target = make_sources()
    return {
        "frame_index": [index],
        "pico_device_timestamp_ns": [target.pico_device_timestamp_ns],
        "hand_source_device_timestamp_ns": [123456],
        "hand_submitted_monotonic_ns": [990_000_000],
        "smpl_pose": target.smpl_pose,
        "smpl_joints": target.smpl_joints,
        "body_quat_w": target.body_quat_w,
        "vr_position": target.vr_position,
        "vr_orientation": target.vr_orientation,
        "left_hand_native6": target.left_hand_native6,
        "right_hand_native6": target.right_hand_native6,
        "left_hand_intent": np.full(6, 0.4),
        "right_hand_intent": np.full(6, 0.5),
        "native6_target_valid": [True],
        "native6_command_sent": [True],
    }


def test_target_subscriber_rejects_duplicate_and_regression():
    messages = iter(
        b"teleop_target" + msgpack.packb(wire_target(index), default=mnp.encode, use_bin_type=True)
        for index in [7, 7, 6, 8]
    )
    client = object.__new__(TeleopTargetClient)
    client._socket = SimpleNamespace(recv=lambda: next(messages))
    client._last_frame_index = -1
    first = client.receive_timed()
    with pytest.raises(TimeoutError, match="receipt was not refreshed"):
        client.receive_timed()
    with pytest.raises(ContractError, match="regressed"):
        client.receive_timed()
    assert client.receive_timed().frame_index == 8
    assert first.hand_submitted_monotonic_ns == 990_000_000


def test_frame_keeps_intent_submission_and_measurement_distinct():
    _, _, _, original = make_sources()
    target = replace(original, left_hand_intent=np.full(6, 0.4), right_hand_intent=np.full(6, 0.5))
    frame = build(target=target)
    np.testing.assert_allclose(frame.observation_left_hand, 0.1)
    np.testing.assert_allclose(frame.action_left_hand, 0.2)
    np.testing.assert_allclose(frame.intent_left_hand, 0.4)


@pytest.mark.parametrize("key", ["hand_source_device_timestamp_ns", "hand_submitted_monotonic_ns"])
def test_target_rejects_negative_hand_timestamp(key):
    message = wire_target()
    message[key] = [-1]
    with pytest.raises(ContractError, match="non-negative"):
        TeleopTargetSample.from_wire(message, received_monotonic_ns=1_000_000_000)


def test_uncommitted_export_cancels_writer_and_keeps_incomplete(tmp_path):
    from gear_sonic.data.robot_model.instantiation.g1_inspire import instantiate_g1_rh56dfx_robot_model
    from gear_sonic.devices.inspire.recording.exporter import InspireLeRobotExporter

    cancellations = []
    writer = SimpleNamespace(video_writers={"ego": SimpleNamespace(cancel=lambda: cancellations.append(True))})
    output = tmp_path / "dataset"
    with InspireLeRobotExporter(
        output,
        robot_model=instantiate_g1_rh56dfx_robot_model(),
        fps=50,
        task_prompt="capture",
        exporter_factory=lambda **kwargs: writer,
    ) as exporter:
        staging = exporter.staging_root
    assert cancellations == [True]
    assert (staging / "INCOMPLETE").is_file()
    assert not output.exists()
    with pytest.raises(RuntimeError, match="not writable"):
        exporter.append(build())


def test_export_never_overwrites_existing_output(tmp_path):
    from gear_sonic.devices.inspire.recording.exporter import InspireLeRobotExporter

    (tmp_path / "dataset").mkdir()
    with pytest.raises(FileExistsError):
        InspireLeRobotExporter(
            tmp_path / "dataset",
            robot_model=None,
            fps=50,
            task_prompt="capture",
            exporter_factory=lambda **kwargs: pytest.fail("writer must not open"),
        )
