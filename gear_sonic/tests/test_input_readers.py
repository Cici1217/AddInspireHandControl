"""The current body reader accepts decoded wire data and native DeviceIO poses."""
from types import SimpleNamespace
import numpy as np

from gear_sonic.utils.teleop.input_readers import _body_data_to_24x7


def test_decoded_body_wire_data_uses_existing_teleop_shape():
    payload = {
        "timestamp": 1_000_000_100,
        "joint_positions": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "joint_orientations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    }

    poses = _body_data_to_24x7(payload)
    assert poses.shape == (24, 7)
    np.testing.assert_allclose(poses[0], [0.1, 0.2, 0.3, 0, 0, 0, 1])
    np.testing.assert_allclose(poses[1], [0.4, 0.5, 0.6, 0, 0, 1, 0])


def test_native_body_without_valid_joints_does_not_emit_fake_pose():
    joint = SimpleNamespace(is_valid=False, pose=SimpleNamespace(
        position=[.1,.2,.3], orientation=[0,0,0,1]))
    body = SimpleNamespace(joints=SimpleNamespace(joints=lambda index:joint))
    assert _body_data_to_24x7(body) is None
    joint.is_valid = True
    poses = _body_data_to_24x7(body)
    assert poses.shape == (24,7)
    np.testing.assert_allclose(poses[-1], [.1,.2,.3,0,0,0,1])
