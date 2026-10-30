import numpy as np
import pytest

from gear_sonic.devices.inspire.config import load_mapping
from gear_sonic.devices.inspire.optical import (
    InspireOpticalCalibration,
    InspireOpticalCalibrationBundle,
    OpticalRetargetingError,
    apply_thumb_index_collision_envelope,
    extract_inspire_features,
)
from gear_sonic.utils.teleop.hand_tracking import TrackedHandJoints

OPEN_Q = np.zeros(6, dtype=np.float32)
CLOSED_Q = np.asarray([1.0, 1.0, 1.0, 0.8, 0.6, 0.3], dtype=np.float32)
ALL_CLOSED = {"pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation"}


def _chain(start, lengths, angles):
    points = [np.asarray(start, dtype=np.float32)]
    for length, angle in zip(lengths, angles):
        direction = np.array([0.0, np.cos(angle), -np.sin(angle)], dtype=np.float32)
        points.append(points[-1] + float(length) * direction)
    return np.asarray(points)


def synthetic_hand(*, closed_channels=(), mirror=False):
    positions = np.zeros((26, 3), dtype=np.float32)
    positions[1] = [0.0, -0.06, 0.0]
    finger_specs = {
        "index": (6, 0.03),
        "middle": (11, 0.01),
        "ring": (16, -0.01),
        "pinky": (21, -0.03),
    }
    for name, (start_index, x) in finger_specs.items():
        angles = [0.0, 0.45, 0.95, 1.30] if name in closed_channels else [0.0] * 4
        positions[start_index : start_index + 5] = _chain([x, 0.0, 0.0], [0.025] * 4, angles)

    thumb_planar = np.asarray(
        [-0.15, 0.988, 0.0] if "thumb_rotation" in closed_channels else [0.72, 0.694, 0.0],
        dtype=np.float32,
    )
    thumb_planar /= np.linalg.norm(thumb_planar)
    bend_angles = [0.10, 0.70, 1.20] if "thumb_bend" in closed_channels else [0.0] * 3
    thumb_points = [np.asarray([0.025, -0.005, 0.0], dtype=np.float32)]
    for angle in bend_angles:
        direction = np.cos(angle) * thumb_planar + np.asarray([0.0, 0.0, -np.sin(angle)], dtype=np.float32)
        thumb_points.append(thumb_points[-1] + 0.022 * direction)
    positions[2:6] = np.asarray(thumb_points)

    if mirror:
        positions[:, 0] *= -1.0
    orientations = np.zeros((26, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    return TrackedHandJoints(
        positions=positions,
        orientations=orientations,
        radii=np.full(26, 0.008, dtype=np.float32),
        valid=np.ones(26, dtype=np.bool_),
    )


def calibration(side="left", mirror=False):
    return InspireOpticalCalibration.capture(
        calibration_id=f"test-{side}",
        side=side,
        open_hand=synthetic_hand(mirror=mirror),
        closed_hand=synthetic_hand(closed_channels=ALL_CLOSED, mirror=mirror),
        open_q=OPEN_Q,
        closed_q=CLOSED_Q,
    )


def calibration_bundle():
    open_left = extract_inspire_features(synthetic_hand())
    open_right = extract_inspire_features(synthetic_hand(mirror=True))
    fingers = {"pinky", "ring", "middle", "index"}
    return InspireOpticalCalibrationBundle.from_independent_feature_poses(
        calibration_id="test-bundle",
        created_realtime_ns=123,
        samples_per_pose=20,
        left_open=open_left,
        right_open=open_right,
        left_fingers_closed=extract_inspire_features(synthetic_hand(closed_channels=fingers)),
        right_fingers_closed=extract_inspire_features(synthetic_hand(closed_channels=fingers, mirror=True)),
        left_thumb_bent=extract_inspire_features(synthetic_hand(closed_channels={"thumb_bend"})),
        right_thumb_bent=extract_inspire_features(synthetic_hand(closed_channels={"thumb_bend"}, mirror=True)),
        left_thumb_rotated=extract_inspire_features(synthetic_hand(closed_channels={"thumb_rotation"})),
        right_thumb_rotated=extract_inspire_features(
            synthetic_hand(closed_channels={"thumb_rotation"}, mirror=True)
        ),
    )


def test_open_closed_calibration_reaches_explicit_native6_endpoints() -> None:
    value = calibration()
    np.testing.assert_allclose(value.retarget(synthetic_hand()), OPEN_Q, atol=1e-6)
    np.testing.assert_allclose(value.retarget(synthetic_hand(closed_channels=ALL_CLOSED)), CLOSED_Q, atol=1e-6)


@pytest.mark.parametrize("name,index", [("pinky", 0), ("ring", 1), ("middle", 2), ("index", 3)])
def test_four_fingers_are_independent_not_one_dimensional(name, index) -> None:
    result = calibration().retarget(synthetic_hand(closed_channels={name}))
    assert result[index] > 0.75 * CLOSED_Q[index]
    others = np.delete(result[:4], index)
    np.testing.assert_allclose(others, 0.0, atol=1e-6)


@pytest.mark.parametrize("name,index", [("thumb_bend", 4), ("thumb_rotation", 5)])
def test_thumb_features_are_independent(name, index) -> None:
    result = calibration().retarget(synthetic_hand(closed_channels={name}))
    assert result[index] > 0.75 * CLOSED_Q[index]
    other = 5 if index == 4 else 4
    assert abs(float(result[other])) < 1e-3


def test_left_right_mirror_produces_same_semantic_features() -> None:
    left = extract_inspire_features(synthetic_hand(closed_channels=ALL_CLOSED))
    right = extract_inspire_features(synthetic_hand(closed_channels=ALL_CLOSED, mirror=True))
    np.testing.assert_allclose(left, right, atol=1e-6)


def test_invalid_required_joint_fails_closed() -> None:
    hand = synthetic_hand()
    valid = np.array(hand.valid, copy=True)
    valid[10] = False
    occluded = TrackedHandJoints(hand.positions, hand.orientations, hand.radii, valid)
    with pytest.raises(OpticalRetargetingError, match="joints are invalid"):
        calibration().retarget(occluded)


def test_degenerate_open_closed_calibration_is_rejected() -> None:
    hand = synthetic_hand()
    with pytest.raises(OpticalRetargetingError, match="degenerate"):
        InspireOpticalCalibration.capture(
            calibration_id="bad",
            side="left",
            open_hand=hand,
            closed_hand=hand,
            open_q=OPEN_Q,
            closed_q=CLOSED_Q,
        )


def test_calibration_round_trip_keeps_native6_order() -> None:
    original = calibration()
    restored = InspireOpticalCalibration.from_dict(original.to_dict())
    assert restored.calibration_id == original.calibration_id
    np.testing.assert_allclose(restored.closed_q, CLOSED_Q)


def test_independent_pose_bundle_uses_exact_native6_hardware_endpoints(tmp_path) -> None:
    original = calibration_bundle()
    path = original.save(tmp_path / "pico-native6.json")
    restored = InspireOpticalCalibrationBundle.load(path)

    np.testing.assert_allclose(restored.left.open_q, [0.0] * 6)
    np.testing.assert_allclose(restored.right.open_q, [0.0] * 6)
    np.testing.assert_allclose(restored.left.closed_q, [1.7, 1.7, 1.7, 1.7, 0.6, 1.3])
    np.testing.assert_allclose(restored.right.closed_q, [1.7, 1.7, 1.7, 1.7, 0.6, 1.3])
    assert restored.samples_per_pose == 20


def test_independent_thumb_rotation_uses_full_hardware_range() -> None:
    result = calibration_bundle().left.retarget(synthetic_hand(closed_channels={"thumb_rotation"}))
    np.testing.assert_allclose(result[:4], 0.0, atol=1e-6)
    assert abs(float(result[4])) < 1e-3
    np.testing.assert_allclose(result[5], 1.3, atol=1e-6)


def test_closed_index_reduces_thumb_rotation_to_ta_collision_free_pose() -> None:
    result = calibration_bundle().left.retarget(synthetic_hand(closed_channels={"index", "thumb_rotation"}))
    np.testing.assert_allclose(result[3], 1.7, atol=1e-6)
    np.testing.assert_allclose(result[5], 0.3, atol=1e-6)


def test_thumb_index_collision_envelope_is_continuous() -> None:
    result = apply_thumb_index_collision_envelope(
        np.asarray([0.0, 0.0, 0.0, 0.85, 0.0, 1.3], dtype=np.float32),
        mapping=load_mapping(),
        side="left",
    )
    np.testing.assert_allclose(result[5], 0.8, atol=1e-6)
