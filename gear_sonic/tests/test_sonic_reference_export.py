"""Native replay CSV compatibility without a simulator or robot connection."""

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from gear_sonic.devices.inspire.recording.sonic_reference import SonicReferenceWriter
from gear_sonic.tests.test_inspire_recording import build


def test_native_reader_loads_measured_angles_velocities_and_root(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native CSV compatibility check requires g++")
    writer = SonicReferenceWriter()
    for index in range(3):
        frame = build(now_ns=1_000_000_000 + index * 20_000_000, body={
            "body_q": np.arange(29) / 100 + index / 10,
            "body_dq": np.arange(29) / 100 + .5,
            "base_quat": [.8, 0, 0, .6],
        })
        writer.append(frame)
    result = writer.finish(tmp_path)
    assert result["frames"] == result["source_frames"] == 3
    # The SMPL root in the source fixture is identity; it must NOT replace IMU.
    np.testing.assert_allclose(np.loadtxt(tmp_path / result["path"] / "body_quat.csv",
                                       delimiter=",", skiprows=1), [[.8, 0, 0, .6]] * 3)
    repo = Path(__file__).resolve().parents[2]
    native_include = repo / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include"
    source = tmp_path / "read_reference.cpp"
    source.write_text(r'''
#include <memory>
#include <span>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "motion_data_reader.hpp"
int main(int argc, char** argv) {
    MotionDataReader reader;
    if (argc != 2 || !reader.ReadFromCSV(argv[1]) || reader.motions.size() != 1) return 1;
    auto motion = reader.motions[0];
    if (motion->timesteps != 3 || motion->GetNumJoints() != 29 ||
        motion->GetNumBodies() != 1 || motion->GetNumBodyQuaternions() != 1) return 2;
    for (int frame = 0; frame < 3; ++frame) {
        for (int j = 0; j < 29; ++j) {
            double position = mujoco_to_isaaclab[j] / 100.0 + frame / 10.0;
            double velocity = mujoco_to_isaaclab[j] / 100.0 + .5;
            if (std::abs(motion->JointPositions(frame)[j] - position) > 1e-6) return 3;
            if (std::abs(motion->JointVelocities(frame)[j] - velocity) > 1e-6) return 4;
        }
        if (std::abs(motion->BodyQuaternions(frame)[0][0] - .8) > 1e-6 ||
            std::abs(motion->BodyQuaternions(frame)[0][3] - .6) > 1e-6) return 5;
        for (double xyz : motion->BodyPositions(frame)[0]) if (xyz != 0) return 6;
    }
}
''')
    binary = tmp_path / "read_reference"
    subprocess.run([compiler, "-std=c++20", "-I", str(native_include), str(source),
                    "-o", str(binary)], check=True, capture_output=True, text=True, timeout=60)
    subprocess.run([str(binary), str(tmp_path / "sonic_reference")], check=True,
                   capture_output=True, text=True, timeout=10)


def test_capture_gap_preserves_motion_duration_and_quaternion_sign(tmp_path):
    writer = SonicReferenceWriter()
    for milliseconds, sign in [(0, 1), (20, -1), (60, 1)]:
        writer.append(build(now_ns=1_000_000_000 + milliseconds * 1_000_000, body={
            "body_q": np.full(29, milliseconds / 1000),
            "body_dq": np.ones(29),
            "base_quat": [2 * sign, 0, 0, 0],
        }))
    result = writer.finish(tmp_path)
    assert result["source_frames"] == 3 and result["frames"] == 4
    assert result["max_source_gap_seconds"] == .04
    directory = tmp_path / result["path"]
    positions = np.loadtxt(directory / "joint_pos.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(positions[:, 0], [0, .02, .04, .06])
    root = np.loadtxt(directory / "body_quat.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(root, [[1, 0, 0, 0]] * 4)
    with pytest.raises(FileExistsError):
        writer.finish(tmp_path)


@pytest.mark.parametrize("bad", ["time", "quaternion"])
def test_invalid_reference_is_not_published(tmp_path, bad):
    first = build()
    second = build(now_ns=1_020_000_000)
    if bad == "time":
        second = replace(second, sample_monotonic_ns=first.sample_monotonic_ns)
    else:
        second = replace(second, body_quaternion=np.zeros(4))
    writer = SonicReferenceWriter()
    writer.append(first)
    writer.append(second)
    with pytest.raises(ValueError):
        writer.finish(tmp_path)
    assert not (tmp_path / "sonic_reference").exists()
