"""Export measured G1 body motion alongside an otherwise unchanged recording."""

from pathlib import Path

import numpy as np

from .schema import CONTROL_FPS

# Gather hardware/MuJoCo columns into SONIC's IsaacLab reference order.
# Matches mujoco_to_isaaclab in the native policy_parameters.hpp.
ISAAC_FROM_HARDWARE = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
REFERENCE_PATH = "sonic_reference/episode_000000"


class SonicReferenceWriter:
    """Buffer small body arrays; write CSVs only when the episode is committed.

    The debug publisher already restores absolute angles and hardware order.
    No default-pose offset is added here. Images, hands, SMPL and training
    columns are left to the existing LeRobot writer.
    """

    def __init__(self):
        self.positions = []
        self.velocities = []
        self.quaternions = []
        self.timestamps = []

    def append(self, frame):
        self.positions.append(frame.body_joint_positions.copy())
        self.velocities.append(frame.body_joint_velocities.copy())
        self.quaternions.append(frame.body_quaternion.copy())
        self.timestamps.append(frame.sample_monotonic_ns)

    def finish(self, dataset_root):
        if len(self.timestamps) < 2:
            raise ValueError("SONIC reference requires at least two body samples")
        time_ns = np.asarray(self.timestamps, dtype=np.int64)
        if np.any(np.diff(time_ns) <= 0):
            raise ValueError("SONIC reference sample times must increase")
        time_ns = time_ns - time_ns[0]
        # Native playback advances one row per 20 ms. Resample only the extra
        # CSVs so dropped capture rows do not speed up this reference motion.
        step_ns = 1_000_000_000 // CONTROL_FPS
        target_ns = np.arange(int(time_ns[-1] // step_ns) + 1, dtype=np.int64) * step_ns
        if len(target_ns) < 2:
            raise ValueError("SONIC reference duration must be at least 20 ms")

        def interpolate(values):
            values = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError("SONIC reference contains non-finite body values")
            return np.column_stack([
                np.interp(target_ns, time_ns, column) for column in values.T
            ])

        positions = interpolate(self.positions)[:, ISAAC_FROM_HARDWARE]
        velocities = interpolate(self.velocities)[:, ISAAC_FROM_HARDWARE]
        quaternions = np.asarray(self.quaternions, dtype=np.float64)
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms < 1e-8):
            raise ValueError("SONIC reference requires valid measured root quaternions")
        quaternions /= norms
        # q and -q represent the same rotation. Align their signs before
        # normalized linear interpolation to avoid an artificial rotation.
        flips = np.where(np.sum(quaternions[1:] * quaternions[:-1], axis=1) < 0, -1, 1)
        quaternions[1:] *= np.cumprod(flips)[:, None]
        quaternions = interpolate(quaternions)
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

        path = Path(dataset_root) / REFERENCE_PATH
        path.mkdir(parents=True, exist_ok=False)
        arrays = {
            "joint_pos": (positions, [f"joint_{i}" for i in range(29)]),
            "joint_vel": (velocities, [f"joint_vel_{i}" for i in range(29)]),
            "body_quat": (quaternions, [f"body_0_{axis}" for axis in "wxyz"]),
            # The live state stream does not measure world translation. G1
            # mode in the release/low_latency policies uses root orientation,
            # not root XYZ. These zeros are format placeholders, not odometry.
            "body_pos": (np.zeros((len(target_ns), 3)), [f"body_0_{axis}" for axis in "xyz"]),
        }
        for name, (values, header) in arrays.items():
            np.savetxt(path / f"{name}.csv", values, delimiter=",", fmt="%.9f",
                       header=",".join(header), comments="")
        (path / "metadata.txt").write_text(
            f"Body part indexes:\n[0]\n\nTotal timesteps: {len(target_ns)}\n", encoding="utf-8"
        )
        (path / "info.txt").write_text(
            "Measured G1 body reference; encoder mode g1 (0); 50 Hz.\n"
            "joint_pos: absolute radians, IsaacLab order, from g1_debug.body_q.\n"
            "joint_vel: measured rad/s, IsaacLab order, from g1_debug.body_dq.\n"
            "body_quat: measured root/IMU orientation, wxyz, normalized.\n"
            "body_pos: zero placeholders; world translation was NOT recorded.\n"
            "Time: 50 Hz from local capture sample times, without extrapolation.\n"
            "Resampling: linear angles/velocities, normalized shortest-arc quaternion interpolation.\n"
            "Hand actions and the scene are not part of this body reference.\n",
            encoding="utf-8",
        )
        return {
            "schema": "inspire.sonic_reference.v1",
            "path": REFERENCE_PATH,
            "fps": CONTROL_FPS,
            "frames": len(target_ns),
            "source_frames": len(time_ns),
            "source": "measured_g1_body",
            "encoder_mode": "g1",
            "joint_order": "isaaclab",
            "root_position_measured": False,
            "time_source": "sample_monotonic_ns",
            "max_source_gap_seconds": float(np.max(np.diff(time_ns))) / 1e9,
        }
