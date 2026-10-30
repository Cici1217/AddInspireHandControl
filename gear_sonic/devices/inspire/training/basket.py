"""Basket expert references and physical success checks; never moves an object."""

import json
from pathlib import Path
import numpy as np
from gear_sonic.utils.hand_control.registry import ROOT

SCENE = ROOT / "scenes/inspire/basket_chair/scene.xml"
FPS = 50
ISAAC_ORDER = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]


def variant(seed):
    rng = np.random.default_rng(seed)
    return {
        "seed": seed,
        "basket_offset_m": [float(rng.uniform(-0.004, 0.004)), float(rng.uniform(-0.003, 0.003)), 0.0],
        "basket_yaw_rad": float(rng.uniform(-0.012, 0.012)),
        "light_scale": float(rng.uniform(0.90, 1.10)),
        "camera_name": "head_camera",
        "speed_divisor": 2,
    }


class BasketExpert:
    """IK reference + correction of SONIC tracking, at 50 Hz.

    MuJoCo here is used only for kinematics. The shared simulator owns dynamics.
    References enter the original SONIC encoder. Hands use the registered adapter
    and its unchanged speed/collision limits.
    """

    def __init__(self, parameters):
        import mujoco
        from scipy.spatial.transform import Rotation

        self.mj = mujoco
        self.model = m = mujoco.MjModel.from_xml_path(str(SCENE))
        self.reference = ref = mujoco.MjData(m)
        self.live = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, ref, m.key("photo_arms_down").id)
        ref.qpos[2] = 0.757
        motors = [i for i in range(m.nu) if "_inspire_" not in m.actuator(i).name]
        joints = m.actuator_trnid[motors, 0]
        self.names = [m.joint(int(j)).name for j in joints]
        self.addresses = qa = m.jnt_qposadr[joints]
        self.bounds = m.jnt_range[joints]
        self.bodies = [m.body(s + "_inspire_base").id for s in ("left", "right")]
        self.arms = [np.arange(15, 22), np.arange(22, 29)]
        ref.qpos[qa[14]] = 0.4
        self.initial = ref.qpos.copy()
        waypoints = [(0.0, ref.qpos[qa].copy(), 0.0), (1.0, ref.qpos[qa].copy(), 0.0)]
        base_rotations = [
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
        ]
        yaw = Rotation.from_euler("z", parameters["basket_yaw_rad"]).as_matrix()
        offset = np.array(parameters["basket_offset_m"])
        self.ik_errors = []
        for t, dz, dy, closure in [
            (4.0, 0.12, 0.035, 0.0),
            (6.0, 0.0, 0.0, 0.0),
            (8.0, 0.0, 0.0, 1.4),
            (11.0, 0.20, 0.0, 1.4),
            (14.0, 0.20, 0.0, 1.4),
        ]:
            if t >= 11:
                ref.qpos[qa[14]] = 0.1
            tilt = 85 if t >= 11 else 50
            for side, sign in enumerate((1, -1)):
                R = yaw @ Rotation.from_euler("y", -tilt, degrees=True).as_matrix() @ base_rotations[side]
                grip = np.array([0.405, 0, 0.430]) + offset + yaw @ np.array([-0.025, sign * 0.221, 0.158])
                # Bring the lifted basket toward the body, clear of the chair back.
                if t >= 11:
                    grip[0] -= 0.05
                pos = grip - R @ np.array([-0.025, -0.155, 0.0]) + np.array([0.0, sign * dy, dz])
                q, error = self.solve(ref, side, pos, R, ref.qpos[qa[self.arms[side]]], natural=True)
                ref.qpos[qa[self.arms[side]]] = q
                self.ik_errors.append([t, side, *error])
            waypoints.append((t, ref.qpos[qa].copy(), closure))
        self.targets, self.hands = [], []
        for frame in range(701):
            t = frame / FPS
            k = next((k for k, w in enumerate(waypoints[1:]) if t <= w[0]), len(waypoints) - 2)
            t0, q0, h0 = waypoints[k]
            t1, q1, h1 = waypoints[k + 1]
            u = np.clip((t - t0) / (t1 - t0), 0, 1)
            u = u * u * (3 - 2 * u)
            self.targets.append(q0 + (q1 - q0) * u)
            self.hands.append([h0 + (h1 - h0) * u] * 4 + [0.0, 0.0])
        self.targets = np.array(self.targets)
        self.targets[:51, 14] = np.linspace(0, 0.4, 51)
        self.hands = np.array(self.hands)
        self.integral = np.zeros(29)
        self.correction = np.zeros(29)

    def solve(self, data, side, pos, orientation, seed, *, natural=False):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        indices = self.arms[side]
        addresses = self.addresses[indices]
        bid = self.bodies[side]
        lo, hi = self.bounds[indices, 0] + 0.01, self.bounds[indices, 1] - 0.01
        if natural:
            lo, hi = self.bounds[indices, 0] + 0.005, self.bounds[indices, 1] - 0.005
            lo[3] = 0.1
            hi[3] = 1.8
            lo[4:] = -0.6
            hi[4:] = 0.6

        def residual(q):
            data.qpos[addresses] = q
            self.mj.mj_kinematics(self.model, data)
            rot = Rotation.from_matrix(orientation @ data.xmat[bid].reshape(3, 3).T).as_rotvec()
            return np.r_[5 * (data.xpos[bid] - pos), rot, 0.02 * (q - seed)]

        result = least_squares(
            residual, np.clip(seed, lo + 0.0001, hi - 0.0001), bounds=(lo, hi), max_nfev=150 if natural else 25
        )
        error = residual(result.x)
        return result.x, (float(np.linalg.norm(error[:3]) / 5), float(np.linalg.norm(error[3:6])))

    def command(self, frame, actual_qpos, measured_body):
        desired = self.targets[frame].copy()
        if frame % 5 == 0:
            ref, live = self.reference, self.live
            ref.qpos[:] = self.initial
            ref.qpos[self.addresses] = desired
            self.mj.mj_kinematics(self.model, ref)
            live.qpos[:] = actual_qpos
            live.qpos[self.addresses[12:15]] = desired[12:15]
            for side, indices in enumerate(self.arms):
                bid = self.bodies[side]
                desired[indices], _ = self.solve(
                    live, side, ref.xpos[bid].copy(), ref.xmat[bid].reshape(3, 3).copy(), desired[indices]
                )
            self.correction = desired - self.targets[frame]
        else:
            desired += self.correction
        self.integral[12:] += (desired - measured_body)[12:] * 0.02 * 0.8
        np.clip(self.integral, -0.8, 0.8, out=self.integral)
        return np.clip(desired + self.integral, self.bounds[:, 0] + 0.01, self.bounds[:, 1] - 0.01)


def assess(physics, start_ns, end_ns):
    """Require an unsupported, bilateral lift throughout the final two seconds."""
    t = (physics["monotonic_ns"] - start_ns) / 1e9
    duration = (end_ns - start_ns) / 1e9
    active = (t >= 0) & (t <= duration)
    final = active & (t >= duration - 2.0)
    initial = active & (t < 1.0)
    if final.sum() < 50 or initial.sum() < 20:
        return {"passed": False, "reason": "missing complete beginning or hold interval"}
    z = physics["basket_xyz"][:, 2]
    lift = float(np.min(z[final]) - np.median(z[initial]))
    bilateral = float(np.mean(np.all(physics["hand_contact_force"][final] > 0.02, axis=1)))
    support = float(np.max(physics["other_contact_force"][final]))
    root_min = float(np.min(physics["qpos"][active, 2]))
    force = float(np.max(physics["external_force"][active]))
    finite = all(np.isfinite(physics[k]).all() for k in ("qpos", "basket_xyz", "hand_contact_force"))
    passed = finite and lift > 0.10 and bilateral >= 0.90 and support < 0.05 and root_min > 0.65 and force < 1e-9
    return {
        "passed": bool(passed),
        "minimum_hold_lift_m": lift,
        "bilateral_hold_fraction": bilateral,
        "nonhand_support_force_N": support,
        "minimum_root_height_m": root_min,
        "maximum_external_force_N": force,
        "duration_s": duration,
        "hold_seconds": 2.0,
        "physics_grasp_verified": bool(passed),
        "basket_pose_assigned_after_reset": False,
        "object_constraints_added": False,
        "body_support_during_recording": False,
    }


def render(episode, output):
    """Render saved dynamic states exactly; no estimated or altered basket pose."""
    import cv2
    import mujoco
    import subprocess

    episode, output = Path(episode), Path(output)
    sample = np.load(episode / "samples.npz")
    params = json.loads((episode / "parameters.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    cam = model.camera("head_camera").id
    model.light_diffuse[:] *= params["light_scale"]
    renderer = mujoco.Renderer(model, 480, 640)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "640x480",
        "-r",
        "50",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = []
    preview_indices = set(np.linspace(0, len(sample["qpos"]) - 1, 8, dtype=int))
    try:
        for i, qpos in enumerate(sample["qpos"]):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="head_camera")
            process.stdin.write(renderer.render().tobytes())
            if i in preview_indices:
                renderer.update_scene(data, camera="basket_chair_front")
                image = renderer.render().copy()
                cv2.putText(image, f"{i / 50:.1f}s", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                frames.append(image)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("video encoder failed")
    finally:
        renderer.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    grid = np.concatenate([np.concatenate(frames[:4], 1), np.concatenate(frames[4:], 1)])
    cv2.imwrite(str(episode / "preview.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    (episode / "camera.json").write_text(
        json.dumps(
            {
                "name": "head_camera",
                "width": 640,
                "height": 480,
                "fovy_deg": float(model.cam_fovy[cam]),
                "local_quat_wxyz": model.cam_quat[cam].tolist(),
                "local_position": model.cam_pos[cam].tolist(),
                "frames": len(sample["qpos"]),
            },
            indent=2,
        )
    )
