"""Local-only SONIC/MuJoCo basket recordings through existing factories."""

import argparse
import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np

from gear_sonic.utils.hand_control.registry import ROOT, deployment
from gear_sonic.utils.hand_control.management import DeviceLease
from .basket import BasketExpert, ISAAC_ORDER, SCENE, assess, render, variant


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run_sim(out):
    import mujoco
    from scipy.spatial.transform import Rotation
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

    params = json.loads((out / "parameters.json").read_text())

    class RecordingEnv(DefaultEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.elastic_band.point = np.array([0.0, 0.0, 0.80])
            m, d = self.mj_model, self.mj_data
            box = m.joint("box_joint").qposadr[0]
            d.qpos[box : box + 3] += params["basket_offset_m"]
            d.qpos[box + 3 : box + 7] = Rotation.from_euler("z", params["basket_yaw_rad"]).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(m, d)
            self.box = m.body("manipulatable_box").id
            if m.jnt_type[m.joint("box_joint").id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("basket must have a free joint")
            for i in range(m.neq):
                if m.eq_type[i] in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
                    if self.box in (m.eq_obj1id[i], m.eq_obj2id[i]):
                        raise ValueError("basket cannot be attached by an equality constraint")
            self.trace = []
            self.steps = 0
            self.contact_forces = np.zeros(3)

        def sim_step(self):
            if (out / "release").exists():
                self.elastic_band.enable = False
                (out / "release").unlink()
            result = super().sim_step()
            m, d = self.mj_model, self.mj_data
            if self.steps % 20 == 0:
                write(out / "current-qpos.tmp", {"monotonic_ns": time.monotonic_ns(), "qpos": d.qpos.tolist()})
                (out / "current-qpos.tmp").replace(out / "current-qpos.json")
            if self.steps % 4 == 0:
                self.contact_forces[:] = 0
                force = np.zeros(6)
                for i, c in enumerate(d.contact):
                    bodies = [int(m.geom_bodyid[int(g)]) for g in (c.geom1, c.geom2)]
                    if self.box not in bodies:
                        continue
                    other = bodies[1] if bodies[0] == self.box else bodies[0]
                    name = m.body(other).name or ""
                    side = 0 if name.startswith("left_inspire_") else 1 if name.startswith("right_inspire_") else 2
                    mujoco.mj_contactForce(m, d, i, force)
                    self.contact_forces[side] += np.linalg.norm(force[:3])
            external = max(
                np.linalg.norm(d.xfrc_applied[self.box]), np.linalg.norm(d.xfrc_applied[self.band_attached_link])
            )
            self.trace.append(
                (
                    time.monotonic_ns(),
                    float(d.time),
                    d.qpos.copy(),
                    d.qvel.copy(),
                    d.body(self.box).xpos.copy(),
                    self.contact_forces.copy(),
                    external,
                )
            )
            self.steps += 1
            return result

        def save(self):
            if self.trace:
                np.savez_compressed(
                    out / "physics.npz",
                    monotonic_ns=[r[0] for r in self.trace],
                    sim_time=[r[1] for r in self.trace],
                    qpos=[r[2] for r in self.trace],
                    qvel=[r[3] for r in self.trace],
                    basket_xyz=[r[4] for r in self.trace],
                    hand_contact_force=[r[5][:2] for r in self.trace],
                    other_contact_force=[r[5][2] for r in self.trace],
                    external_force=[r[6] for r in self.trace],
                )

    config = SimLoopConfig(
        interface="sim",
        hand="inspire",
        hand_config=str(out / "hand.yaml"),
        enable_onscreen=False,
        enable_offscreen=False,
    )
    device = deployment("inspire", "sim_shared", options={"hand_config": str(out / "hand.yaml")})
    settings, options = device.body_integration().simulation(config.load_wbc_yaml(), config)
    settings.update(ROBOT_SCENE=str(SCENE), INITIAL_KEYFRAME="photo_arms_down")
    options["env_factory"] = RecordingEnv
    with DeviceLease(device, "full_body_simulation"):
        sim = SimulatorFactory.create_simulator(settings, onscreen=False, offscreen=False, **options)
        try:
            sim.start()
        finally:
            sim.close()
            sim.sim_env.save()


def run_expert(out):
    from gear_sonic.utils.inference.sonic_session import SonicSession
    from gear_sonic.utils.hand_control.interface import EffectorAction
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    params = json.loads((out / "parameters.json").read_text())
    expert = BasketExpert(params)
    write(out / "reference.json", {"ik_errors": expert.ik_errors, "joint_names": expert.names})
    cfg = deployment("inspire", "sim_shared", options={"hand_config": str(out / "hand.yaml")})
    times = []
    tokens = []
    hands = []
    measured = []
    targets = []
    body_indices = []
    session = {"complete": False}
    try:
        with DeviceLease(cfg, "basket_simulation_expert"), ExitStack() as resources:
            body = SonicSession("tcp://127.0.0.1:5556", "tcp://127.0.0.1:5557", state_timeout=0.5)
            resources.callback(body.close)
            hand = cfg.create()
            resources.callback(hand.close)
            resources.callback(body.stop)
            hand.connect()
            previous = hand.seed_action(hand.get_state())
            hand.send_action(previous)
            deadline = time.monotonic() + 20
            while True:
                body.idle()
                body.command(planner=True)
                hand.send_action(previous)
                if body.feedback(required=False):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("SONIC CONTROL feedback missing")
                time.sleep(0.02)
            start = time.monotonic()
            released = False
            while time.monotonic() - start < 6:
                body.idle()
                body.feedback()
                hand.get_state()
                hand.send_action(previous)
                if time.monotonic() - start > 2 and not released:
                    (out / "release").touch()
                    released = True
                time.sleep(0.02)
            session["start_ns"] = time.monotonic_ns()
            deadline = time.monotonic()
            print("RECORDING_STARTED", flush=True)
            for tick in range(len(expert.targets) * params["speed_divisor"]):
                frame = min(tick // params["speed_divisor"], len(expert.targets) - 1)
                body.feedback()
                state = hand.get_state()
                if state.fault:
                    raise RuntimeError("hand backend fault: " + str(state.fault))
                desired = EffectorAction({side: tuple(expert.hands[frame]) for side in ("left", "right")})
                previous = hand.limit_action(previous, desired, 0.02)
                hand.send_action(previous)
                physics = json.loads((out / "current-qpos.json").read_text())
                if time.monotonic_ns() - physics["monotonic_ns"] > 500_000_000:
                    raise RuntimeError("simulator feedback stale")
                target = expert.command(frame, physics["qpos"], np.asarray(body.state["body_q"]))
                indices = np.clip(np.arange(frame, frame + 15), 0, len(expert.targets) - 1)
                q = (expert.targets[indices] + target - expert.targets[frame])[:, ISAAC_ORDER].astype(np.float32)
                payload = {
                    "joint_pos": q,
                    "joint_vel": np.gradient(q, 0.04, axis=0),
                    "body_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (15, 1)).astype(np.float32),
                    "frame_index": np.arange(tick, tick + 15, dtype=np.int64),
                }
                body.socket.send(pack_pose_message(payload, version=1))
                if tick < 3:
                    body.command(planner=False)
                    body.pose_selected = True
                times.append(time.monotonic_ns())
                tokens.append(np.array(body.state["token_state"]).copy())
                hands.append(np.array([previous.values[s] for s in ("left", "right")]))
                measured.append(np.array(body.state["body_q"]).copy())
                targets.append(target)
                body_indices.append(body.state["index"])
                deadline += 0.02
                time.sleep(max(0, deadline - time.monotonic()))
            session.update(end_ns=time.monotonic_ns(), complete=True)
            print("RECORDING_COMPLETE", flush=True)
    finally:
        write(out / "session.json", session)
        np.savez_compressed(
            out / "commands.npz",
            monotonic_ns=times,
            motion_token=tokens,
            hand_action=hands,
            measured_body=measured,
            reference_body=targets,
            sonic_index=body_indices,
        )


def sample_episode(out):
    import mujoco
    from scipy.spatial.transform import Rotation

    physics = np.load(out / "physics.npz")
    commands = np.load(out / "commands.npz")
    session = json.loads((out / "session.json").read_text())
    if not session["complete"]:
        return {"passed": False, "reason": "controller did not complete"}
    report = assess(physics, session["start_ns"], session["end_ns"])
    if not report["passed"]:
        return report
    start = max(session["start_ns"], int(commands["monotonic_ns"][0]))
    end = min(session["end_ns"], int(physics["monotonic_ns"][-1]), int(commands["monotonic_ns"][-1]))
    stamps = np.arange(start, end, 20_000_000, dtype=np.int64)
    pi = np.searchsorted(physics["monotonic_ns"], stamps, side="right") - 1
    ci = np.searchsorted(commands["monotonic_ns"], stamps, side="right") - 1
    ages = (stamps - physics["monotonic_ns"][pi]) / 1e9
    command_age = (stamps - commands["monotonic_ns"][ci]) / 1e9
    if min(pi) < 0 or min(ci) < 0 or max(ages) > 0.04 or max(command_age) > 0.08:
        return report | {"passed": False, "reason": "recording has excessive synchronization gaps"}
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    robot = deployment("inspire", "sim_shared").robot_model()
    state = physics["qpos"][pi][:, [model.joint(n).qposadr[0] for n in robot.joint_names]].astype(np.float32)
    quat = physics["qpos"][pi, 3:7]
    gravity = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).inv().apply(np.tile([0.0, 0.0, -1.0], (len(pi), 1)))
    np.savez_compressed(
        out / "samples.npz",
        qpos=physics["qpos"][pi],
        qvel=physics["qvel"][pi],
        state=state,
        left_hand=state[:, robot.get_hand_actuated_joint_indices("left")],
        right_hand=state[:, robot.get_hand_actuated_joint_indices("right")],
        projected_gravity=gravity,
        motion_token=commands["motion_token"][ci],
        hand_action=commands["hand_action"][ci],
        timestamp=np.arange(len(stamps)) / 50,
        monotonic_ns=stamps,
        physics_age_s=ages,
        command_age_s=command_age,
        physics_indices=pi,
        command_indices=ci,
    )
    return report | {
        "frames": len(stamps),
        "physics_age_max_s": float(max(ages)),
        "command_age_max_s": float(max(command_age)),
    }


def run_episode(out, seed, sim_python):
    import yaml

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    parameters = variant(seed)
    parameters["implementation_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(__file__).with_name("basket.py"))
    }
    write(out / "parameters.json", parameters)
    processes = []
    logs = []

    def launch(name, command, cwd=ROOT):
        log = (out / (name + ".log")).open("w")
        logs.append(log)
        p = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=os.environ | {"MUJOCO_GL": "egl", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)},
        )
        processes.append(p)
        return p

    def wait(p, condition, timeout):
        end = time.monotonic() + timeout
        while not condition():
            if p.poll() is not None:
                raise RuntimeError(f"local process exited {p.returncode}; {out}")
            if time.monotonic() > end:
                raise TimeoutError(str(out))
            time.sleep(0.1)

    try:
        for port in (5556, 5557, 18561, 18562):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
        raw = yaml.safe_load((ROOT / "gear_sonic/devices/inspire/assets/hand_config.yaml").read_text())
        raw["hardware"].update(command_endpoint="tcp://127.0.0.1:18561", state_endpoint="tcp://127.0.0.1:18562")
        (out / "hand.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        module = "gear_sonic.devices.inspire.training.rollout"
        sim = launch("simulation", [sim_python, "-B", "-m", module, "sim", str(out)])

        def sim_ready():
            with socket.socket() as sock:
                return sock.connect_ex(("127.0.0.1", 18562)) == 0

        wait(sim, sim_ready, 90)
        deploy = ROOT / "gear_sonic_deploy"
        body = launch(
            "sonic",
            [
                str(deploy / "target/release/g1_deploy_onnx_ref"),
                "lo",
                "policy/low_latency/model_decoder.onnx",
                "reference/example/",
                "--obs-config",
                "policy/low_latency/observation_config.yaml",
                "--encoder-file",
                "policy/low_latency/model_encoder.onnx",
                "--planner-file",
                "planner/target_vel/V2/planner_sonic.onnx",
                "--sim-initial-pose",
                str(ROOT / "gear_sonic/devices/inspire/assets/vla_initial_pose.txt"),
                "--hand",
                "external",
                "--input-type",
                "zmq_manager",
                "--output-type",
                "zmq",
                "--zmq-host",
                "127.0.0.1",
                "--disable-crc-check",
            ],
            deploy,
        )
        wait(body, lambda: "Init Done" in (out / "sonic.log").read_text(), 180)
        expert = launch("expert", [sim_python, "-B", "-m", module, "expert", str(out)])
        if expert.wait(timeout=100) != 0:
            raise RuntimeError("expert failed: " + str(out / "expert.log"))
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        for log in logs:
            log.close()
    report = sample_episode(out)
    write(out / "quality.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=["sim", "expert", "episode", "render"])
    p.add_argument("directory", type=Path)
    p.add_argument("--seed", type=int, default=20261021)
    a = p.parse_args()
    out = a.directory.resolve()
    if a.role == "sim":
        run_sim(out)
    elif a.role == "expert":
        run_expert(out)
    elif a.role == "render":
        render(out, out / "ego_view.mp4")
    else:
        import sys

        print(json.dumps(run_episode(out, a.seed, sys.executable)), flush=True)


if __name__ == "__main__":
    main()
