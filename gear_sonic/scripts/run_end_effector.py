"""Inspect, control, retarget or record the built-in Inspire hand."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from gear_sonic.utils.hand_control.interface import EffectorAction
from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment, registrations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="inspire")
    parser.add_argument("--backend", default="sim")
    parser.add_argument("--deployment", help="Local deployment JSON; overrides device/backend")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--hand-config", help="Legacy Inspire deployment option")
    parser.add_argument("--operation", choices=("inspect", "feedback", "target", "sequence", "teleop", "audit", "export"), default="inspect")
    parser.add_argument("--source", help="Recorded trajectory directory for audit/export")
    parser.add_argument("--output", help="New LeRobot directory for export")
    parser.add_argument("--targets", help='Six angles per hand: {"left":[0,0,0,0.4,0,0],"right":[0,0,0,0,0,0]}')
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--record", help="New trajectory directory; existing data is never overwritten")
    parser.add_argument("--scene", help="Custom standalone MuJoCo scene XML")
    parser.add_argument("--view", action="store_true", help="Open the local MuJoCo viewer")
    parser.add_argument("--task", default="End effector movement")
    parser.add_argument("--outcome", choices=("unlabeled", "success", "failure"), default="unlabeled")
    parser.add_argument("--list", action="store_true")
    return parser


def run(args):
    if args.list:
        result = [entry.public() for entry in registrations(args.registry_dir).values()]
        print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    if args.operation in ('audit', 'export'):
        if not args.source or (args.operation == 'export' and not args.output):
            raise ValueError('audit requires --source; export requires --source and --output')
        from gear_sonic.utils.data_collection.end_effector import audit_trajectory, export_lerobot
        result = audit_trajectory(args.source)
        if args.operation == 'export':
            result['export'] = str(export_lerobot(args.source, args.output))
        print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 1800:
        raise ValueError("seconds must be finite and in (0, 1800]")
    config = deployment(args.device, args.backend, config_path=args.deployment,
                        registry_dir=args.registry_dir,
                        options={"hand_config":args.hand_config} if args.hand_config else None)
    if getattr(args, 'scene', None):
        from gear_sonic.utils.mujoco_sim.scenes import standalone_scene
        config = standalone_scene(config, args.scene)
    reg = config.registration
    reg.check_dependencies(config.backend)
    if args.operation == "inspect":
        if reg.backends[config.backend].get('factory'):
            config.create()  # Constructors validate options; connect is never called.
        result = config.snapshot() | {"connections_opened":False}
        print(json.dumps(result, ensure_ascii=False, indent=2)); return result
    if "control" not in reg.backends[config.backend]["capabilities"]:
        raise ValueError("selected backend does not support direct end-effector control")
    if args.record and "capture" not in reg.backends[config.backend]["capabilities"]:
        raise ValueError("selected backend does not support capture")
    if args.record and reg.backends[config.backend].get('capture_source') != 'local':
        raise ValueError("bench recording requires local simulation; use the stream recorder for shared-world/hardware capture")
    if args.record and args.operation == 'feedback':
        raise ValueError('recording actions requires target, sequence or teleop operation')
    if args.targets and args.operation != 'target':
        raise ValueError('--targets requires --operation target')
    if args.operation == "teleop":
        return run_optical(args, config)
    device = config.create()
    target = EffectorAction(json.loads(args.targets)) if args.targets else None
    if args.operation == "target" and target is None:
        raise ValueError("target operation requires --targets")
    if target:
        device.validate_action(target)
    writer = session = reader = None
    command_stamps, feedback_stamps = [], []
    try:
        device.connect()
        if args.view:
            device.open_viewer()
        state = device.get_state()
        body = device.body_observation() if args.record else ()
        if args.record:
            from gear_sonic.utils.data_collection.end_effector import TrajectoryWriter
            writer = TrajectoryWriter(args.record, config, task=args.task, outcome=args.outcome,
                                      source=args.operation, body_state_names=device.body_state_names())
            image = device.render()
            image_ns = time.monotonic_ns()
        state = device.get_state()
        previous = device.seed_action(state)
        period = 1 / device.description.command_hz
        begin = deadline = last_sent = time.monotonic()
        seeded = False
        while time.monotonic()-begin < args.seconds:
            if args.view and not device.sync_viewer():
                break
            now = time.monotonic()
            intent = target or previous
            if args.operation == "sequence":
                fraction = .5-.5*math.cos(2*math.pi*(now-begin)/args.seconds)
                intent = EffectorAction({mount: tuple(c.lower+fraction*.6*(c.upper-c.lower) for c in device.description.action_channels) for mount in device.mounts})
            if args.operation != "feedback":
                submitted = device.limit_action(previous, intent, min(period, max(1e-9, now-last_sent))) if seeded else previous
                receipt = device.send_action(submitted)
                previous = submitted
                seeded = True
                last_sent = receipt.submitted_monotonic_ns / 1e9
                command_stamps.append(receipt.submitted_monotonic_ns)
            state = device.get_state()
            state.require_fresh(time.monotonic_ns(), device.description)
            if not feedback_stamps or feedback_stamps[-1] != state.received_monotonic_ns:
                feedback_stamps.append(state.received_monotonic_ns)
            if writer:
                if time.monotonic_ns()-image_ns > 80_000_000:
                    image=device.render(); image_ns=time.monotonic_ns()
                writer.append(state=state, receipt=receipt, intent=intent, image=image, image_timestamp_ns=image_ns,
                              body_state=device.body_observation())
            deadline += period
            time.sleep(max(0, deadline-time.monotonic()))
            if deadline < time.monotonic()-period:
                deadline=time.monotonic()
        def hz(stamps):
            return (len(stamps)-1)*1e9/(stamps[-1]-stamps[0]) if len(stamps)>1 else None
        result={"device":reg.description.device_id,"backend":config.backend,"target_hz":device.description.command_hz,
                "submitted":len(command_stamps),"command_hz":hz(command_stamps),"observed_feedback_hz":hz(feedback_stamps),
                "final_state":dict(state.values),"fault":state.fault,"mode":reg.backends[config.backend]['mode']}
        if writer:
            output=writer.commit(result)
            from gear_sonic.utils.data_collection.end_effector import audit_trajectory
            result["trajectory"]=str(output); result["audit"]=audit_trajectory(output)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return result
    finally:
        if session is not None:
            session.close()
        if reader is not None:
            reader.stop()
        device.close()
        if writer is not None:
            writer.abort()


def run_optical(args, config, *, reader=None, retargeter=None):
    """PICO acquisition is shared; only the registered retargeter varies by device."""
    from gear_sonic.utils.teleop.hand_tracking import OpticalPicoReader
    from gear_sonic.utils.teleop.end_effector import OpticalSession
    from gear_sonic.utils.data_collection.end_effector import TrajectoryWriter, audit_trajectory
    reader = reader or OpticalPicoReader(body=False)
    mapper = retargeter or config.retargeter()
    writer = session = None
    image = None
    image_ns = 0

    def factory():
        nonlocal image, image_ns, writer
        device = config.create()
        try:
            if args.record or getattr(args,'view',False):
                # Warm rendering before the first command starts its watchdog.
                device.connect()
            if getattr(args,'view',False):
                device.open_viewer()
            if args.record:
                image = device.render(); image_ns = time.monotonic_ns()
                body = device.body_observation()
                writer = TrajectoryWriter(args.record, config, task=args.task, outcome=args.outcome,
                                          source="optical_tracking", body_state_names=device.body_state_names())
            return device
        except BaseException:
            device.close()
            raise

    def observe(snapshot, device):
        nonlocal image, image_ns
        if getattr(args,'view',False) and not device.sync_viewer():
            session.request_stop()
        if writer is None:
            return
        if time.monotonic_ns()-image_ns > 80_000_000:
            image=device.render();image_ns=time.monotonic_ns()
        writer.append(state=snapshot.measured, receipt=snapshot.receipt, intent=snapshot.intent,
                      image=image, image_timestamp_ns=image_ns, body_state=device.body_observation(),
                      tracking_valid=snapshot.tracking_valid)

    try:
        reader.start()
        deadline=time.monotonic()+30
        while reader.get_latest() is None:
            if time.monotonic()>deadline:raise TimeoutError("no PICO hand frame")
            time.sleep(.01)
        mapper.prepare(reader)
        session=OpticalSession(reader, config, retargeter=mapper, backend_factory=factory, observer=observe)
        session.start()
        print('PICO ready. End-effector control active; Ctrl+C stops.', flush=True)
        deadline=time.monotonic()+args.seconds if args.seconds is not None else float('inf')
        while time.monotonic()<deadline and session.is_running():
            session.check();time.sleep(.02)
        session.close()
        result=session.summary()
        if writer:
            output=writer.commit(result)
            result.update(trajectory=str(output),audit=audit_trajectory(output))
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    finally:
        try:
            if session:session.close()
        finally:
            reader.stop()
            if writer:writer.abort()


def main():
    parser=build_parser()
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        parser.exit(130,"End-effector control stopped.\n")
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f"End-effector failed: {exc}\n")


if __name__=="__main__":
    main()
