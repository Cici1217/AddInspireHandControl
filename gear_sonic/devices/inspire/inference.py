"""Original native6 asynchronous VLA, integrated with current device/SONIC sessions.

The existing adapter owns all hand communication and protection. This module
owns observation freshness, RTC chunks, the 50 Hz schedule and SONIC mode changes.
"""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from gear_sonic.utils.hand_control.interface import EffectorAction
from gear_sonic.utils.hand_control.management import DeviceLease, atomic_json
from gear_sonic.utils.hand_control.registry import ROOT
from gear_sonic.utils.inference.policy_client import PolicyClient
from gear_sonic.utils.inference.ready_pose import UPPER_ORDER, parse_ready_pose
from gear_sonic.utils.inference.sonic_session import SonicSession
from .policy import HORIZON, ObservationBuilder, constant_chunk, rtc_options
from .scheduling import AsyncPolicyWorker, BodyTokenContinuityGuard, TimedActionBuffer

RATE = 50
STATE_TIMEOUT = .25
MAXIMUM_HOLD = .10
MAXIMUM_LATENESS = .020


def record_failure(report, exc):
    """Retain the triggering failure when control cleanup also raises."""
    chain = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(f'{type(current).__name__}: {current}')
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    report['error'] = chain[0]
    report['error_chain'] = list(reversed(chain))
    report['traceback'] = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for cause in report['error_chain'][:-1]:
        print(f'VLA_CAUSE {cause}', flush=True)


def limit_for_tick(hand, previous, desired, last_sent_ns, now_ns):
    # A late tick followed by catch-up can be shorter than 20ms. The driver
    # measures actual send intervals, so never ramp using a fictitious full tick.
    elapsed = max(1e-9, min(1/RATE, (now_ns-last_sent_ns)/1e9))
    return hand.limit_action(previous, desired, elapsed)


def enter_ready():
    import select
    import sys
    if not select.select([sys.stdin], [], [], 0)[0]:
        return False
    if sys.stdin.readline() == '':
        raise RuntimeError('VLA terminal input closed')
    return True


def validate_result(result, hand, guard):
    if result.error is not None:
        raise RuntimeError(f'PolicyServer request failed: {result.error}') from result.error
    if result.chunk is None or result.chunk.horizon != HORIZON:
        raise ValueError('VLA requires exactly 40 body64/left6/right6 action frames')
    if result.info.get('rtc_applied') is not True:
        raise RuntimeError('PolicyServer did not apply RTC; use the original RTC-enabled GR00T runtime')
    guard.validate_chunk(result.chunk)
    for index in range(HORIZON):
        hand.validate_action(EffectorAction({'left': result.chunk.left_hand[index],
                                            'right': result.chunk.right_hand[index]}))


def load_pose(path, config):
    if not path:
        return None
    import xml.etree.ElementTree as ET
    from gear_sonic.utils.hand_control.models import resource_path
    values = np.loadtxt(path)
    if values.shape != (29,) or not np.isfinite(values).all():
        raise ValueError('initial pose requires 29 finite angles')
    model = ET.parse(resource_path(config, 'mjcf')).getroot()
    joints = {j.get('name'): j for j in model.iter('joint') if j.get('name')}
    names = [a.get('joint') for a in model.find('actuator') if '_inspire_' not in a.get('joint', '')]
    if len(names) != 29:
        raise ValueError('initial pose requires the G1 body29 actuator order')
    for name, value in zip(names, values):
        low, high = map(float, joints[name].get('range').split())
        if not low <= value <= high:
            raise ValueError('initial pose outside limit: '+name)
    return values


def run(args, config):
    if config.backend not in ('sim_shared', 'real'):
        raise ValueError('VLA requires the shared G1 body and Inspire hand backend')
    if args.auto_start and config.backend != 'sim_shared':
        raise ValueError('automatic start is only allowed for local simulation tests')
    if args.initial_pose and config.backend != 'sim_shared':
        raise ValueError('initial pose override is only allowed in simulation')
    ready_values = getattr(args, 'ready_pose_values', None)
    if ready_values and args.initial_pose:
        raise ValueError('choose ready pose or legacy simulation initial pose')
    if ready_values:
        pose = np.asarray(parse_ready_pose(ready_values))
    else:
        pose = load_pose(args.initial_pose, config)
    summary = {'device': config.registration.description.device_id, 'backend': config.backend,
               'target': args.target, 'prompt': args.prompt,
               'control_hz': RATE, 'action_horizon': HORIZON, 'rtc_overlap_steps': 20,
               'rtc_frozen_steps': 5, 'state_timeout_s': STATE_TIMEOUT,
               'maximum_hold_s': MAXIMUM_HOLD, 'maximum_lateness_s': MAXIMUM_LATENESS,
               'control_connections_opened': False, 'shadow': args.shadow}
    if not args.execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2)); print('VLA_CHECK_OK 未连接控制。', flush=True)
        return summary
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
    model = config.robot_model()
    report = summary | {'deployment': config.snapshot(), 'frames_sent': 0, 'policy_requests': 0, 'policy_results': 0,
                        'hand_limited_frames': 0, 'maximum_hand_target_error_rad': 0., 'completed': False,
                        'initial_pose': args.initial_pose, 'ready_pose': pose.tolist() if ready_values else None, 'duration_s': args.duration,
                        'policy_endpoint': f'tcp://{args.policy_host}:{args.policy_port}',
                        'body_endpoint': args.body_endpoint, 'state_endpoint': args.state_endpoint,
                        'camera_endpoint': f'tcp://{args.camera_host}:{args.camera_port}'}
    output = ROOT/'outputs/vla_runs'/(time.strftime('%Y%m%d-%H%M%S-')+uuid.uuid4().hex[:8]+'.json')
    guard = BodyTokenContinuityGuard()
    started = time.monotonic()
    hand = body = None
    finished = False
    try:
        with DeviceLease(config, 'vla_shadow' if args.shadow else 'vla'), ExitStack() as owned:
            hand = config.create(); owned.callback(hand.close); hand.connect()
            report['control_connections_opened'] = True
            body = SonicSession(args.body_endpoint, args.state_endpoint, state_timeout=STATE_TIMEOUT)
            owned.callback(body.close)
            camera = ComposedCameraClientSensor(server_ip=args.camera_host, port=args.camera_port, include_depth=False)
            owned.callback(camera.close)
            builder = ObservationBuilder(camera, model, hand.description, args.prompt)
            worker = AsyncPolicyWorker(lambda: PolicyClient(args.policy_host, args.policy_port, timeout_ms=5000))
            owned.callback(lambda: worker.close(join_timeout_s=6))
            def release_control():
                if not finished:
                    try: body.stop()
                    finally: hand.close()
            # Stop control before waiting for an in-flight inference to finish.
            owned.callback(release_control)
            worker.wait_ready(6)

            def feedback(required=True):
                body.feedback(required=required)
                state = hand.get_state()
                state.require_fresh(time.monotonic_ns(), hand.description)
                if (time.monotonic_ns()-state.received_monotonic_ns)/1e9 > STATE_TIMEOUT:
                    raise RuntimeError('VLA hand feedback is stale')
                return state

            def idle():
                if pose is None:
                    body.idle()
                else:
                    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message
                    body.socket.send(build_planner_message(0,[0,0,0],[1,0,0],
                                     upper_body_position=pose[list(UPPER_ORDER)], upper_body_velocity=np.zeros(17)))

            state = feedback(required=False)
            hold = hand.seed_action(state)
            if not args.shadow:
                print('VLA_READY 保留支撑；按 Enter 启动 SONIC 站立。', flush=True)
                while not args.auto_start and not enter_ready():
                    feedback(required=False); time.sleep(.02)
                print('VLA_STARTING 等待身体反馈和站立稳定。', flush=True)
                deadline = time.monotonic()+args.startup_timeout
                hand.send_action(hold)
                body.last_feedback = None
                while True:
                    idle(); body.command(planner=True); hand.send_action(hold); feedback(required=False)
                    if body.last_feedback is not None:
                        break
                    if time.monotonic()>deadline:
                        raise TimeoutError('no SONIC feedback after start')
                    time.sleep(.02)
                steady_until = time.monotonic()+3
                while time.monotonic()<steady_until:
                    idle(); hand.send_action(hold); feedback(); time.sleep(.02)
                print('VLA_STANDING 确认站稳并按现场方式释放支撑；按 Enter 开始推理。', flush=True)
                while not args.auto_start and not enter_ready():
                    idle(); hand.send_action(hold); feedback(); time.sleep(.02)
                # Check the scene pose after the operator releases support. The
                # elastic band changes body dynamics and must not gate that button.
                if pose is not None:
                    deadline=time.monotonic()+args.startup_timeout
                    stable_since=None; next_report=0.
                    while True:
                        idle(); hand.send_action(hold); feedback(); now=time.monotonic()
                        errors=np.abs(np.asarray(body.state['body_q'])[12:]-pose[12:])
                        error=float(np.max(errors)); index=12+int(np.argmax(errors))
                        if error<=.15:
                            if stable_since is None: stable_since=now
                            if now-stable_since>=.5: break
                        else: stable_since=None
                        if now>=next_report:
                            print(f'VLA_POSE_WAIT error_rad={error:.4f} body_index={index} measured={body.state["body_q"][index]:.4f} target={pose[index]:.4f}',flush=True)
                            next_report=now+1
                        if now>deadline: raise TimeoutError(f'VLA initial pose did not settle; upper-body error={error:.4f} rad')
                        time.sleep(.02)
            else:
                # Observe only. No start, mode, arm or action is sent.
                deadline = time.monotonic()+args.startup_timeout
                while body.last_feedback is None:
                    feedback(required=False)
                    if time.monotonic()>deadline: raise TimeoutError('shadow mode needs an already running body state stream')
                    time.sleep(.02)

            def observation():
                state = feedback()
                return builder.build(body.state, state, time.monotonic_ns())

            def initial_result():
                state = feedback()
                seed = hand.seed_action(state)
                token = np.asarray(body.state['token_state'], dtype=np.float32)
                guard.seed(token)
                history = constant_chunk(token, seed)
                requested = False
                deadline = time.monotonic()+args.startup_timeout
                while time.monotonic()<deadline:
                    if not args.shadow: idle(); hand.send_action(hold)
                    feedback()
                    if not requested:
                        obs = observation()
                        if obs is not None:
                            worker.submit(obs, options=rtc_options(history)); requested=True
                            report['policy_requests'] += 1
                    result = worker.poll()
                    if result is not None:
                        validate_result(result, hand, guard)
                        guard.validate_transition(result.chunk.motion_token[0])
                        report['policy_results'] += 1
                        return result
                    time.sleep(.02)
                raise TimeoutError('waiting for fresh camera / first RTC policy result')

            first = initial_result()
            if args.shadow:
                report['completed'] = True
                report['action_shapes'] = {k: list(v.shape) for k,v in vars(first.chunk).items()}
                print('VLA_SHADOW_OK 模型回复及动作检查通过；未发送控制命令。', flush=True)
                finished=True
                return report
            # Warm-up inference may exceed 0.8s. Rebase only this stationary seed,
            # as in the original VLA; all subsequent chunks use their real age.
            now_ns = time.monotonic_ns()
            timeline = TimedActionBuffer(control_rate_hz=RATE, maximum_hold_s=MAXIMUM_HOLD)
            timeline.activate(replace(first, submitted_monotonic_ns=now_ns, completed_monotonic_ns=now_ns),
                              activated_monotonic_ns=now_ns)
            feedback(); guard.seed(np.asarray(body.state['token_state']))
            guard.validate_transition(first.chunk.motion_token[0])
            body.token(first.chunk.motion_token[0], 0); time.sleep(.02)
            body.command(planner=False); body.pose_selected=True
            print('VLA_STARTED 身体与双手以 50 Hz 执行；Ctrl+C 停止。', flush=True)
            previous = hold
            last_sent_ns = time.monotonic_ns()
            stream_start = deadline = time.monotonic()
            requested_generation = None
            while not args.duration or time.monotonic()-stream_start < args.duration:
                late = max(0., time.monotonic()-deadline)
                if late>MAXIMUM_LATENESS:
                    raise RuntimeError(f'50 Hz VLA deadline missed by {late:.3f}s')
                feedback()
                result = worker.poll()
                if result is not None:
                    validate_result(result, hand, guard)
                    candidate = TimedActionBuffer(control_rate_hz=RATE, maximum_hold_s=MAXIMUM_HOLD)
                    activation = candidate.activate(result)
                    guard.validate_transition(candidate.sample().motion_token)
                    timeline = candidate
                    report['policy_results'] += 1
                    print(f'VLA_CHUNK generation={result.generation} latency_ms={activation.latency_s*1000:.1f} skipped={activation.skipped_frames}', flush=True)
                action = timeline.sample()
                overlap = HORIZON-action.chunk_index
                if not worker.busy and requested_generation != action.generation and 5<overlap<=20:
                    obs = observation()
                    if obs is not None:
                        worker.submit(obs, options=rtc_options(timeline.chunk, overlap))
                        requested_generation = action.generation
                        report['policy_requests'] += 1
                desired = EffectorAction({'left': action.left_hand, 'right': action.right_hand})
                limited = limit_for_tick(hand, previous, desired, last_sent_ns, time.monotonic_ns())
                delta = float(max(abs(a-b) for side in hand.mounts for a,b in zip(limited.values[side], desired.values[side])))
                report['maximum_hand_target_error_rad'] = max(report['maximum_hand_target_error_rad'], delta)
                report['hand_limited_frames'] += int(delta>.005)
                guard.validate_and_commit(action.motion_token)
                last_sent_ns = hand.send_action(limited).submitted_monotonic_ns
                body.token(action.motion_token, report['frames_sent']+1)
                previous=limited; report['frames_sent'] += 1
                if report['frames_sent']%RATE == 0:
                    print(f'VLA_PROGRESS frames={report["frames_sent"]} seconds={time.monotonic()-stream_start:.1f}', flush=True)
                deadline += 1/RATE
                time.sleep(max(0.,deadline-time.monotonic()))
            report['command_hz'] = report['frames_sent']/(time.monotonic()-stream_start)
            for index in range(40):
                idle(); feedback(); hand.send_action(previous)
                if index<3: body.command(planner=True)
                time.sleep(.02)
            body.pose_selected=False; hand.close(); finished=True; report['completed']=True
            print('VLA_COMPLETE 推理结束，身体回到站立，手部控制已释放。', flush=True)
    except BaseException as exc:
        record_failure(report, exc)
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-started
        report['maximum_token_step'] = guard.maximum_observed_step
        atomic_json(output, report)
        print(f'VLA_REPORT {output}', flush=True)
    return report
