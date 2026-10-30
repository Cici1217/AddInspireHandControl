"""Captured G1/Inspire actions → SONIC pose stream + registered native6 adapter.

Ported from the old Inspire replay_dataset_sim/replay_episode implementation.
The dataset is always local; only the selected deployment decides sim vs real.
Body references use either recorded SMPL or exported measured G1 motion.
Both pass through SONIC; feedback angles are never sent directly to motors.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import select
import sys
import time
import uuid

import numpy as np

from gear_sonic.utils.hand_control.interface import EffectorAction, EffectorError
from gear_sonic.utils.hand_control.management import DeviceLease, atomic_json
from gear_sonic.utils.hand_control.registry import ROOT, read_json
from .config import deployment_config, load_mapping
from .contract import JOINT_ORDER

FPS = 50


def matrix(value, columns, name):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != columns or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must contain finite frames × {columns} values')
    result = result.copy(); result.setflags(write=False)
    return result


def mapping_semantics(document):
    """Deployment IP, serial paths and simulation asset paths are not action units."""
    keys = ('domain_index', 'direction', 'scale', 'offset', 'position_limit', 'velocity_limit')
    return {key: document.get(key) for key in ('schema', 'joint_order', 'position_unit')} | {
        'joints': {side: {name: {k: document['joints'][side][name][k] for k in keys}
                          for name in JOINT_ORDER} for side in ('left', 'right')}}


@dataclass(frozen=True)
class ReplayEpisode:
    dataset: Path
    index: int
    source: str
    sha256: str
    timestamps: np.ndarray
    left: np.ndarray
    right: np.ndarray
    smpl_pose: np.ndarray | None
    smpl_joints: np.ndarray | None
    body_quat: np.ndarray
    vr_position: np.ndarray | None
    vr_orientation: np.ndarray | None
    compatibility: str
    body_reference: str = 'smpl'
    joint_positions: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None

    @property
    def frames(self):
        return len(self.timestamps)

    def summary(self):
        return {'dataset': str(self.dataset), 'episode': self.index, 'frames': self.frames,
                'fps': FPS, 'duration_s': self.frames / FPS, 'recorded_in': self.source,
                'body_reference': self.body_reference,
                'body_action': ('recorded measured G1 references through SONIC' if self.body_reference == 'joints'
                                else 'recorded SMPL / VR references through SONIC'),
                'hand_action': 'recorded native6 submitted targets', 'compatibility': self.compatibility,
                'source_sha256': self.sha256, 'vr_tracking': self.vr_position is not None}


def load_joint_reference(root, capture, rows, hands):
    """Read the extra native export and align hands to its actual capture clock."""
    reference = capture.get('sonic_reference')
    if not reference:
        raise ValueError('这条录制没有实测关节角回放文件，请选择「SMPL 人体参考」；新录制会自动保存两种参考。')
    if (reference.get('schema') != 'inspire.sonic_reference.v1' or reference.get('fps') != FPS
            or reference.get('source_frames') != rows or reference.get('joint_order') != 'isaaclab'
            or reference.get('encoder_mode') != 'g1' or reference.get('time_source') != 'sample_monotonic_ns'):
        raise ValueError('incompatible measured G1 reference metadata')

    def local_path(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('measured reference path escapes dataset')
        return path

    sources = [local_path(str(Path(reference['path']) / f'{name}.csv'))
               for name in ('joint_pos', 'joint_vel', 'body_quat')]
    q, dq, quat = [matrix(np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2), size, path.name)
                   for path, size in zip(sources, (29, 29, 4))]
    count = reference.get('frames')
    if type(count) is not int or count < 2 or any(len(v) != count for v in (q, dq, quat)):
        raise ValueError('measured G1 reference frame counts disagree')
    timing_path = local_path(capture['timing_trace']['path'])
    with np.load(timing_path, allow_pickle=False) as timing:
        samples = timing['sample_monotonic_ns']
    if (samples.shape != (rows,) or samples.dtype.kind not in 'iu' or np.any(samples < 0)
            or np.any(np.diff(samples.astype(np.int64)) <= 0)):
        raise ValueError('invalid measured reference source timing')
    sample_ns = samples.astype(np.int64) - int(samples[0])
    step_ns = 1_000_000_000 // FPS
    if count != int(sample_ns[-1] // step_ns) + 1:
        raise ValueError('measured reference duration disagrees with source timing')
    target_ns = np.arange(count, dtype=np.int64) * step_ns
    aligned = [matrix(np.column_stack([np.interp(target_ns, sample_ns, col) for col in hand.T]),
                      6, 'aligned hand actions') for hand in hands]
    return q, dq, quat, target_ns / 1e9, aligned, sources + [timing_path]


def load_episode(dataset, episode_index=0, *, config, body_reference='smpl'):
    """Read v4 newbranch captures and v3 captures from the original project."""
    import pyarrow.parquet as pq
    import yaml

    if body_reference not in ('smpl', 'joints'):
        raise ValueError('body reference must be smpl or joints')

    root = Path(dataset).expanduser().resolve()
    if (root / 'INCOMPLETE').exists():
        raise ValueError('dataset is marked INCOMPLETE')
    info = read_json(root / 'meta/info.json')
    if type(episode_index) is not int or not 0 <= episode_index < info.get('total_episodes', 0):
        raise ValueError('episode index is outside the dataset')
    from gear_sonic.utils.data_collection.batch import episode_metadata
    capture_path = episode_metadata(root, episode_index, 'inspire_capture.json')
    capture = read_json(capture_path)
    schema = capture.get('capture_schema')
    if schema not in ('inspire.lerobot.capture.v3', 'inspire.lerobot.capture.v4'):
        raise ValueError('requires a complete G1 + Inspire capture (v3/v4), not a hand-only bench trajectory')
    expected = {'embodiment_tag': 'UNITREE_G1_SONIC', 'body_dof': 29, 'state_dof': 41,
                'hand_dof_per_side': 6, 'joint_order': list(JOINT_ORDER), 'fps': FPS}
    if any(capture.get(key) != value for key, value in expected.items()) or info.get('fps') != FPS:
        raise ValueError('dataset G1/native6 channel contract or 50 Hz timing is incompatible')
    source = capture.get('validation_level')
    if source not in ('SIM_ONLY', 'REAL_CAPTURE_READ_ONLY'):
        raise ValueError('unknown recording source')
    if type(episode_index) is not int or not 0 <= episode_index < info.get('total_episodes', 0):
        raise ValueError('episode index is outside the dataset')
    chunk = info.get('chunks_size', 1000)
    if type(chunk) is not int or chunk < 1:
        raise ValueError('invalid dataset chunk size')
    relative = info.get('data_path', 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet').format(
        episode_chunk=episode_index // chunk, episode_index=episode_index)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('episode path escapes dataset')
    table = pq.read_table(path)
    rows = table.num_rows
    if rows < 2:
        raise ValueError('replay requires at least two frames')

    def column(key, size):
        if key not in table.column_names:
            raise ValueError(f'dataset is missing {key}; body replay requires recorded source references')
        feature = info.get('features', {}).get(key, {})
        if feature.get('shape') != [size]:
            raise ValueError(f'incompatible feature shape: {key}')
        return matrix(table[key].to_pylist(), size, key)

    indices = np.asarray(table['frame_index'].to_pylist())
    episodes = np.asarray(table['episode_index'].to_pylist())
    timestamps = np.asarray(table['timestamp'].to_pylist(), dtype=np.float64)
    if (indices.dtype.kind not in 'iu' or not np.array_equal(indices, np.arange(rows))
            or episodes.dtype.kind not in 'iu' or not np.all(episodes == episode_index)
            or timestamps.shape != (rows,) or not np.all(np.isfinite(timestamps))
            or timestamps[0] < 0 or not np.allclose(np.diff(timestamps), 1 / FPS, rtol=.003, atol=1e-6)):
        raise ValueError('episode frames/timestamps must be contiguous at 50 Hz')
    if info['total_episodes'] == 1 and (info.get('total_frames') != rows or capture.get('actual_frames') != rows):
        raise ValueError('dataset frame counts disagree')
    if (root / 'meta/capture_batch.json').exists() and capture.get('actual_frames') != rows:
        raise ValueError('batch episode frame count disagrees')
    mapping_path = deployment_config(config)
    mapping = load_mapping(mapping_path)
    hands = [column(f'action.{side}_hand_joints', 6) for side in ('left', 'right')]
    for side, values in zip(('left', 'right'), hands):
        names = info['features'][f'action.{side}_hand_joints'].get('names')
        if names != list(JOINT_ORDER):
            raise ValueError(f'{side} hand feature order differs from native6')
        for row in values:
            mapping.validate_limits(row, field_name=f'replay.{side}', side=side)
    compatibility = 'legacy v3 native6 schema; no device-package version was recorded'
    metadata_path = episode_metadata(root, episode_index, 'end_effector.json')
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        registration = metadata['deployment']['registration']
        active = json.loads(json.dumps(config.registration.description.to_dict()))
        for key in ('device_id', 'control_mode', 'action_channels', 'state_channels'):
            if registration.get(key) != active.get(key):
                raise ValueError(f'dataset device configuration differs: {key}')
        saved = metadata.get('configuration_files', {}).get('hand_config', {})
        if not saved.get('text') or hashlib.sha256(saved['text'].encode()).hexdigest() != saved.get('sha256'):
            raise ValueError('dataset has no valid saved hand mapping')
        if mapping_semantics(yaml.safe_load(saved['text'])) != mapping_semantics(yaml.safe_load(mapping_path.read_text())):
            raise ValueError('recorded and active hand action mappings differ')
        compatibility = 'device channels and saved action mapping match; deployment endpoints may differ'
    elif schema.endswith('v4'):
        if capture.get('mapping_sha256') != hashlib.sha256(mapping_path.read_bytes()).hexdigest():
            raise ValueError('v4 mapping hash differs and no saved mapping is available for comparison')
        compatibility = 'v4 exact mapping hash match'
    pose = joints = q = dq = None
    reference_sources = []
    if body_reference == 'joints':
        q, dq, quat, timestamps, hands, reference_sources = load_joint_reference(root, capture, rows, hands)
    else:
        pose = column('teleop.smpl_pose', 63)
        joints = column('teleop.smpl_joints', 72)
        quat = column('teleop.body_quat_w', 4)
    norms = np.linalg.norm(quat, axis=1)
    if np.any(np.abs(norms - 1) > .02):
        raise ValueError('body reference quaternions are not normalized')
    pos = orientation = None
    vr_keys = ('teleop.vr_3pt_position', 'teleop.vr_3pt_orientation')
    if body_reference == 'smpl' and any(key in table.column_names for key in vr_keys):
        from gear_sonic.utils.data_collection.transforms import rot6d_to_quat
        pos, rot = column(vr_keys[0], 9), column(vr_keys[1], 18)
        r = rot.reshape(-1, 6)
        if np.any(np.linalg.norm(np.cross(r[:, :3], r[:, 3:]), axis=1) < .5):
            raise ValueError('invalid recorded VR rotation axes')
        orientation = rot6d_to_quat(r).reshape(rows, 12)
        orientation.setflags(write=False)
    digest = hashlib.sha256()
    sources = [root / 'meta/info.json', capture_path, path, *reference_sources]
    if metadata_path.exists(): sources.append(metadata_path)
    for file in sources:
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    timestamps.setflags(write=False)
    return ReplayEpisode(root, episode_index, source, digest.hexdigest(), timestamps,
                         *hands, pose, joints, quat, pos, orientation, compatibility, body_reference, q, dq)


def reconstruct_wrist_joint_positions(smpl_pose):
    """Original PICO SMPL-to-G1 conversion, as used by the old source replay."""
    from scipy.spatial.transform import Rotation as R
    from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
    pose = matrix(smpl_pose, 63, 'teleop.smpl_pose').reshape(-1, 21, 3)
    elbows = []
    for index in (17, 18):
        # The upstream helper divides by angle; identity rotations need no decomposition.
        nonzero = np.linalg.norm(pose[:, index], axis=1) > 1e-8
        swing = np.tile([1., 0., 0., 0.], (len(pose), 1))
        if np.any(nonzero):
            _, swing[nonzero] = decompose_rotation_aa(pose[nonzero, index], np.array([0, 1, 0]))
        elbows.append(R.from_quat(swing[:, [1, 2, 3, 0]]).as_euler('XYZ'))
    left, right = (R.from_rotvec(pose[:, i]).as_euler('XYZ') for i in (19, 20))
    q = np.zeros((len(pose), 29), dtype=np.float64)
    q[:, 23], q[:, 25], q[:, 27] = elbows[0][:, 0] + left[:, 0], left[:, 1], elbows[0][:, 2] + left[:, 2]
    q[:, 24], q[:, 26], q[:, 28] = -(elbows[1][:, 0] + right[:, 0]), -right[:, 1], elbows[1][:, 2] + right[:, 2]
    return q


def pose_window(episode, frame, wrists=None, *, stream_frame=None, count=None):
    """SMPL uses five frames; G1 covers release's 46-frame and low_latency's 10-frame windows."""
    if count is None:
        count = 50 if episode.body_reference == 'joints' else 5
    if not 0 <= frame < episode.frames or count < 1:
        raise ValueError('pose window is outside the episode')
    indices = np.minimum(np.arange(frame, frame + count), episode.frames - 1)
    start = frame if stream_frame is None else stream_frame
    if episode.body_reference == 'joints':
        return {'joint_pos': episode.joint_positions[indices],
                'joint_vel': episode.joint_velocities[indices],
                'body_quat_w': episode.body_quat[indices],
                'frame_index': np.arange(start, start + count, dtype=np.int64)}
    data = {'smpl_pose': episode.smpl_pose[indices].reshape(count, 21, 3),
            'smpl_joints': episode.smpl_joints[indices].reshape(count, 24, 3),
            'body_quat_w': episode.body_quat[indices], 'joint_pos': wrists[indices],
            'joint_vel': np.zeros((count, 29), dtype=np.float64),
            'frame_index': np.arange(start, start + count, dtype=np.int64),
            'left_hand_joints': np.zeros(7, dtype=np.float32),
            'right_hand_joints': np.zeros(7, dtype=np.float32)}
    if episode.vr_position is not None:
        data.update(vr_position=episode.vr_position[frame], vr_orientation=episode.vr_orientation[frame])
    return data


from gear_sonic.utils.inference.sonic_session import SonicSession as BodyReplay


def tick(deadline, *, maximum_lateness=.15):
    delay = deadline - time.monotonic()
    if delay > 0: time.sleep(delay)
    late = max(0., time.monotonic() - deadline)
    if late > maximum_lateness:
        raise RuntimeError(f'50 Hz replay deadline missed by {late:.3f}s')
    return late


def enter_ready():
    if not select.select([sys.stdin], [], [], 0)[0]: return False
    if sys.stdin.readline() == '': raise RuntimeError('replay terminal input closed')
    return True


def hand_alignment_status(state, requested, submitted, elapsed_s):
    """Keep the requested pose, limited command and measured pose distinguishable."""
    status = {'elapsed_s': elapsed_s, 'tolerance_rad': .12, 'command_tolerance_rad': .005,
              'requested': {}, 'submitted': {}, 'measured': {}, 'max_command_error_rad': 0.}
    worst = None
    for side in ('left', 'right'):
        for key, values in (('requested', requested.values), ('submitted', submitted.values),
                            ('measured', state.values)):
            status[key][side] = list(map(float, values[side]))
        for index, name in enumerate(JOINT_ORDER):
            target = status['requested'][side][index]
            command = status['submitted'][side][index]
            measured = status['measured'][side][index]
            status['max_command_error_rad'] = max(status['max_command_error_rad'], abs(target-command))
            error = abs(target-measured)
            if worst is None or error > worst['error_rad']:
                worst = {'hand': side, 'joint': name, 'target_rad': target,
                         'submitted_rad': command, 'measured_rad': measured, 'error_rad': error}
    status['worst_joint'] = worst
    status['max_target_error_rad'] = worst['error_rad']
    return status


def alignment_failure(status):
    joint = status['worst_joint']
    side = {'left': '左手', 'right': '右手'}[joint['hand']]
    name = dict(zip(JOINT_ORDER, ('小指弯曲', '无名指弯曲', '中指弯曲', '食指弯曲',
                                 '拇指弯曲', '拇指旋转')))[joint['joint']]
    return (f"手部首帧对齐超时：{side}{name}，目标 {joint['target_rad']:.3f}、"
            f"发送 {joint['submitted_rad']:.3f}、实测 {joint['measured_rad']:.3f} rad，"
            f"误差 {joint['error_rad']:.3f} rad（允许 {status['tolerance_rad']:.3f}）。"
            f"命令与首帧目标的最大差值 {status['max_command_error_rad']:.3f} rad。身体回放尚未开始。")


def execute(episode, config, *, frames=0, auto_start=False,
            body_endpoint='tcp://127.0.0.1:5556', state_endpoint='tcp://127.0.0.1:5557'):
    """Two operator gates: start standing, then play after support is released."""
    if config.registration.description.device_id != 'inspire' or config.backend not in ('sim_shared', 'real'):
        raise ValueError('full-body replay requires Inspire sim_shared or real deployment')
    if auto_start and config.backend != 'sim_shared':
        raise ValueError('automatic start is only allowed for local simulation tests')
    count = episode.frames if frames == 0 else frames
    if type(count) is not int or not 1 <= count <= episode.frames:
        raise ValueError('requested replay frames exceed the selected episode')
    wrists = reconstruct_wrist_joint_positions(episode.smpl_pose) if episode.body_reference == 'smpl' else None
    report = episode.summary() | {'target': config.backend, 'deployment': config.snapshot(),
                                  'requested_frames': count, 'played_frames': 0, 'completed': False}
    output = ROOT / 'outputs/replay_runs' / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8] + '.json')
    started = time.monotonic(); body = hand = None
    finished = False
    try:
        with DeviceLease(config, 'dataset_replay'), ExitStack() as resources:
            body = BodyReplay(body_endpoint, state_endpoint)
            resources.callback(body.close)
            hand = config.create()
            resources.callback(hand.close)
            resources.callback(lambda: body.stop() if not finished else None)
            hand.connect()
            def send_pose(frame):
                data = pose_window(episode, frame, wrists)
                if episode.body_reference == 'joints':
                    body.pose(data, version=1)
                else:
                    body.pose(data)
            def hand_state():
                state = hand.get_state()
                report['last_hand_feedback'] = {side: list(map(float, values)) if values is not None else None
                                                for side, values in state.values.items()}
                # Like the original replay/live PICO, trust the gateway's measured
                # state contract, not command limits as hard physics boundaries.
                # MuJoCo's soft limits can yield tiny negative measured angles.
                # Seed/action limits and all driver guards remain enforced below.
                if state.fault:
                    raise EffectorError('backend_fault', state.fault, device='inspire')
                age = (time.monotonic_ns() - state.received_monotonic_ns) / 1e9
                if not 0 <= age <= hand.description.stale_after_s:
                    raise EffectorError('feedback_stale', f'feedback age {age:.3f}s', device='inspire')
                for side in ('left', 'right'):
                    values = state.values.get(side)
                    if not state.valid.get(side) or np.shape(values) != (6,) or not np.all(np.isfinite(values)):
                        raise EffectorError('feedback_invalid', 'missing finite native6 measured state', device='inspire')
                return state
            hand_state()
            print(f'REPLAY_PROGRESS frame=0/{count} time_s=0.00', flush=True)
            print('REPLAY_READY 请保留支撑；按 Enter 启动 SONIC 站立。', flush=True)
            while not auto_start and not enter_ready():
                hand_state(); time.sleep(.02)
            print('REPLAY_STARTING 正在启动站立，等待身体反馈。', flush=True)
            # Acquire the hands before body motion, just as live PICO does.
            # Keep the measured pose while standing; recorded targets start at the second gate.
            hold = hand.seed_action(hand_state())
            hand.send_action(hold)
            while body.feedback(required=False): pass  # Discard output before this Start request.
            body.last_feedback = None
            # Start planner before consuming data, as in the original replay.
            deadline = time.monotonic() + 15
            while True:
                body.idle(); body.command(planner=True); hand_state(); hand.send_action(hold)
                if body.feedback(required=False): break
                if time.monotonic() > deadline: raise TimeoutError('no SONIC CONTROL output after planner Start')
                time.sleep(.02)
            # Continue idle while the operator releases support. No dataset commands yet.
            standing = time.monotonic()
            while time.monotonic() - standing < 3:
                body.idle(); body.feedback(); hand_state(); hand.send_action(hold); time.sleep(.02)
            print('REPLAY_STANDING 身体已进入控制。确认站稳、释放支撑后，按 Enter 开始回放。', flush=True)
            while not auto_start and not enter_ready():
                body.idle(); body.feedback(); hand_state(); hand.send_action(hold); time.sleep(.02)
            first = EffectorAction({'left': tuple(episode.left[0]), 'right': tuple(episode.right[0])})
            previous = hold
            print('REPLAY_ALIGNING 从实际手部反馈平滑对齐首帧。', flush=True)
            alignment = time.monotonic(); deadline = alignment
            while True:
                state = hand_state(); body.feedback(); body.idle()
                previous = hand.limit_action(previous, first, 1 / FPS)
                hand.send_action(previous)
                status = hand_alignment_status(state, first, previous, time.monotonic() - alignment)
                report['hand_alignment'] = status
                if status['elapsed_s'] >= .5 and status['max_target_error_rad'] < .12 and status['max_command_error_rad'] < .005: break
                if status['elapsed_s'] > 10:
                    raise RuntimeError(alignment_failure(status))
                deadline += 1 / FPS; tick(deadline)
            # Preload first pose before selecting streaming, matching old source replay.
            send_pose(0); time.sleep(.02)
            body.command(planner=False); body.pose_selected = True
            print(f'REPLAY_STARTED frames={count} fps={FPS}', flush=True)
            replay_start = time.monotonic(); deadline = replay_start
            maximum_lateness = 0.; maximum_error = 0.; limited_frames = 0
            for frame in range(count):
                hand_state(); body.feedback()
                desired = EffectorAction({'left': tuple(episode.left[frame]), 'right': tuple(episode.right[frame])})
                previous = hand.limit_action(previous, desired, 1 / FPS)
                hand.send_action(previous)
                send_pose(frame)
                if frame < 3: body.command(planner=False)
                frame_error = max(abs(a-b) for side in ('left','right')
                                  for a,b in zip(previous.values[side], desired.values[side]))
                maximum_error = max(maximum_error, frame_error)
                limited_frames += int(frame_error > .005)
                report['played_frames'] = frame + 1
                if frame == 0 or (frame+1) % FPS == 0 or frame+1 == count:
                    print(f'REPLAY_PROGRESS frame={frame+1}/{count} time_s={(frame+1)/FPS:.2f}', flush=True)
                deadline += 1 / FPS
                lateness = tick(deadline)
                maximum_lateness = max(maximum_lateness, lateness)
                if lateness > 1 / FPS: deadline = time.monotonic()  # Never catch up with a command burst.
            report.update(command_hz=count/(time.monotonic()-replay_start), maximum_lateness_s=maximum_lateness,
                          maximum_hand_target_error_rad=float(maximum_error), hand_limited_frames=limited_frames)
            if limited_frames:
                print(f'REPLAY_LIMITED 手部限速/碰撞约束调整了 {limited_frames} 帧；最大目标差 {maximum_error:.4f} rad。', flush=True)
            # Complete replay returns to the original planner/idle path, never cuts motor control.
            for index in range(40):
                body.feedback(); hand_state(); hand.send_action(previous)
                body.idle()
                if index < 3: body.command(planner=True)
                time.sleep(.02)
            body.pose_selected = False
            hand.close()
            finished = True; report['completed'] = True
            print('REPLAY_COMPLETE 回放完成，身体保持站立，手部已停止并释放控制。', flush=True)
    except BaseException as exc:
        report['completed'] = False
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_s'] = time.monotonic() - started
        atomic_json(output, report)
        print(f'REPLAY_REPORT {output}', flush=True)
    return report


def run(args, config):
    root = Path(args.dataset).expanduser().resolve()
    playback = getattr(args, 'playback', 'auto')
    if playback == 'simulation' or (playback == 'auto' and not (root / 'meta/inspire_capture.json').exists()
                                   and (root / 'meta/simulation_replay.json').exists()):
        from .recording.simulation import run as replay_simulation
        return replay_simulation(args, config)
    episode = load_episode(args.dataset, args.episode, config=config,
                           body_reference=getattr(args, 'body_reference', 'smpl'))
    if args.frames < 0 or args.frames > episode.frames:
        raise ValueError('--frames must be 0 (whole episode) or within the episode length')
    if args.auto_start and (not args.execute or args.target != 'sim'):
        raise ValueError('--auto-start is only for executed local simulation tests')
    summary = episode.summary() | {'target': args.target, 'requested_frames': args.frames or episode.frames,
                                   'connections_opened': False}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print('REPLAY_CHECK_OK 数据可用于 G1 + Inspire 姿态参考回放；尚未连接控制。', flush=True)
        return summary
    return execute(episode, config, frames=args.frames, auto_start=args.auto_start,
                   body_endpoint=args.body_endpoint, state_endpoint=args.state_endpoint)
