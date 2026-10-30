"""Portable MuJoCo state recordings, separate from commands sent to a robot."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from gear_sonic.utils.hand_control.management import atomic_json
from gear_sonic.utils.hand_control.registry import ROOT, read_json

SCHEMA = 'inspire.simulation_replay.v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(dataset, scene):
    """Add scene + aligned states to a physics dataset without changing training columns."""
    import mujoco
    root, scene = Path(dataset).resolve(), Path(scene).resolve()
    info = read_json(root / 'meta/info.json')
    provenance = read_json(root / 'meta/provenance.json')
    if provenance.get('capture_schema') != 'inspire.basket_physics.v1' or (root / 'INCOMPLETE').exists():
        raise ValueError('requires a completed physics recording')
    target = root / 'simulation'
    if target.exists() or (root / 'meta/simulation_replay.json').exists():
        raise FileExistsError('simulation replay is already packaged')
    stage = root / ('.simulation-' + uuid.uuid4().hex)
    stage.mkdir()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        mujoco.mj_saveModel(model, str(stage / 'scene.mjb'))
        records = []
        for index, record in enumerate(provenance['episodes']):
            source = Path(record['simulation_run'])
            if digest(source / 'physics.npz') != record['physics_sha256']:
                raise ValueError('original simulation recording changed')
            with np.load(source / 'samples.npz', allow_pickle=False) as samples:
                arrays = {k: samples[k] for k in ('qpos', 'qvel', 'timestamp')}
            check_arrays(arrays, model, record['frames'], info['fps'])
            name = f'episode_{index:06d}.npz'
            np.savez_compressed(stage / name, **arrays)
            records.append({'file': 'simulation/' + name, 'sha256': digest(stage / name),
                            'frames': record['frames'],
                            'light_scale': read_json(source / 'parameters.json')['light_scale']})
        metadata = {'schema': SCHEMA, 'device': 'inspire', 'source': 'sim', 'fps': info['fps'],
                    'mujoco_version': mujoco.__version__, 'scene': 'simulation/scene.mjb',
                    'scene_sha256': digest(stage / 'scene.mjb'), 'source_scene': str(scene),
                    'episodes': records, 'playback': 'recorded states; no controller or hardware commands'}
        stage.rename(target)
        try:
            atomic_json(root / 'meta/simulation_replay.json', metadata)
        except BaseException:
            shutil.rmtree(target)
            raise
        return metadata
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def check_arrays(arrays, model, frames, fps):
    if frames < 2 or fps != 50:
        raise ValueError('requires at least two frames at 50 Hz')
    for key, shape in [('qpos', (frames, model.nq)), ('qvel', (frames, model.nv)), ('timestamp', (frames,))]:
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'invalid simulation {key} shape or values')
    if not np.allclose(arrays['timestamp'], np.arange(frames) / fps, atol=1e-6):
        raise ValueError('simulation timestamps are not contiguous at 50 Hz')


def load(dataset, index):
    import mujoco
    root = Path(dataset).expanduser().resolve()
    meta = read_json(root / 'meta/simulation_replay.json')
    if (root / 'INCOMPLETE').exists() or meta.get('schema') != SCHEMA:
        raise ValueError('invalid or incomplete simulation recording')
    if meta.get('mujoco_version') != mujoco.__version__:
        raise ValueError('recorded scene requires MuJoCo ' + str(meta.get('mujoco_version')))
    if type(index) is not int or not 0 <= index < len(meta['episodes']):
        raise ValueError('episode index is outside the dataset')

    def resource(name, expected):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('simulation resource path or hash differs')
        return path

    model = mujoco.MjModel.from_binary_path(str(resource(meta['scene'], meta['scene_sha256'])))
    record = meta['episodes'][index]
    with np.load(resource(record['file'], record['sha256']), allow_pickle=False) as samples:
        arrays = {k: samples[k] for k in ('qpos', 'qvel', 'timestamp')}
    check_arrays(arrays, model, record['frames'], meta['fps'])
    scale = record['light_scale']
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid recorded lighting')
    model.light_diffuse[:] *= scale
    return model, arrays


def run(args, config):
    import mujoco
    if args.target != 'sim':
        raise ValueError('仿真状态回放只能在仿真查看；不能将 qpos 或反馈 token 当作真机控制命令')
    model, arrays = load(args.dataset, args.episode)
    total = len(arrays['timestamp'])
    count = args.frames or total
    if not 1 <= count <= total:
        raise ValueError('requested frames exceed the recorded episode')
    report = {'dataset': str(Path(args.dataset).resolve()), 'episode': args.episode,
              'recorded_in': 'sim', 'target': 'sim', 'playback': 'simulation_state',
              'frames': total, 'requested_frames': count, 'played_frames': 0,
              'completed': False, 'connections_opened': False}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not args.execute:
        print('REPLAY_CHECK_OK 场景与物理状态已检查，可直接播放。', flush=True)
        return report
    data = mujoco.MjData(model)
    output = ROOT / 'outputs/replay_runs' / (time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8] + '.json')
    try:
        with ExitStack() as stack:
            viewer = None
            if not args.headless:
                import mujoco.viewer
                before = set(threading.enumerate())
                viewer = mujoco.viewer.launch_passive(model, data)
                # Handle.close() only requests exit. Join the threads created by
                # launch_passive before interpreter/GL teardown (MuJoCo 3.12).
                owned_threads = set(threading.enumerate()) - before
                def close_viewer():
                    viewer.close()
                    for thread in owned_threads:
                        thread.join(timeout=10)
                        if thread.is_alive():
                            raise RuntimeError('simulation viewer did not finish closing')
                stack.callback(close_viewer)
                camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'basket_chair_front')
                if camera >= 0:
                    with viewer.lock():
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = camera
            print('REPLAY_STARTED 仿真状态回放；关闭窗口或 Ctrl+C 停止。', flush=True)
            started = time.monotonic()
            for frame in range(count):
                if viewer and not viewer.is_running():
                    break
                data.qpos[:] = arrays['qpos'][frame]
                data.qvel[:] = arrays['qvel'][frame]
                data.time = float(arrays['timestamp'][frame])
                mujoco.mj_forward(model, data)
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, started + (frame + 1) / 50 - time.monotonic()))
                report['played_frames'] = frame + 1
                if frame % 50 == 0 or frame == count - 1:
                    print(f'REPLAY_PROGRESS frame={frame+1}/{count} time_s={(frame+1)/50:.2f}', flush=True)
            report['completed'] = report['played_frames'] == count
            print('REPLAY_COMPLETE 仿真回放结束。' if report['completed'] else 'REPLAY_STOPPED 窗口已关闭。', flush=True)
    finally:
        atomic_json(output, report)
        print(f'REPLAY_REPORT {output}', flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Package existing physics states for simulation replay')
    parser.add_argument('dataset')
    parser.add_argument('--scene', required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.dataset, args.scene), ensure_ascii=False))
