"""Append completed, single-episode recordings to one LeRobot v2 batch.

Video and observations are copied without resampling. Only dataset-wide indices
are rewritten. A staged directory is published after all metadata is ready.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from gear_sonic.utils.hand_control.management import atomic_json

SCHEMA = 'gear_sonic.capture_batch.v1'


def read(path):
    return json.loads(Path(path).read_text())


def episode_metadata(root, index, name):
    root = Path(root)
    if not (root / 'meta/capture_batch.json').exists():
        return root / 'meta' / name
    batch = read(root / 'meta/capture_batch.json')
    if batch.get('schema') != SCHEMA or not 0 <= index < len(batch['episodes']):
        raise ValueError('invalid capture batch episode')
    return root / 'meta/recordings' / f'episode_{index:06d}' / name


def label_outcome(root, index, outcome):
    root = Path(root)
    if outcome not in ('success', 'failure', 'unlabeled'):
        raise ValueError('invalid episode outcome')
    batch_path = root / 'meta/capture_batch.json'
    paths = [episode_metadata(root, index, 'end_effector.json')]
    raw = root / 'raw' / f'episode_{index:06d}' / 'metadata.json' if batch_path.exists() else root / 'raw/metadata.json'
    if raw.exists():
        paths.insert(0, raw)
    records = [(path, read(path)) for path in paths]
    for path, document in records:
        document['outcome'] = outcome
        atomic_json(path, document)
    if batch_path.exists():
        batch = read(batch_path)
        batch['episodes'][index]['outcome'] = outcome
        atomic_json(batch_path, batch)


def _copy_snapshot(source, destination):
    """Hard-link immutable large files; metadata must remain independently writable."""
    source, destination = Path(source), Path(destination)
    def copy(src, dst):
        relative = Path(src).relative_to(source)
        if relative.parts[0] in ('data', 'videos', 'depth'):
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)
    shutil.copytree(source, destination, copy_function=copy)


def append_episode(source, output):
    """Publish one complete episode, preserving every previously committed episode."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gear_sonic.data.exporter import Gr00tDatasetMetadata, compute_episode_stats

    source, output = Path(source).resolve(), Path(output).resolve()
    info = read(source / 'meta/info.json')
    if (source / 'INCOMPLETE').exists() or info.get('total_episodes') != 1 or info.get('codebase_version') != 'v2.1':
        raise ValueError('batch append requires a complete single-episode LeRobot v2.1 recording')
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('source and batch must be separate directories')
    existing = output.exists()
    index = offset = 0
    if existing:
        batch = read(output / 'meta/capture_batch.json')
        previous = read(output / 'meta/info.json')
        if batch.get('schema') != SCHEMA:
            raise ValueError('cannot append to an unrelated dataset')
        index, offset = previous['total_episodes'], previous['total_frames']
        if len(batch['episodes']) != index:
            raise ValueError('batch episode count disagrees')
        for key in ('fps', 'features', 'robot_type', 'chunks_size', 'data_path', 'video_path'):
            if info.get(key) != previous.get(key):
                raise ValueError(f'batch recording contract changed: {key}')
        for name in ('modality.json', 'preprocessing.json', 'tasks.jsonl'):
            if (source / 'meta' / name).read_bytes() != (output / 'meta' / name).read_bytes():
                raise ValueError(f'batch metadata changed: {name}')
        old = read(episode_metadata(output, 0, 'end_effector.json'))
        new = read(source / 'meta/end_effector.json')
        for key in ('deployment', 'configuration_files', 'source', 'backend'):
            if old.get(key) != new.get(key):
                raise ValueError(f'batch device configuration changed: {key}')
        capture = source / 'meta/inspire_capture.json'
        old_capture = output / 'meta/inspire_capture.json'
        if capture.exists() != old_capture.exists():
            raise ValueError('batch capture type changed')
        if capture.exists() and read(capture)['validation_level'] != read(old_capture)['validation_level']:
            raise ValueError('batch recording source changed')
    else:
        batch = {'schema': SCHEMA, 'episodes': []}

    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{output.name}.append-', dir=output.parent))
    stage, backup = work / 'dataset', work / 'previous'
    try:
        _copy_snapshot(output if existing else source, stage)
        if existing:
            meta = Gr00tDatasetMetadata(repo_id='local/capture', root=stage)
            path = source / info['data_path'].format(episode_chunk=0, episode_index=0)
            table = pq.read_table(path)
            frames = table.num_rows
            if frames != info['total_frames']:
                raise ValueError('episode frame count disagrees')
            for key, values in {'episode_index': np.full(frames, index), 'index': np.arange(offset, offset + frames)}.items():
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(values, type=table.schema.field(key).type))
            target = stage / meta.get_data_file_path(index)
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            for key in meta.video_keys:
                src = source / info['video_path'].format(episode_chunk=0, episode_index=0, video_key=key)
                target = stage / meta.get_video_file_path(index, key)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
            features = {key: value for key, value in meta.features.items() if value['dtype'] != 'video'}
            arrays = {key: np.asarray(table[key].to_pylist()) for key in features}
            stats = compute_episode_stats(arrays, features)
            tasks = [json.loads(line)['task'] for line in (source / 'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
            meta.save_episode(index, frames, tasks, stats)

        saved = stage / 'meta/recordings' / f'episode_{index:06d}'
        saved.mkdir(parents=True, exist_ok=True)
        device = read(source / 'meta/end_effector.json')
        capture_path = source / 'meta/inspire_capture.json'
        if capture_path.exists():
            capture = read(capture_path)
            timing = source / capture['timing_trace']['path']
            shutil.copy2(timing, saved / 'timing.npz')
            trace_path = (saved / 'timing.npz').relative_to(stage).as_posix()
            capture['timing_trace']['path'] = trace_path
            device['timestamp_trace'] = capture['timing_trace']
            if 'sonic_reference' in capture:
                reference = capture['sonic_reference']
                new_path = f'sonic_reference/episode_{index:06d}'
                if existing:
                    shutil.copytree(source / reference['path'], stage / new_path)
                reference['path'] = new_path
            if 'depth' in capture:
                depth = read(source / capture['depth']['manifest'])
                old_dir = 'depth/episode_000000'
                new_dir = f'depth/episode_{index:06d}'
                if index:
                    shutil.copytree(source / old_dir, stage / new_dir)
                depth['episode_index'] = index
                for row in depth['frames']:
                    row['depth_file'] = row['depth_file'].replace(old_dir + '/', new_dir + '/')
                    row['dataset_frame_index'] += offset
                manifest = f'meta/depth/episode_{index:06d}.json'
                atomic_json(stage / manifest, depth)
                capture['depth'].update(episode_index=index, manifest=manifest)
            atomic_json(saved / 'inspire_capture.json', capture)
        atomic_json(saved / 'end_effector.json', device)
        if (source / 'raw').is_dir():
            if not existing:
                (stage / 'raw').rename(work / 'raw')
                (stage / 'raw').mkdir()
                (work / 'raw').rename(stage / 'raw/episode_000000')
            else:
                shutil.copytree(source / 'raw', stage / 'raw' / f'episode_{index:06d}')
        batch['episodes'].append({'episode_index': index, 'frames': info['total_frames'],
                                  'outcome': device.get('outcome', 'unlabeled')})
        atomic_json(stage / 'meta/capture_batch.json', batch)
        aggregate = read(stage / 'meta/end_effector.json')
        aggregate.update(frames=offset + info['total_frames'], outcome='unlabeled',
                         episode_metadata='meta/recordings/episode_{episode_index:06d}/end_effector.json')
        atomic_json(stage / 'meta/end_effector.json', aggregate)
        if capture_path.exists():
            aggregate = read(stage / 'meta/inspire_capture.json')
            aggregate.update(actual_frames=offset + info['total_frames'],
                             episode_metadata='meta/recordings/episode_{episode_index:06d}/inspire_capture.json')
            atomic_json(stage / 'meta/inspire_capture.json', aggregate)
        # Existing batches remain valid throughout capture and metadata preparation.
        # On a failed publication restore the previous directory before propagating.
        try:
            if existing:
                output.rename(backup)
            stage.rename(output)
        except BaseException:
            if backup.exists() and not output.exists():
                backup.rename(output)
            raise
        return index
    finally:
        # If restoring a failed rename itself fails, keep the last good dataset.
        if not backup.exists() or output.exists():
            shutil.rmtree(work)
