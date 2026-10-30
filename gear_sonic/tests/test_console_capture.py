"""Capture window, output ownership and actual dataset/depth publication."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts.capture_session import next_output, reserve_prefix, build_parser, run
from gear_sonic.tests.test_depth_capture import depth_frame
from gear_sonic.tests.test_inspire_recording import FakeRobotModel, build, make_sources
from gear_sonic.devices.inspire.recording.collector import InspireLeRobotCollector
from gear_sonic.devices.inspire.config import load_mapping


class Clock:
    value = 1_000_000_000

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += int(seconds * 1_000_000_000)


def collector(clock, camera_missing=False):
    samples = lambda: make_sources(clock.now())

    def camera():
        if camera_missing or clock.now() < 1_200_000_000:
            return None
        result = samples()[1]
        result["timestamps"]["ego_view"] = clock.now() / 1e9
        return result

    frames = []
    return InspireLeRobotCollector(
        camera=SimpleNamespace(read=camera),
        body_state=SimpleNamespace(get_msg=lambda: samples()[0]),
        hand_state=SimpleNamespace(
            receive_timed=lambda: SimpleNamespace(state=samples()[2], received_monotonic_ns=clock.now())
        ),
        teleop_target=SimpleNamespace(receive_timed=lambda: samples()[3]),
        writer=SimpleNamespace(append=frames.append),
        robot_model=FakeRobotModel(),
        mapping=load_mapping(),
        clock_ns=clock.now,
        sleep=clock.sleep,
    ), frames


def test_wall_clock_capture_does_not_extend_for_missing_samples():
    clock = Clock()
    value, frames = collector(clock)
    stats = value.collect_for(2)
    assert stats.written_frames == len(frames) == 90
    assert stats.drops["missing_camera"] == 10
    assert stats.as_dict()["duration_seconds"] == 2
    assert stats.stop_reason == "duration_limit"


def test_missing_stream_refuses_successful_dataset():
    value, frames = collector(Clock(), camera_missing=True)
    with pytest.raises(RuntimeError, match="only 0 valid frames"):
        value.collect_for(1)
    assert not frames


def test_batch_prefix_lock_and_existing_output_are_preserved(tmp_path):
    assert next_output(tmp_path, "capture").name == "capture"
    (tmp_path / "capture").mkdir()
    (tmp_path / "capture_000").mkdir()
    (tmp_path / "capture/keep.txt").write_text("recorded data")
    assert next_output(tmp_path, "capture").name == "capture_001"
    (tmp_path / "capture_001").mkdir()
    assert next_output(tmp_path, "capture").name == "capture_002"
    with reserve_prefix(tmp_path, "capture"):
        with pytest.raises(RuntimeError, match="owns"):
            with reserve_prefix(tmp_path, "capture"):
                pytest.fail("must not acquire same prefix twice")
    with reserve_prefix(tmp_path, "capture"):
        assert (tmp_path / "capture_000").is_dir()
        assert (tmp_path / "capture/keep.txt").read_text() == "recorded data"


@pytest.mark.parametrize('sim', [True, False])
def test_batch_defaults_split_sources_and_explicit_output_stays_exact(sim, tmp_path, capsys, monkeypatch):
    from gear_sonic.utils.data_collection.paths import PROJECT_ROOT
    monkeypatch.chdir(tmp_path)  # Defaults must not depend on the shell directory.
    base = ['--task-prompt', 'path validation', '--dry-run'] + (['--sim'] if sim else [])
    args = build_parser().parse_args(base)
    run(args)
    result = json.loads(capsys.readouterr().out)
    mode = 'sim' if sim else 'real'
    assert result['output_pattern'] == str(PROJECT_ROOT / f'outputs/datasets/{mode}/inspire_capture')
    assert result['connections_opened'] is False
    explicit = tmp_path / 'legacy location'
    run(build_parser().parse_args(base + ['--output-root', str(explicit)]))
    assert json.loads(capsys.readouterr().out)['next_output'] == str(explicit / 'inspire_capture')
    assert not explicit.exists()


@pytest.fixture
def mock_batch_append(monkeypatch):
    """Isolate the batch coordinator from encoding; real append is checked below."""
    from pathlib import Path
    import shutil
    from gear_sonic.utils.data_collection import batch
    def append(source, output):
        source, output = Path(source), Path(output)
        manifest = output / 'meta/capture_batch.json'
        document = json.loads(manifest.read_text()) if manifest.exists() else {'schema': batch.SCHEMA, 'episodes': []}
        index = len(document['episodes'])
        saved = output / f'meta/recordings/episode_{index:06d}'
        saved.mkdir(parents=True)
        shutil.copy2(source / 'meta/end_effector.json', saved / 'end_effector.json')
        metadata = json.loads((saved / 'end_effector.json').read_text())
        if (source / 'raw').exists():
            shutil.copytree(source / 'raw', output / f'raw/episode_{index:06d}')
        document['episodes'].append({'episode_index': index, 'outcome': metadata['outcome']})
        manifest.write_text(json.dumps(document))
        return index
    monkeypatch.setattr(batch, 'append_episode', append)


@pytest.mark.parametrize("skip_first", [False, True])
def test_batch_labels_each_saved_episode_before_recording_next(tmp_path, monkeypatch, mock_batch_append, capsys, skip_first):
    from gear_sonic.scripts import record_pico_lerobot
    recorded = []
    confirmed = int(skip_first)
    answers = iter(([('', 0)] if not skip_first else []) + [('success' if i % 2 == 0 else 'failure', i + 1) if j == 0 else ('', i + 1) for i in range(10) for j in range(1 if i == 9 else 2)])
    output = tmp_path / 'inspire_capture'

    def record(args):
        assert args.outcome == 'unlabeled'
        assert confirmed == len(recorded) + 1
        if recorded:
            assert json.loads((output / 'meta/recordings/episode_000000/end_effector.json').read_text())['outcome'] == 'success'
        from pathlib import Path
        pending = Path(args.output)
        for relative in ('meta/end_effector.json', 'raw/metadata.json'):
            path = pending / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'outcome': args.outcome, 'task': 'pick cup', 'frames': 50}))
        recorded.append(pending)

    def choose():
        nonlocal confirmed
        answer, count = next(answers)
        assert len(recorded) == count
        if answer:
            assert (output / f'meta/recordings/episode_{count-1:06d}/end_effector.json').exists()
        else:
            confirmed += 1
        return answer

    monkeypatch.setattr(record_pico_lerobot, 'run', record)
    monkeypatch.setattr('builtins.input', choose)
    args = build_parser().parse_args(['--sim', '--task-prompt', 'pick cup', '--output-root', str(tmp_path),
                                     '--episodes', '10', '--seconds', '1',
                                     '--outcome', 'failure', '--label-after-each'] + (['--start-immediately'] if skip_first else []))
    run(args)
    assert len(recorded) == confirmed == 10
    assert capsys.readouterr().out.count('CAPTURE_READY ') == 10 - int(skip_first)
    for index, outcome in enumerate(('success', 'failure') * 5):
        for relative in (f'meta/recordings/episode_{index:06d}/end_effector.json', f'raw/episode_{index:06d}/metadata.json'):
            assert json.loads((output / relative).read_text()) == {'outcome': outcome, 'task': 'pick cup', 'frames': 50}
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == ['inspire_capture']


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, EOFError])
def test_interrupted_label_keeps_saved_episode_unlabeled(tmp_path, monkeypatch, interruption, mock_batch_append):
    from gear_sonic.scripts import record_pico_lerobot
    def record(args):
        from pathlib import Path
        meta = Path(args.output) / 'meta/end_effector.json'
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({'outcome': args.outcome}))
    def interrupt():
        raise interruption
    monkeypatch.setattr(record_pico_lerobot, 'run', record)
    monkeypatch.setattr('builtins.input', interrupt)
    args = build_parser().parse_args(['--sim', '--task-prompt', 'pick cup', '--output-root', str(tmp_path),
                                     '--episodes', '2', '--start-immediately', '--label-after-each'])
    with pytest.raises(interruption):
        run(args)
    assert json.loads((tmp_path / 'inspire_capture/meta/recordings/episode_000000/end_effector.json').read_text())['outcome'] == 'unlabeled'
    assert not (tmp_path / 'inspire_capture_001').exists()


def test_real_lerobot_export_with_depth_sidecar(tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    import cv2
    from gear_sonic.data.robot_model.instantiation.g1_inspire import instantiate_g1_rh56dfx_robot_model
    from gear_sonic.devices.inspire.recording.audit import audit_lerobot_dataset
    from gear_sonic.devices.inspire.recording.exporter import InspireLeRobotExporter
    from gear_sonic.utils.hand_control.registry import deployment

    output = tmp_path / "rgbd_dataset"
    model = instantiate_g1_rh56dfx_robot_model()
    depth = depth_frame()
    with InspireLeRobotExporter(
        output, robot_model=model, fps=50, task_prompt="synthetic RGB-D export test",
        deployment=deployment('inspire','sim_shared'),
        outcome='success',
    ) as writer:
        for index in range(40):
            frame = replace(
                build(now_ns=1_000_000_000 + index * 20_000_000), teleop_target_frame_index=index, depth=depth
            )
            writer.append(frame)
        writer.commit(
            {
                "source": "synthetic test frames; no robot",
                "written_frames": 40,
                "requested_frames": 50,
                "duration_seconds": 1,
                "drops": {"missing_camera": 10},
                "stop_reason": "duration_limit",
                "hand_sequence_repeats": 39,
                "camera_unique_frames": 1,
                "teleop_target_unique_frames": 40,
            }
        )
    report = audit_lerobot_dataset(output, robot_model=model)
    assert report.frames == report.video_frames == 40
    manifest = json.loads((output / "meta/depth/episode_000000.json").read_text())
    assert len(manifest["frames"]) == 40
    assert manifest["unique_depth_frames"] == 1
    np.testing.assert_array_equal(
        cv2.imread(str(output / manifest["frames"][0]["depth_file"]), cv2.IMREAD_UNCHANGED), depth.image
    )
    assert (output / "meta/inspire_capture.json").is_file()
    capture = json.loads((output / "meta/inspire_capture.json").read_text())
    reference = capture['sonic_reference']
    assert reference['frames'] == 40 and reference['fps'] == 50
    assert reference['root_position_measured'] is False
    reference_files = {path.name: path.read_bytes() for path in (output / reference['path']).iterdir()}
    assert set(reference_files) == {'joint_pos.csv', 'joint_vel.csv', 'body_quat.csv',
                                    'body_pos.csv', 'metadata.txt', 'info.txt'}
    generic = json.loads((output/'meta/end_effector.json').read_text())
    assert generic['outcome'] == 'success' and generic['control_writes'] is False
    assert generic['deployment']['registration']['action_dim'] == 6
    assert generic['frames'] == 40 and generic['retargeting_version']
    preprocessing = json.loads((output/'meta/preprocessing.json').read_text())
    assert preprocessing['fields']['action']['left_hand_joints']['source_shape'] == [6]
    assert preprocessing['fields']['action']['body_quat_w']['rotation_type'] == 'quaternion'
    assert generic['configuration_files']['hand_config']['present']
    from gear_sonic.devices.inspire.replay import load_episode
    for target in ('sim_shared', 'real'):
        episode = load_episode(output, config=deployment('inspire', target))
        assert episode.frames == 40 and episode.source == 'SIM_ONLY'
        np.testing.assert_allclose(episode.left, .2)
    from gear_sonic.utils.data_collection.batch import append_episode, label_outcome, episode_metadata
    import pyarrow.parquet as pq
    batch = tmp_path / 'batch'
    for index in range(3):
        assert append_episode(output, batch) == index
        label_outcome(batch, index, ('success', 'failure', 'unlabeled')[index])
    assert audit_lerobot_dataset(batch, robot_model=model).episodes == 3
    assert json.loads((batch / 'meta/info.json').read_text())['total_frames'] == 120
    for index in range(3):
        label = json.loads(episode_metadata(batch, index, 'end_effector.json').read_text())['outcome']
        capture = json.loads(episode_metadata(batch, index, 'inspire_capture.json').read_text())
        reference_path = capture['sonic_reference']['path']
        assert reference_path == f'sonic_reference/episode_{index:06d}'
        assert {path.name: path.read_bytes() for path in (batch / reference_path).iterdir()} == reference_files
        assert label == ('success', 'failure', 'unlabeled')[index]
        table = pq.read_table(batch / f'data/chunk-000/episode_{index:06d}.parquet')
        assert table['index'].to_pylist() == list(range(index * 40, (index + 1) * 40))
        for target in ('sim_shared', 'real'):
            episode = load_episode(batch, index, config=deployment('inspire', target))
            assert episode.frames == 40
            np.testing.assert_allclose(episode.left, .2)
            measured = load_episode(batch, index, config=deployment('inspire', target), body_reference='joints')
            assert measured.frames == 40 and measured.body_reference == 'joints'
            np.testing.assert_allclose(measured.left, episode.left)
            np.testing.assert_allclose(measured.right, episode.right)
        manifest = json.loads((batch / f'meta/depth/episode_{index:06d}.json').read_text())
        assert manifest['episode_index'] == index
        assert all((batch / frame['depth_file']).is_file() for frame in manifest['frames'])
        assert manifest['frames'][0]['dataset_frame_index'] == index * 40
    # A failed publication must restore the already committed three episodes.
    from pathlib import Path
    original_rename = Path.rename
    def fail_publication(path, target):
        if path.name == 'dataset' and Path(target) == batch:
            raise OSError('publication failure')
        return original_rename(path, target)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'rename', fail_publication)
        with pytest.raises(OSError, match='publication failure'):
            append_episode(output, batch)
    assert audit_lerobot_dataset(batch, robot_model=model).frames == 120
    assert len(list((batch / 'sonic_reference').iterdir())) == 3
    assert json.loads((output / 'meta/info.json').read_text())['total_episodes'] == 1
    capture_path = output / "meta/inspire_capture.json"
    capture = json.loads(capture_path.read_text())
    capture["summary"]["stop_reason"] = "frame_limit"
    capture_path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="frame counts"):
        audit_lerobot_dataset(output, robot_model=model)
