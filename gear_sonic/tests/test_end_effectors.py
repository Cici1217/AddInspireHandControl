"""Registration contracts and actual MuJoCo trajectories; no physical devices."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.hand_control.interface import Channel, EffectorAction, EffectorError, EffectorState, ActionReceipt
from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment, registrations
from gear_sonic.utils.data_collection.end_effector import audit_trajectory


def test_registry_is_stdlib_only_and_has_honest_capabilities(monkeypatch):
    import importlib
    monkeypatch.setattr(importlib, 'import_module', lambda *args: pytest.fail('configuration imported device code'))
    devices = registrations()
    assert set(devices) == {'inspire', 'dex3'}
    assert devices['inspire'].description.action_dim == 6
    assert devices['inspire'].supports('real', 'control')
    assert devices['dex3'].supports('sim', 'body')
    assert not devices['dex3'].supports('sim', 'teleop')
    assert not devices['dex3'].supports('real', 'body')
    with pytest.raises(EffectorError, match='unavailable'):
        deployment('dex3', 'real')


@pytest.mark.parametrize("mutation", ["dimension", "limits", "entry", "rate"])
def test_invalid_registration_rejected_before_connect(monkeypatch, mutation):
    import gear_sonic.utils.hand_control.registry as registry
    original = registry.read_json
    doc = original(DEFAULT_REGISTRY/'inspire/manifest.json')
    if mutation == 'dimension': doc['description']['action_channels'] = []
    if mutation == 'limits': doc['description']['action_channels'][0]['upper'] = -1
    if mutation == 'entry': doc['backends']['sim']['factory'] = 'not a module'
    if mutation == 'rate': doc['description']['command_hz'] = 0
    monkeypatch.setattr(registry, 'read_json', lambda path: doc if path.parent.name == 'inspire' else original(path))
    with pytest.raises(EffectorError, match='configuration'):
        registrations()


def test_actions_own_their_values_and_reject_wrong_dimensions():
    values = {'left':[0]*6, 'right':[0]*6}
    action = EffectorAction(values); values['left'][0] = .7
    assert action.values['left'] == (0,)*6
    with pytest.raises(TypeError): action.values['left'] = (.7,)*6
    device = deployment('inspire').create()
    with pytest.raises(EffectorError, match='expected 6'):
        device.validate_action(EffectorAction({'left':[0,0], 'right':[0]*6}))
    with pytest.raises(EffectorError, match='outside'):
        device.validate_action(EffectorAction({'left':[float('nan')]*6, 'right':[0]*6}))
    with pytest.raises(EffectorError, match='mounts'):
        device.validate_action(EffectorAction({'left':[0]*6}))
    with pytest.raises(EffectorError, match='not connected'): device.get_state()
    device.close(); device.close(); device.reset()


@pytest.mark.parametrize('backend', ['sim', 'sim_shared', 'real'])
def test_inspire_simulation_roundoff_preserves_feedback_and_strict_targets(backend):
    measured = [-8e-5, 0, 0, 0, 0, 0]
    from gear_sonic.utils.hand_control.interface import HandState
    native = SimpleNamespace(read_state=lambda: HandState(
        tuple(measured), tuple(measured), time.monotonic_ns()))
    device = deployment('inspire', backend).create()
    device._backend, device._connected = native, True
    state = device.get_state()
    assert state.values['left'][0] == -8e-5  # Never fabricate an in-range measurement.
    if backend == 'real':
        with pytest.raises(EffectorError, match='feedback_invalid'):
            device.seed_action(state)
    else:
        assert device.seed_action(state).values['left'][0] == 0
        measured[0] = -2e-4
        with pytest.raises(EffectorError, match='feedback_invalid'):
            device.seed_action(device.get_state())
    with pytest.raises(EffectorError, match='action_rejected'):
        device.validate_action(EffectorAction({'left': [-8e-6]*6, 'right': [0]*6}))


def test_fault_is_not_hidden_by_old_feedback_and_inspire_rejection_is_classified():
    config = deployment('inspire')
    stale_fault = EffectorState({'left':[0]*6, 'right':[0]*6}, 0,
                                {'left':False, 'right':False}, 'gateway FAULT')
    with pytest.raises(EffectorError, match='backend_fault.*gateway FAULT'):
        stale_fault.require_fresh(time.monotonic_ns(), config.registration.description)
    device = config.create()
    try:
        device.connect()
        with pytest.raises(EffectorError, match='action_rejected.*first target'):
            device.send_action(EffectorAction({'left':[.4]*6, 'right':[0]*6}))
        # The underlying driver still latches the invalid first command as a fault.
        assert device.get_state().fault
    finally:
        device.close()


@pytest.mark.parametrize('device',['inspire'])
def test_actual_mujoco_control_feedback_and_shared_capture(tmp_path, monkeypatch, device):
    monkeypatch.setenv('MUJOCO_GL','egl')
    from gear_sonic.scripts.run_end_effector import build_parser,run
    out=tmp_path/device
    args=build_parser().parse_args(['--device',device,'--operation','sequence','--seconds','1.2','--record',str(out)])
    result=run(args)
    assert result['submitted']>=40 and result['fault'] is None
    assert 35<result['command_hz']<65
    audit=audit_trajectory(out)
    assert audit['validation']=='simulation'
    with np.load(out/'trajectory.npz') as data:
        mount='left'
        measured=data[f'observation.effectors.{mount}']
        targets=data[f'action.effectors.{mount}']
        assert np.ptp(measured[:,0])>.001
        assert not np.array_equal(measured[:,:targets.shape[1]],targets)
        assert np.all(np.diff(data['submitted_ns'])>0)
    assert audit['action_dim_per_mount']==6
    assert audit['state_dim_per_mount']==6
    assert len(list((out/'images').glob('*.png')))>=3
    # Corrupt a slice: the preprocessing validator must not accept a changed action space.
    modality=json.loads((out/'modality.json').read_text());modality['action'][mount]['end']+=1
    (out/'modality.json').write_text(json.dumps(modality))
    with pytest.raises(ValueError,match='slice'):audit_trajectory(out)


@pytest.mark.parametrize('device', ['inspire'])
def test_optical_mapping_drives_actual_mujoco_and_records(tmp_path, monkeypatch, device):
    """Synthetic OpenXR input, real mapper/worker/MuJoCo/camera/writer for both tools."""
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    from gear_sonic.scripts.run_end_effector import build_parser, run_optical
    from gear_sonic.tests.test_inspire_optical import synthetic_hand, calibration_bundle
    from gear_sonic.utils.teleop.hand_tracking import TrackedHandsFrame
    from gear_sonic.devices.inspire.retargeting import InspireRetargeter

    class Reader:
        stopped = False

        def start(self):
            pass

        def get_latest(self):
            now = time.monotonic()
            stamp = time.monotonic_ns()
            hand = synthetic_hand(closed_channels={'pinky', 'ring', 'middle', 'index'})
            return {'timestamp_ns': stamp, 'timestamp_monotonic': now,
                    'tracked_hands': TrackedHandsFrame(stamp, now, time.time(), hand, hand)}

        def stop(self):
            self.stopped = True

    config = deployment(device)
    mapper = InspireRetargeter(config, calibration=calibration_bundle())
    reader = Reader()
    out = tmp_path / device
    args = build_parser().parse_args(['--device', device, '--operation', 'teleop', '--seconds', '1.2', '--record', str(out)])
    result = run_optical(args, config, reader=reader, retargeter=mapper)
    assert reader.stopped
    assert result['submissions'] >= 40
    assert 35 < result['submission_rate_hz'] < 65
    with np.load(out / 'trajectory.npz') as data:
        mount = config.mounts[0]
        assert np.ptp(data[f'observation.effectors.{mount}'][:, 0]) > .001
        assert np.all(data['tracking_valid'])
        assert np.all(data['source_ns'] > 0)
        assert np.all(np.diff(data['feedback_received_ns']) > 0)
        assert data[f'action.effectors.{mount}'].shape[1] == config.registration.description.action_dim
    assert audit_trajectory(out)['frames'] >= 40


def test_dependency_check_validates_symbol_without_importing():
    from dataclasses import replace
    registration = registrations()['inspire']
    spec = registration.backends | {'sim': registration.backends['sim'] | {'factory':'./adapter.py:MissingAdapter'}}
    with pytest.raises(EffectorError, match='entry point not declared'):
        replace(registration,backends=spec).check_dependencies('sim')


@pytest.mark.parametrize('mounts,options', [('tool', {}), ([], {}), ([['tool']], {}), (['tool'], [])])
def test_malformed_deployment_reports_configuration(mounts, options):
    with pytest.raises(EffectorError, match='configuration'):
        deployment('inspire', mounts=mounts, options=options)


def test_inspire_feedback_time_does_not_refresh_when_physics_is_stopped(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    device = deployment('inspire').create()
    try:
        device.connect()
        device._backend._exit.set()
        device._backend._thread.join(2)
        assert not device._backend._thread.is_alive()
        state = device.get_state()
        time.sleep(.01)
        assert device.get_state().received_monotonic_ns == state.received_monotonic_ns
        with pytest.raises(EffectorError, match='feedback_stale'):
            state.require_fresh(state.received_monotonic_ns + 400_000_000, device.description)
    finally:
        device.close()


def test_inspire_stop_rejects_actions_and_explicit_reset_starts_new_receipts():
    device = deployment('inspire').create()
    try:
        device.connect()
        seed = device.seed_action(device.get_state())
        assert device.send_action(seed).sequence == 1
        device.stop()
        with pytest.raises(EffectorError, match='backend is stopped'):
            device.send_action(seed)
        device.close()
        device.reset()
        device.connect()
        assert device.send_action(device.seed_action(device.get_state())).sequence == 1
    finally:
        device.close()


@pytest.mark.parametrize('corruption', ['png_crc', 'png_payload', 'camera_future', 'stale_feedback', 'invalid_state', 'outcome', 'image_path', 'missing_modality', 'config_file'])
def test_recording_audit_rejects_corrupt_training_sources(tmp_path, corruption):
    from gear_sonic.utils.data_collection.end_effector import TrajectoryWriter
    config = deployment('inspire')
    writer = TrajectoryWriter(tmp_path/'record', config, task='audit test', source='unit')
    action = EffectorAction({'left':[.04]*6, 'right':[.04]*6})
    now = time.monotonic_ns()
    for i in range(2):
        stamp = now+i*20_000_000
        state = EffectorState({'left':[.04]*6, 'right':[.04]*6}, stamp, {'left':True, 'right':True})
        writer.append(state=state, receipt=ActionReceipt(action,stamp,i+1), intent=action,
                      image=np.zeros((4,5,3),dtype=np.uint8), image_timestamp_ns=stamp, sample_ns=stamp)
    output = writer.commit()
    assert audit_trajectory(output)['frames'] == 2
    if corruption.startswith('png_'):
        png = output/'images/000000.png'
        raw = bytearray(png.read_bytes())
        if corruption == 'png_crc':
            raw[40] ^= 1
        else:
            # A valid signature with no image data is not a valid image.
            raw = raw[:8]
        png.write_bytes(raw)
    elif corruption in ('outcome','image_path','config_file'):
        meta = json.loads((output/'metadata.json').read_text())
        if corruption == 'outcome': meta['outcome'] = 'maybe'
        elif corruption == 'config_file': meta['configuration_files']['hand_config']['text'] += 'changed'
        else: meta['images'][0] = '../metadata.json'
        (output/'metadata.json').write_text(json.dumps(meta))
    elif corruption == 'missing_modality':
        meta = json.loads((output/'modality.json').read_text())
        del meta['action']['left']
        (output/'modality.json').write_text(json.dumps(meta))
    else:
        with np.load(output/'trajectory.npz') as data:
            values = {key: data[key].copy() for key in data.files}
        if corruption == 'camera_future': values['camera_timestamp_ns'] += 1
        if corruption == 'stale_feedback': values['feedback_received_ns'] -= 400_000_000
        if corruption == 'invalid_state': values['valid.left'][0] = False
        np.savez_compressed(output/'trajectory.npz', **values)
    with pytest.raises((ValueError, EffectorError)):
        audit_trajectory(output)


@pytest.mark.parametrize('device', ['inspire'])
def test_lerobot_export_preserves_schema_timing_and_is_transactional(tmp_path, monkeypatch, device):
    pytest.importorskip('lerobot', reason='run this test in the data-collection environment')
    import av
    import pyarrow.parquet as pq
    from gear_sonic.data.exporter import Gr00tDataExporter
    from gear_sonic.utils.data_collection.end_effector import TrajectoryWriter, export_lerobot

    config = deployment(device)
    desc = config.registration.description
    action = EffectorAction({m: [(c.lower+c.upper)/2 for c in desc.action_channels] for m in config.mounts})
    values = {m: [(c.lower+c.upper)/2 for c in desc.state_channels] for m in config.mounts}
    writer = TrajectoryWriter(tmp_path/'native', config, task='export regression', source='unit', outcome='success')
    now = time.monotonic_ns()
    for i in range(4):
        stamp = now+i*20_000_000
        state = EffectorState(values, stamp, {m: True for m in config.mounts})
        writer.append(state=state, receipt=ActionReceipt(action,stamp,i+1), intent=action,
                      image=np.full((32,32,3),i*50,dtype=np.uint8), image_timestamp_ns=stamp, sample_ns=stamp)
    source = writer.commit()
    output = export_lerobot(source, tmp_path/'exported')
    table = pq.read_table(next(output.rglob('*.parquet')))
    assert table.num_rows == 4
    np.testing.assert_array_equal(np.asarray(table['diagnostics.timestamp_ns'].to_pylist()).reshape(-1),
                                  [now+i*20_000_000 for i in range(4)])
    for mount in config.mounts:
        for prefix, dim in (('observation', desc.state_dim), ('action', desc.action_dim), ('intent', desc.action_dim)):
            assert np.asarray(table[f'{prefix}.effectors.{mount}'].to_pylist()).reshape(4,-1).shape == (4,dim)
    with av.open(next(output.rglob('*.mp4'))) as video:
        assert len(list(video.decode(video=0))) == 4
    metadata = json.loads((output/'meta/end_effector.json').read_text())
    assert metadata['outcome'] == 'success'
    assert metadata['deployment']['registration']['device_id'] == device
    preprocessing = json.loads((output/'meta/preprocessing.json').read_text())
    assert preprocessing['fields']['action'][config.mounts[0]]['source_shape'] == [desc.action_dim]
    assert preprocessing['end_effector']['action_channels'] == metadata['deployment']['registration']['action_channels']
    with pytest.raises(FileExistsError):
        export_lerobot(source, output)

    def fail_frame(*args, **kwargs):
        raise RuntimeError('injected export failure')

    monkeypatch.setattr(Gr00tDataExporter, 'add_frame', fail_frame)
    with pytest.raises(RuntimeError, match='injected export failure'):
        export_lerobot(source, tmp_path/'failed')
    assert not (tmp_path/'failed').exists()
    assert not list(tmp_path.glob('.failed-*'))
    assert audit_trajectory(source)['frames'] == 4


def test_recording_snapshots_configuration_content_not_only_paths(tmp_path):
    from gear_sonic.utils.data_collection.end_effector import recording_metadata
    source = tmp_path/'model.xml'
    source.write_text('<mujoco/>')
    config = deployment('inspire', options={'hand_config':str(source)})
    metadata = recording_metadata(config, task='snapshot', source='unit')
    source.write_text('<mujoco model="changed"/>')
    assert metadata['configuration_files']['hand_config']['text'] == '<mujoco/>'
    source.unlink()
    with pytest.raises(ValueError, match='missing configuration file'):
        recording_metadata(config, task='snapshot', source='unit')


def test_preprocessing_rejects_equal_dimension_different_units_and_bad_slices():
    import copy
    from gear_sonic.utils.data_collection.end_effector import preprocessing_config, validate_preprocessing, modality_config
    config = deployment('inspire')
    features = {f'{kind}.effectors.{side}':{'dtype':'float32','shape':[6],'names':list('abcdef')}
                for kind in ('observation','action') for side in ('left','right')}
    features['observation.images.ego_view'] = {'dtype':'video','shape':[32,32,3],'names':['height','width','channels']}
    modality = modality_config(config.registration.description, config.mounts)
    expected = preprocessing_config(features, modality, config.snapshot())
    validate_preprocessing(expected, copy.deepcopy(expected))
    changed = copy.deepcopy(expected)
    changed['end_effector']['action_channels'][0]['unit'] = 'meter'
    with pytest.raises(ValueError, match='end_effector'):
        validate_preprocessing(changed, expected)
    modality['action']['left']['end'] = 7
    with pytest.raises(ValueError, match='slice'):
        preprocessing_config(features, modality, config.snapshot())
    with pytest.raises(ValueError, match='unsupported'):
        validate_preprocessing({}, {})
