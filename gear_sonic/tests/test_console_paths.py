"""Relocatable profiles and command plans, without opening robot connections."""

import json
from pathlib import Path

import pytest

from gear_sonic.utils.console import config
from gear_sonic.utils.console.catalog import build_plan
from gear_sonic.utils.console.directories import list_directories


@pytest.mark.parametrize('feature,modes', [
    ('body', ('sim', 'real')), ('hands', ('sim', 'real')),
    ('teleop', ('sim', 'real')), ('capture', ('sim', 'real')),
    ('replay', ('sim', 'real')), ('vla', ('sim', 'real')), ('setup', ('local',)),
])
def test_absolute_and_relative_profiles_build_identical_commands(feature, modes, tmp_path, monkeypatch):
    relative = config.defaults() | {
        'local.ready_pose': 'gear_sonic/data/poses/g1/box_ready.txt',
        'task.sim_scenes': {'inspire': 'table_box'},
    }
    absolute = config.resolve_paths(relative)
    monkeypatch.chdir(tmp_path)
    for mode in modes:
        old = build_plan(absolute, feature, mode)
        new = build_plan(relative, feature, mode)
        assert old['steps'] == new['steps']
        assert old['log_root'] == new['log_root']
        assert old['output_directories'] == new['output_directories']
        for step in new['steps']:
            assert Path(step['cwd']).is_absolute()


def test_profile_moves_with_checkout_and_preserves_external_tools(tmp_path, monkeypatch):
    first = tmp_path / 'first checkout'
    second = tmp_path / 'another checkout'
    monkeypatch.setattr(config, 'ROOT', first)
    old = config.resolve_paths(config.defaults()) | {
        'local.groot_root': '/external/Isaac-GR00T',
        'local.model': str(first / 'outputs/checkpoints/finetuned/my model'),
        'local.ready_pose': str(first / 'gear_sonic/data/poses/g1/box_ready.txt'),
        'task.prompt': 'keep my task',
    }
    settings = first / 'configs/console'
    settings.mkdir(parents=True)
    (settings / 'profiles.json').write_text(json.dumps({'active': 'saved', 'profiles': {'saved': old}}))
    migrated = config.Profiles(settings).snapshot()['profiles']['saved']
    assert migrated['local.root'] == '.'
    assert migrated['local.model'] == 'outputs/checkpoints/finetuned/my model'
    assert migrated['local.ready_pose'] == 'gear_sonic/data/poses/g1/box_ready.txt'
    assert migrated['local.groot_root'] == '/external/Isaac-GR00T'
    assert str(first) not in (settings / 'profiles.json').read_text()
    monkeypatch.setattr(config, 'ROOT', second)
    monkeypatch.chdir(tmp_path)
    loaded = config.validate(migrated, check_devices=False)
    assert loaded['local.model'] == str(second / 'outputs/checkpoints/finetuned/my model')
    assert loaded['local.sim_python'] == str(second / '.venv_sim/bin/python')
    assert loaded['local.groot_root'] == '/external/Isaac-GR00T'
    assert loaded['task.prompt'] == 'keep my task'
    assert loaded['g1.body_script'] == old['g1.body_script']


def test_default_store_imports_old_settings_once_without_changing_source(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    old = home / '.config/gear-sonic-newbranch-console/profiles.json'
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({'active': 'saved', 'profiles': {'saved': config.defaults() | {'task.name': 'my_recording'}}}))
    original = old.read_bytes()
    local = tmp_path / 'project/configs/console'
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    monkeypatch.setattr(config, 'STATE_DIRECTORY', local)
    first = config.Profiles()
    assert first.path == local / 'profiles.json'
    assert first.snapshot()['profiles']['saved']['task.name'] == 'my_recording'
    first.save('saved', config.defaults() | {'task.name': 'new_recording'})
    assert config.Profiles().snapshot()['profiles']['saved']['task.name'] == 'new_recording'
    assert old.read_bytes() == original


def test_directory_picker_and_venv_paths_are_relative_to_project_not_cwd(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    model = root / 'outputs/checkpoints/finetuned/my model'
    model.mkdir(parents=True)
    python = root / '.venv_sim/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to('/usr/bin/python3')
    monkeypatch.chdir(tmp_path)
    result = list_directories('outputs/checkpoints/finetuned', project_root=root)
    assert result['directories'] == [{'name': 'my model', 'path': str(model)}]
    c = config.validate(config.defaults() | {'local.root': str(root)}, check_devices=False)
    # Dereferencing this symlink would silently switch from the venv to system Python.
    assert c['local.sim_python'] == str(python)


def test_onboarding_reports_missing_model_files_without_running_programs(tmp_path, monkeypatch):
    from gear_sonic.utils.console import bootstrap
    monkeypatch.setattr(bootstrap.subprocess, 'run', lambda *a, **k: pytest.fail('onboarding ran a program'))
    root = tmp_path / 'new project'
    policy = root / 'policy/model'
    policy.parent.mkdir(parents=True)
    Path(str(policy) + '_encoder.onnx').write_bytes(b'encoder')
    checkpoint = root / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    c = config.defaults() | {'local.root': str(root), 'local.sonic_checkpoint': 'policy/model', 'local.model': 'checkpoint'}
    report = bootstrap.inspect_configuration(c)
    rows = {row['key']: row for row in report['checks']}
    assert not report['remote_checked']
    assert rows['local.sonic_checkpoint']['status'] == 'missing'
    assert 'model_decoder.onnx' in rows['local.sonic_checkpoint']['detail']
    assert rows['local.model']['status'] == 'missing'
    assert rows['local.calibration']['status'] == 'optional'
    assert rows['local.sim_python']['path'] == '.venv_sim/bin/python'
    assert not (root / 'outputs').exists()
    Path(str(policy) + '_decoder.onnx').write_bytes(b'decoder')
    (checkpoint / 'model.safetensors').write_bytes(b'weights')
    rows = {row['key']: row for row in bootstrap.inspect_configuration(c)['checks']}
    assert rows['local.sonic_checkpoint']['status'] == 'found'
    assert rows['local.model']['status'] == 'found'


def test_g1_initialization_checks_configured_remote_paths_without_control():
    c = config.defaults() | {'setup.role': 'g1', 'g1.body_script': '/lab/my-g1/run_body.sh',
                             'g1.camera_root': '/lab/camera', 'g1.hand_root': '/lab/hand'}
    plan = build_plan(c, 'setup', 'local')
    step = next(s for s in plan['steps'] if s['id'] == 'check')
    code = step['argv'][step['argv'].index('-c') + 1]
    compile(code, '<g1-read-only-check>', 'exec')
    for path in ('/lab/my-g1/run_body.sh', '/lab/camera/gear_sonic/camera/composed_camera.py', '/lab/hand/bridge.env'):
        assert path in code
    assert 'subprocess' not in code
    assert 'run_body.sh' not in step['argv']
