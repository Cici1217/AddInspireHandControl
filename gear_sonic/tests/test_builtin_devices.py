"""Built-in hand selection, model consistency and import boundaries."""
import time

import pytest

from gear_sonic.utils.hand_control.interface import EffectorAction, EffectorError
from gear_sonic.utils.hand_control.management import DeviceLease, active_leases
from gear_sonic.utils.hand_control.models import check_models
from gear_sonic.utils.hand_control.registry import ROOT, deployment, registrations


@pytest.mark.parametrize('name,dof', [('inspire', 41), ('dex3', 43)])
def test_real_models_and_declared_joint_actuator_consistency(name, dof):
    report = check_models(deployment(name))
    assert report['robot_model']['dof'] == dof
    assert report['mujoco']['actuators'] == dof
    assert report['mapping']


def test_inspire_local_simulation_never_imports_hardware_driver():
    import subprocess
    import sys
    code = '''
import sys
from gear_sonic.utils.hand_control.registry import deployment
device = deployment('inspire').simulation()
try:
    device.connect()
    device.send_action(device.seed_action(device.get_state()))
    assert device.get_state().valid['left']
    assert not any(name.endswith('.driver.client') or name.endswith('.driver.session') for name in sys.modules)
finally:
    device.close()
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_inspire_factory_drives_actual_physics():
    config = deployment('inspire')
    assert config.robot_model().num_joints == 41
    device = config.simulation(); device.connect()
    try:
        start = device.get_state(); previous = device.seed_action(start)
        desired = EffectorAction({mount: tuple(c.lower + .65 * (c.upper-c.lower) for c in device.description.action_channels) for mount in device.mounts})
        deadline = time.monotonic() + .8
        while time.monotonic() < deadline:
            previous = device.limit_action(previous, desired, .02)
            device.send_action(previous); time.sleep(.02)
        state = device.get_state()
        state.require_fresh(time.monotonic_ns(), device.description)
        assert state.received_monotonic_ns > start.received_monotonic_ns
        assert any(abs(a-b) > .005 for mount in device.mounts for a,b in zip(start.values[mount], state.values[mount]))
    finally:
        device.close()


def test_builtin_factory_has_one_python_identity():
    import sys
    from gear_sonic.devices.inspire.adapter import InspireAdapter
    from gear_sonic.devices.inspire.retargeting import InspireRetargeter
    config = deployment('inspire')
    assert type(config.create()) is InspireAdapter
    assert config.registration.resolve_entry(config.registration.retargeting['factory']) is InspireRetargeter
    assert not any(name.startswith('_effector_') for name in sys.modules)


def test_external_registry_is_rejected_without_loading_code(tmp_path):
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(EffectorError, match='custom device registries'):
        registrations(tmp_path)
    assert set(registrations()) == {'inspire', 'dex3'}


def test_runtime_lease_tracks_and_releases_process():
    config = deployment('inspire')
    with DeviceLease(config, 'builtin-test') as lease:
        assert any(row['id'] == lease.row['id'] for row in active_leases(device='inspire'))
        saved = lease.path
    assert not saved.exists()


@pytest.mark.parametrize('option,value', [('waist_location','wrong'), ('high_elbow_pose',1), ('operation_timeout_s',float('nan'))])
def test_invalid_builtin_options_fail_before_connect(option,value):
    with pytest.raises(EffectorError,match='invalid field'):
        deployment('inspire',options={option:value})
