"""Deployment relocation, old entry points, and standalone import boundaries."""
import subprocess
import sys

from gear_sonic.utils.hand_control.registry import ROOT, deployment


def test_native_resource_and_compatibility_paths_resolve_to_one_source():
    registration = deployment('inspire').registration
    native = ROOT / 'gear_sonic_deploy/inspire/native'
    assert registration.resource('native_driver').resolve() == native
    for path in ('gear_sonic/devices/inspire/driver/native', 'gear_sonic_deploy/src/inspire_hand_bridge'):
        assert (ROOT / path).is_symlink()
        assert (ROOT / path).resolve() == native
    assert (native / 'CMakeLists.txt').is_file()


def test_example_compatibility_is_the_same_implementation():
    from gear_sonic.scripts import inspire_hand_example as old
    from gear_sonic.devices.inspire.examples import hand_control as new
    assert old is new
    for module in ('gear_sonic.scripts.inspire_hand_example',
                   'gear_sonic.devices.inspire.examples.hand_control'):
        result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert '--sequence' in result.stdout and '--left' in result.stdout


def test_real_hand_imports_without_body_perception_or_simulation_dependencies():
    code = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mujoco', 'pinocchio', 'cv2', 'lerobot', 'unitree_sdk2py', 'xrobotoolkit_sdk'}:
            raise AssertionError('standalone hand imported optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from gear_sonic.devices.inspire.driver.client import InspireGatewayBackend
from gear_sonic.devices.inspire.driver.supervisor import preflight
from gear_sonic.devices.inspire.examples.hand_control import main
from gear_sonic.utils.hand_control.registry import deployment
config = deployment('inspire', 'real')
device = config.create()
try:
    assert device.description.action_dim == 6
    preflight('real', device.config_path, '/unused', remote=True)
finally:
    device.close()
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_hand_only_rejects_body_command_before_launch(tmp_path):
    marker = tmp_path / 'unexpected-body'
    result = subprocess.run(
        [sys.executable, '-m', 'gear_sonic.devices.inspire.driver.supervisor',
         '--mode', 'sim', '--hand-only', '--', sys.executable, '-c',
         f'from pathlib import Path; Path({str(marker)!r}).touch()'],
        cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'cannot be combined' in result.stderr
    assert not marker.exists()
