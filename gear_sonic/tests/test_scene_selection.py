"""Custom scenes and declared hand actions use the existing simulator/driver paths."""
import json
from pathlib import Path

import pytest

from gear_sonic.utils.console.catalog import build_plan
from gear_sonic.utils.console.config import ROOT, Profiles, defaults, effector_catalog
from gear_sonic.utils.hand_control.models import resource_path
from gear_sonic.utils.hand_control.registry import deployment, deployment_for_mode
from gear_sonic.utils.mujoco_sim.scenes import check_scene, list_scenes, resolve_scene, standalone_scene


@pytest.mark.parametrize('device,name', [('inspire', 'table_box'), ('dex3', 'table_box')])
def test_scene_compiles_preserves_robot_and_adds_physical_object(device, name):
    import mujoco
    config = deployment_for_mode(device, 'sim')
    reference = resource_path(config, config.registration.package['model']['scene'])
    path = check_scene(resolve_scene(ROOT, device, name), reference)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    # Added free-joint object actually falls onto a surface; no fake model backend.
    object_joint = model.joint('work_box_free')
    z = object_joint.qposadr[0] + 2
    start = data.qpos[z]
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[z] - start) > 1e-6
    assert model.nu == config.registration.package['model']['dof']
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    launch = SimLoopConfig(hand=device, enable_onscreen=False)
    wbc, _ = config.body_integration().simulation(launch.load_wbc_yaml(), launch)
    wbc['ROBOT_SCENE'] = str(path)
    # The actual SONIC environment must still resolve body/hand indices.
    # Construct only the physics environment: no DDS, sockets or robot connection.
    env = DefaultEnv(wbc)
    assert len(env.body_joint_index) == 29
    assert len(env.left_hand_index) == (6 if device == 'inspire' else 7)
    assert env.mj_model.camera('head_camera').id >= 0

def test_scene_rejects_wrong_device_missing_file_and_path_escape(tmp_path):
    reference = resource_path(deployment('dex3'), 'scene')
    with pytest.raises(ValueError, match='不匹配|执行器'):
        check_scene(ROOT / 'scenes/inspire/table_box/scene.xml', reference)
    for name in ('../inspire/table_box', '/tmp/outside', ''):
        with pytest.raises(ValueError):
            resolve_scene(ROOT, 'dex3', name)
    with pytest.raises(ValueError, match='不存在'):
        resolve_scene(ROOT, 'dex3', 'missing')
    directory = tmp_path / 'scenes/dex3'
    directory.mkdir(parents=True)
    (directory / 'external').symlink_to(ROOT / 'scenes/dex3/table_box', target_is_directory=True)
    assert list_scenes(tmp_path, 'dex3') == []


def test_scene_sensor_binding_is_checked_by_name_after_site_renumbering(tmp_path):
    xml = '''<mujoco><worldbody>{world}<body name="robot">
      <geom size=".1"/><site name="imu"/><site name="other"/>
      </body></worldbody><sensor><gyro name="gyro" site="{sensor}"/></sensor></mujoco>'''
    reference, custom = tmp_path/'reference.xml', tmp_path/'scene.xml'
    reference.write_text(xml.format(world='', sensor='imu'))
    custom.write_text(xml.format(world='<site name="marker"/>', sensor='imu'))
    assert check_scene(custom, reference) == custom
    # Wrong object can have either the same numeric ID or a different one.
    for world, sensor in [('<site name="marker"/>', 'marker'), ('', 'other')]:
        custom.write_text(xml.format(world=world, sensor=sensor))
        with pytest.raises(ValueError, match='绑定改变'):
            check_scene(custom, reference)


@pytest.mark.parametrize('feature', ['body', 'hands', 'scene', 'teleop', 'capture', 'replay'])
def test_scene_choice_reaches_shared_sim_only(feature):
    c = defaults() | {'task.sim_scenes': {'inspire': 'table_box'}}
    plan = build_plan(c, feature, 'sim')
    scene = next(s for s in plan['steps'] if s['id'] == 'scene')
    assert scene['argv'][scene['argv'].index('--scene') + 1] == str(ROOT / 'scenes/inspire/table_box/scene.xml')
    assert all('--scene' not in s['argv'] for s in plan['steps'] if s['id'] != 'scene')
    if feature != 'scene':
        assert all('--scene' not in s['argv'] for s in build_plan(c, feature, 'real')['steps'])


@pytest.mark.parametrize('mode', ['sim', 'real'])
def test_hand_scripts_are_declared_and_original_sequence_is_reused(mode):
    for name in ('open', 'close', 'sequence'):
        c = defaults() | {'task.pose': 'sequence', 'task.hand_action': name}
        step = build_plan(c, 'hands', mode)['steps'][-1]
        if name == 'sequence':
            assert 'gear_sonic.scripts.inspire_hand_example' in step['argv']
            assert '--sequence' in step['argv']
        else:
            assert 'gear_sonic.scripts.run_end_effector' in step['argv']
            targets = json.loads(step['argv'][step['argv'].index('--targets') + 1])
            assert targets['left'] == targets['right']
            if name == 'close':
                assert targets['left'] == [1.7, 1.7, 1.7, 1.7, .6, .3]
            else:
                assert targets['left'] == [0] * 6
    with pytest.raises(ValueError, match='测试动作'):
        build_plan(c | {'task.hand_action': 'unknown'}, 'hands', mode)


def test_scene_and_hand_settings_persist_legacy_read_only_choice_migrates(tmp_path):
    c = defaults() | {'task.pose': 'feedback', 'task.sim_scenes': {'inspire': 'table_box'}, 'task.hand_action': 'close'}
    profiles = Profiles(tmp_path)
    profiles.save('test', c)
    saved = Profiles(tmp_path).snapshot()['profiles']['test']
    assert saved['task.pose'] == 'targets'
    assert saved['task.hand_action'] == 'close'
    assert saved['task.sim_scenes'] == {'inspire': 'table_box'}
    row = next(d for d in effector_catalog(saved) if d['device_id'] == 'inspire')
    assert 'table_box' in [scene['name'] for scene in row['scenes']]


@pytest.mark.parametrize('scene_root', ['scenes', str(ROOT / 'scenes')])
def test_common_scene_directory_resolves_relative_to_project(scene_root):
    c = defaults() | {'local.scene_root': scene_root, 'task.sim_scenes': {'inspire': 'basket_chair'}}
    rows = effector_catalog(c)
    row = next(d for d in rows if d['device_id'] == 'inspire')
    assert row['scene_directory'] == str(ROOT / 'scenes/inspire')
    plan = build_plan(c, 'scene', 'sim')
    argv = plan['steps'][0]['argv']
    assert argv[argv.index('--scene') + 1] == str(ROOT / 'scenes/inspire/basket_chair/scene.xml')


def test_restored_basket_scene_keeps_dimensions_and_works_in_shared_environment():
    import mujoco
    import numpy as np
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    config = deployment_for_mode('inspire', 'sim')
    path = check_scene(ROOT / 'scenes/inspire/basket_chair/scene.xml', resource_path(config, 'scene'))
    launch = SimLoopConfig(hand='inspire', enable_onscreen=False)
    wbc, _ = config.body_integration().simulation(launch.load_wbc_yaml(), launch)
    wbc['ROBOT_SCENE'] = str(path)
    env = DefaultEnv(wbc)
    model, data = env.mj_model, env.mj_data
    mujoco.mj_forward(model, data)
    assert len(env.body_joint_index) == 29 and len(env.left_hand_index) == 6
    assert model.body('manipulatable_box').mass[0] == pytest.approx(.65)
    assert model.body('photo_chair').dofnum[0] == 0
    geom = model.geom('chair_seat').id
    mesh = model.geom_dataid[geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    points = model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    np.testing.assert_allclose(np.ptp(points, axis=0)[:2], [.45, .45], atol=1e-6)
    assert points[:, 2].max() == pytest.approx(.43, abs=1e-6)
    # Retain original contact dynamics while holding the unpowered robot for this check.
    joint = model.joint('box_joint'); qa, da = int(joint.qposadr[0]), int(joint.dofadr[0])
    initial = data.qpos.copy()
    for _ in range(240):
        data.qpos[:qa] = initial[:qa]; data.qvel[:da] = 0
        mujoco.mj_step(model, data)
    assert not any(w.number for w in data.warning)
    assert data.qpos[qa+2] == pytest.approx(.43, abs=.001)
    np.testing.assert_allclose(data.qpos[qa:qa+2], initial[qa:qa+2], atol=.002)
