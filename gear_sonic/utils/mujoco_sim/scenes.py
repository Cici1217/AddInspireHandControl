"""User-owned MJCF scenes; device models remain in their registered packages."""
from dataclasses import replace
from pathlib import Path


def scene_directory(root, scene_root='scenes'):
    return (Path(root) / Path(scene_root).expanduser()).resolve()


def resolve_scene(root, device, name, scene_root='scenes'):
    directory = (scene_directory(root, scene_root) / device).resolve()
    relative = Path(name)
    path = (directory / relative / 'scene.xml').resolve()
    if not name or relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(directory):
        raise ValueError('场景必须位于 scenes/设备名/场景名/scene.xml')
    if not path.is_file():
        raise ValueError(f'场景文件不存在：{path}')
    return path


def list_scenes(root, device, scene_root='scenes'):
    directory = scene_directory(root, scene_root) / device
    result = []
    for file in sorted(directory.glob('*/scene.xml')):
        try:
            path = resolve_scene(root, device, file.parent.name, scene_root)
        except ValueError:
            continue
        result.append({'name': file.parent.name, 'path': str(path)})
    return result


def check_scene(path, reference):
    """Offline regression check for bundled scenes; not part of scene startup.

    Objects may append passive/free joints. Added actuators or changed robot
    transmissions require a device integration, not a scene-only override.
    """
    import mujoco
    import numpy as np
    path = Path(path).expanduser().resolve()
    expected = mujoco.MjModel.from_xml_path(str(reference))
    actual = mujoco.MjModel.from_xml_path(str(path))
    for kind, count in [('joint', expected.njnt), ('actuator', expected.nu),
                        ('sensor', expected.nsensor), ('body', expected.nbody)]:
        for i in range(count):
            try:
                same = getattr(expected, kind)(i).name == getattr(actual, kind)(i).name
            except IndexError:
                same = False
            if not same:
                raise ValueError(f'场景与设备不匹配：{kind}[{i}] 名称或顺序改变')
    # Cameras are selected by name; scene overview cameras may precede robot cameras.
    for i in range(expected.ncam):
        name = expected.camera(i).name
        if name and mujoco.mj_name2id(actual, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            raise ValueError(f'场景缺少设备相机：{name}')
    if actual.nu != expected.nu:
        raise ValueError('场景不能增减机器人执行器')
    for field in ('jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_bodyid', 'jnt_range', 'jnt_limited',
                  'actuator_trntype', 'actuator_trnid', 'actuator_gear', 'actuator_ctrlrange',
                  'actuator_ctrllimited', 'actuator_gainprm', 'actuator_biasprm',
                  'sensor_type', 'sensor_objtype', 'sensor_dim', 'sensor_adr'):
        baseline = getattr(expected, field)
        value = getattr(actual, field)[:len(baseline)]
        if value.shape != baseline.shape or not np.allclose(value, baseline):
            raise ValueError(f'场景与设备不匹配：{field} 改变')
    # Added world sites may renumber IMU sites while leaving the sensor binding intact.
    for i in range(expected.nsensor):
        kind = int(expected.sensor_objtype[i])
        before, after = int(expected.sensor_objid[i]), int(actual.sensor_objid[i])
        name = mujoco.mj_id2name(expected, kind, before) if before >= 0 else None
        target = mujoco.mj_id2name(actual, kind, after) if after >= 0 else None
        if name != target or (not name and before != after):
            raise ValueError(f'场景与设备不匹配：传感器 {expected.sensor(i).name} 绑定改变')
    return path


def standalone_scene(config, path):
    """Override a registered standalone simulator's XML, never a hardware driver."""
    backend = config.registration.backends[config.backend]
    key = config.registration.package.get('model', {}).get('xml_option')
    if backend['mode'] != 'sim' or backend.get('capture_source') != 'local' or not key:
        raise ValueError('--scene 仅用于独立仿真；整机场景请在 run_sim_loop 中选择')
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'场景文件不存在：{path}')
    return replace(config, options={**config.options, key: str(path)})
