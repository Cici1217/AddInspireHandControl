"""Package schema, configuration checks and the shared RobotModel factory.

No physics, hardware or kinematics imports occur during registry discovery.
Native model loading is explicit and always local, including for real devices.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .interface import EffectorError

FIELD_TYPES = {'string', 'path', 'number', 'integer', 'boolean', 'select', 'json'}


def validate_package(package):
    from .registry import entrypoint
    if not isinstance(package, dict) or not package.get('version'):
        raise ValueError('package requires a version')
    resources = package.get('resources')
    if not isinstance(resources, dict) or not resources:
        raise ValueError('package requires named resources')
    for name, spec in resources.items():
        if not isinstance(spec, dict) or not isinstance(spec.get('path'), str):
            raise ValueError(f'invalid resource: {name}')
        path = Path(spec['path'])
        if path.is_absolute() or '..' in path.parts or not spec['path']:
            raise ValueError(f'resource must be relative to package/shared root: {name}')
        if type(spec.get('shared', False)) is not bool:
            raise ValueError(f'invalid shared resource flag: {name}')
    model = package.get('model', {})
    if model.get('factory'):
        entrypoint(model['factory'])
    if model.get('resource_factory'):
        entrypoint(model['resource_factory'])
    for name in ('urdf', 'mjcf', 'scene'):
        if model.get(name) not in resources:
            raise ValueError(f'model requires a declared {name} resource')
    groups = model.get('joint_groups', {})
    seen = set()
    for group, rows in groups.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'empty joint group: {group}')
        for row in rows:
            if not all(isinstance(row.get(k), str) and row[k] for k in ('urdf', 'mjcf', 'actuator')):
                raise ValueError(f'incomplete joint/actuator mapping in {group}')
            if row['mjcf'] in seen:
                raise ValueError('joint mapped more than once')
            seen.add(row['mjcf'])
            low, high = row['limits']
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (low, high)) or low >= high:
                raise ValueError('invalid model joint limits')
    for mount, spec in model.get('mounts', {}).items():
        for key, size in (('xyz', 3), ('rpy', 3)):
            value = spec.get(key)
            if not isinstance(value, list) or len(value) != size or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
                raise ValueError(f'{mount}.{key} must contain {size} finite values')
        if not spec.get('parent_frame') or not spec.get('frame'):
            raise ValueError('mount requires parent_frame and frame')
    keys = set()
    for field in package.get('options', []):
        if field['type'] not in FIELD_TYPES or field['key'] in keys:
            raise ValueError('invalid/duplicate configuration field')
        keys.add(field['key'])
        if field['type'] == 'select' and not field.get('options'):
            raise ValueError('select field requires options')


def option_value(options, key, default=None):
    value = options
    for part in key.split('.'):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def field_visible(field, options, backend):
    if field.get('backends') and backend not in field['backends']:
        return False
    return all(option_value(options, key) == value for key, value in field.get('when', {}).items())


def validate_options(package, options, backend):
    for spec in package.get('options', []):
        if not field_visible(spec, options, backend):
            continue
        key, kind = spec['key'], spec['type']
        value = option_value(options, key, spec.get('default'))
        if value is None or value == '':
            if spec.get('required'):
                raise EffectorError('configuration', f'required field: {key}')
            continue
        valid = True
        if kind in ('number', 'integer'):
            valid = type(value) in ((int,) if kind == 'integer' else (int, float)) and math.isfinite(value)
            valid = valid and spec.get('min', -math.inf) <= value <= spec.get('max', math.inf)
        elif kind in ('string', 'path', 'select'):
            valid = isinstance(value, str)
            if kind == 'select':
                valid = valid and value in spec['options']
        elif kind == 'boolean':
            valid = type(value) is bool
        elif kind == 'json':
            try:
                json.dumps(value, allow_nan=False)
            except (ValueError, TypeError):
                valid = False
        if not valid:
            raise EffectorError('configuration', f'invalid field {key}: {value!r}')

def resource_path(deployment, name):
    override = deployment.options.get('resources', {}).get(name)
    if override:
        return deployment.path(override)
    model = deployment.registration.package.get('model', {})
    if model.get('resource_factory'):
        paths = deployment.registration.resolve_entry(model['resource_factory'])(deployment)
        if name in paths:
            return Path(paths[name])
    if name in (model.get('mjcf'), model.get('scene')) and model.get('xml_option'):
        return deployment.path(deployment.options[model['xml_option']])
    return deployment.registration.resource(name)


def create_robot_model(deployment):
    """One factory for assembled G1 models and standalone tool URDFs."""
    spec = deployment.registration.package.get('model', {})
    if not spec:
        raise EffectorError('capability', 'device has no RobotModel declaration')
    try:
        if spec.get('factory'):
            model = deployment.registration.resolve_entry(spec['factory'])(deployment)
        else:
            from gear_sonic.data.robot_model.robot_model import RobotModel
            path = resource_path(deployment, spec['urdf'])
            model = RobotModel(str(path), str(path.parent))
        if spec.get('dof') is not None and model.num_joints != spec['dof']:
            raise ValueError(f'expected {spec["dof"]} joints, got {model.num_joints}')
        for rows in spec.get('joint_groups', {}).values():
            for row in rows:
                index = model.dof_index(row['urdf'])
                actual = [float(model.lower_joint_limits[index]), float(model.upper_joint_limits[index])]
                if any(abs(a-b) > 1e-4 for a, b in zip(actual, row['limits'])):
                    raise ValueError(f'RobotModel limits differ: {row["urdf"]}')
        return model
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise EffectorError('configuration', f'RobotModel: {exc}') from exc


def check_resources(registration):
    """Resolve resources and detect missing/LFS files without importing device code."""
    result = {}
    for name, spec in registration.package.get('resources', {}).items():
        path = registration.resource(name)
        if not path.exists():
            raise EffectorError('configuration', f'missing resource {name}: {path}')
        if path.is_file():
            with path.open('rb') as stream:
                if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    raise EffectorError('configuration', f'LFS pointer instead of resource: {path}')
        result[name] = {'path': str(path), 'shared': spec.get('shared', False)}
    return result


def check_models(deployment, *, load=True):
    """Validate URDF + MuJoCo + mappings. Never connects a real driver."""
    registration = deployment.registration
    resources = check_resources(registration)
    spec = registration.package['model']
    urdf = resource_path(deployment, spec['urdf']) if load else registration.resource(spec['urdf'])
    tree = ET.parse(urdf)
    joints = {row.get('name'): row for row in tree.findall('joint')}
    links = {row.get('name') for row in tree.findall('link')}
    for mesh in tree.findall('.//mesh'):
        path = urdf.parent / mesh.attrib['filename']
        if not path.is_file():
            raise EffectorError('configuration', f'missing URDF mesh: {path}')
        with path.open('rb') as stream:
            if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise EffectorError('configuration', f'LFS pointer instead of mesh: {path}')
    mapping = []
    for group, rows in spec.get('joint_groups', {}).items():
        for row in rows:
            if row['urdf'] not in joints or joints[row['urdf']].get('type') == 'fixed':
                raise EffectorError('configuration', f'missing movable URDF joint: {row["urdf"]}')
            limit = joints[row['urdf']].find('limit')
            actual = [float(limit.get('lower')), float(limit.get('upper'))]
            if any(abs(a-b) > 1e-4 for a, b in zip(actual, row['limits'])):
                raise EffectorError('configuration', f'URDF limit mismatch: {row["urdf"]}')
            mapping.append({'group': group, **row})
    for mount in spec.get('mounts', {}).values():
        if mount['frame'] not in links or mount['parent_frame'] not in links:
            raise EffectorError('configuration', 'mount frame absent from URDF')
        if mount['frame'] != mount['parent_frame']:
            joint = next((j for j in joints.values() if j.find('child').get('link') == mount['frame']
                          and j.find('parent').get('link') == mount['parent_frame']), None)
            if joint is None:
                raise EffectorError('configuration', 'mount does not match URDF parent/child frames')
            origin = joint.find('origin')
            for key in ('xyz', 'rpy'):
                actual = [float(v) for v in (origin.get(key, '0 0 0') if origin is not None else '0 0 0').split()]
                if len(actual) != 3 or any(abs(a-b) > 1e-5 for a,b in zip(actual, mount[key])):
                    raise EffectorError('configuration', f'mount {key} differs from URDF')
    report = {'resources': resources, 'mapping': mapping, 'robot_model': None, 'mujoco': None,
              'policy_compatibility': registration.integration.get('policy_compatibility', 'not validated')}
    if load:
        import mujoco
        model = create_robot_model(deployment)
        mj = mujoco.MjModel.from_xml_path(str(resource_path(deployment, spec['scene'])))
        for row in mapping:
            jid = mj.joint(row['mjcf']).id
            aid = mj.actuator(row['actuator']).id
            if int(mj.actuator_trnid[aid, 0]) != jid:
                raise EffectorError('configuration', f'actuator joint mismatch: {row["actuator"]}')
            if any(abs(float(a)-b) > 1e-4 for a, b in zip(mj.jnt_range[jid], row['limits'])):
                raise EffectorError('configuration', f'MuJoCo limit mismatch: {row["mjcf"]}')
        for group, names in spec.get('physical_groups', {}).items():
            if len(names) != len(registration.description.physical_joints) or len(set(names)) != len(names):
                raise EffectorError('configuration', f'physical joint count differs from description: {group}')
            for name in names:
                mj.joint(name)  # Includes passive joints omitted by a simplified FK URDF.
        report['robot_model'] = {'dof': model.num_joints, 'joint_names': list(model.joint_to_dof_index)}
        report['mujoco'] = {'nq': mj.nq, 'nv': mj.nv, 'actuators': mj.nu,
                            'physics_hz': 1 / mj.opt.timestep,
                            'command_hz': registration.description.command_hz}
    return report


def package_fingerprint(registration):
    """Owned source + assets, excluding generated files and external symlink targets."""
    hashes = {}
    if not registration.package:
        return {'manifest': registration.sha256}
    base = registration.source.parent
    for path in sorted(base.rglob('*')):
        relative = path.relative_to(base)
        if any(p.startswith('.') or p in ('__pycache__', 'build') for p in relative.parts):
            continue
        if path.is_symlink():
            hashes[str(relative)] = 'symlink:' + str(path.readlink())
        elif path.is_file():
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': hashes}
