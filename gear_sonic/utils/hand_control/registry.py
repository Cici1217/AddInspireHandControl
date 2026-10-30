"""Built-in Inspire and upstream Dex3 configurations; no plugin installation.

Descriptions load without physics or hardware. Runtime code uses ordinary
Python imports, so each driver has one module identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ast
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re

from .interface import Channel, DeviceDescription, EffectorError, EndEffector

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "gear_sonic/data/end_effectors"
DEFAULT_REGISTRY = ROOT / "gear_sonic/devices"
BUILTIN_DEVICES = ("inspire", "dex3")


def registry_directory(directory=DEFAULT_REGISTRY):
    path = Path(directory).expanduser().resolve()
    if path not in (DEFAULT_REGISTRY.resolve(), (DATA / 'devices').resolve()):
        raise EffectorError('configuration', 'Only built-in Inspire and Dex3 are supported; custom device registries have been removed')
    return DEFAULT_REGISTRY


def entrypoint(value):
    if isinstance(value, str) and re.fullmatch(r"\./(?:[A-Za-z_]\w*/)*[A-Za-z_]\w*\.py:[A-Za-z_]\w*", value):
        return value.split(":")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", value):
        raise ValueError(f"invalid module entry point: {value!r}")
    return value.split(":")


def resolve_entry(value, registration=None):
    module, symbol = entrypoint(value)
    try:
        if module.startswith('./'):
            if registration is None:
                raise ValueError('package entry point requires a registration')
            device = registration.description.device_id
            if device not in BUILTIN_DEVICES:
                raise ValueError(f'unsupported device: {device}')
            module = f'gear_sonic.devices.{device}.' + module[2:-3].replace('/', '.')
        factory = getattr(importlib.import_module(module), symbol)
    except (ImportError, AttributeError) as exc:
        raise EffectorError("dependency", f"cannot load {value}: {exc}") from exc
    if not callable(factory):
        raise EffectorError("configuration", f"entry point is not callable: {value}")
    return factory


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        raise EffectorError("configuration", f"{path}: {exc}") from exc


@dataclass(frozen=True)
class Registration:
    description: DeviceDescription
    backends: dict
    retargeting: dict | None
    integration: dict
    source: Path
    sha256: str
    package: dict = field(default_factory=dict)
    registry_dir: Path = DEFAULT_REGISTRY

    def resource(self, name):
        try:
            spec = self.package['resources'][name]
            base = ROOT if spec.get('shared', False) else self.source.parent
            path = Path(spec['path'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('resource paths must stay within their declared root')
            return base / path
        except (KeyError, TypeError, ValueError) as exc:
            raise EffectorError('configuration', f'invalid resource {name}: {exc}') from exc

    def resolve_entry(self, value):
        return resolve_entry(value, self)

    def supports(self, mode, capability):
        return any(b["mode"] == mode and capability in b["capabilities"] for b in self.backends.values())

    def public(self):
        return self.description.to_dict() | {
            "backends": self.backends, "retargeting": self.retargeting,
            "integration": self.integration, "registration_sha256": self.sha256,
            "package": self.package,
        }

    def check_dependencies(self, backend, *, modules=True):
        if backend not in self.backends:
            raise EffectorError("configuration", f"unsupported backend {backend}", device=self.description.device_id)
        spec = self.backends[backend]
        missing = [name for name in spec.get("requires", []) if modules and importlib.util.find_spec(name) is None]
        # Inspect local entry point source without importing (and executing) device code.
        model = self.package.get('model', {})
        for point in [spec.get("factory"), spec.get('recorder'), (self.retargeting or {}).get("factory"), self.integration.get("factory"), self.integration.get('dataset_replay'), self.integration.get('inference'), self.integration.get('policy_check'), model.get('factory'), model.get('resource_factory')]:
            if point:
                module, symbol = entrypoint(point)
                source = (self.source.parent / module) if module.startswith('./') else ROOT / (module.replace(".", "/") + ".py")
                if not source.is_file():
                    missing.append(module)
                    continue
                try:
                    tree = ast.parse(source.read_text(), filename=str(source))
                except (OSError, SyntaxError) as exc:
                    raise EffectorError("configuration", f"cannot inspect {point}: {exc}") from exc
                exports = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
                # Local entry points must be declared directly, not dynamically re-exported.
                if symbol not in exports:
                    raise EffectorError("configuration", f"entry point not declared: {point}", device=self.description.device_id)
        if missing:
            raise EffectorError("dependency", "missing: " + ", ".join(missing), device=self.description.device_id)
        return spec


@dataclass(frozen=True)
class Deployment:
    registration: Registration
    backend: str
    mounts: tuple[str, ...]
    options: dict
    source: str = "inline"

    def path(self, value):
        """Only resource: names use package scope; legacy option paths keep repo scope."""
        return self.registration.resource(value[9:]) if str(value).startswith('resource:') else asset_path(value)

    def robot_model(self):
        from .models import create_robot_model
        return create_robot_model(self)

    def simulation(self):
        if self.registration.backends[self.backend]['mode'] != 'sim':
            raise EffectorError('capability', 'simulation requires a sim backend')
        return self.create()

    def snapshot(self):
        return {"schema": "end_effector.deployment.v1", "device": self.registration.description.device_id,
                "backend": self.backend, "mounts": list(self.mounts), "options": self.options,
                "registration": self.registration.public(), "source": self.source}

    def create(self) -> EndEffector:
        from .management import managed_device
        backend = self.registration.check_dependencies(self.backend)
        if not backend.get("factory"):
            raise EffectorError("capability", "this registration exposes only existing SONIC integration", device=self.registration.description.device_id)
        try:
            instance = self.registration.resolve_entry(backend["factory"])(self)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise EffectorError("configuration", f"invalid adapter options: {exc}", device=self.registration.description.device_id) from exc
        if not isinstance(instance, EndEffector):
            raise EffectorError("configuration", "adapter must implement EndEffector", device=self.registration.description.device_id)
        return managed_device(instance, self)

    def retargeter(self):
        spec = self.registration.retargeting
        if not spec:
            raise EffectorError("capability", "no retargeter registered", device=self.registration.description.device_id)
        return self.registration.resolve_entry(spec["factory"])(self)

    def body_integration(self):
        point = self.registration.integration.get("factory")
        if not point:
            raise EffectorError("capability", "no body integration registered", device=self.registration.description.device_id)
        return self.registration.resolve_entry(point)(self)


def registrations(directory=DEFAULT_REGISTRY):
    result = {}
    directory = registry_directory(directory)
    files = [directory / name / 'manifest.json' for name in BUILTIN_DEVICES]
    for path in files:
        row = read_json(path)
        try:
            if row.get("schema") not in ("end_effector.device.v1", "end_effector.device.v2"):
                raise ValueError("unsupported registration schema")
            spec = dict(row["description"])
            spec["physical_joints"] = tuple(spec["physical_joints"])
            for key in ("state_channels", "action_channels"):
                spec[key] = tuple(Channel(**c) for c in spec[key])
            description = DeviceDescription(**spec)
            if description.device_id != path.parent.name:
                raise ValueError('device id differs from its built-in directory')
            if not re.fullmatch(r"[a-z][a-z0-9_]*", description.device_id):
                raise ValueError("device_id must be a simple identifier")
            if description.device_id in result:
                raise ValueError("duplicate device_id")
            backends = row["backends"]
            if not backends:
                raise ValueError("no backend declared")
            for name, backend in backends.items():
                if backend["mode"] not in ("sim", "real"):
                    raise ValueError("backend mode must be sim or real")
                if not isinstance(backend["capabilities"], list) or any(c not in ("control", "teleop", "capture", "body", "replay", "dataset_replay", "scene", "inference") for c in backend["capabilities"]):
                    raise ValueError("invalid capabilities")
                if 'capture' in backend['capabilities'] and backend.get('capture_source') not in ('local', 'streams'):
                    raise ValueError("capture requires a local or streams capture_source")
                if backend.get('capture_source') == 'streams':
                    entrypoint(backend.get('recorder'))
                if backend.get("factory"):
                    entrypoint(backend["factory"])
                elif any(c in backend["capabilities"] for c in ("control", "teleop", "capture", "inference")):
                    raise ValueError("control, teleop, capture and inference require an adapter")
                for dependency in backend.get("requires", []):
                    if not re.fullmatch(r"[A-Za-z_]\w*", dependency):
                        raise ValueError("dependencies must be top-level module names")
            retarget = row.get("retargeting")
            if retarget:
                entrypoint(retarget["factory"])
                if not retarget.get("version"):
                    raise ValueError("retargeter requires a version")
            if any("teleop" in b["capabilities"] for b in backends.values()) and not retarget:
                raise ValueError("teleop requires retargeting")
            integration = row.get("integration", {})
            if not isinstance(integration, dict):
                raise ValueError("integration must be an object")
            if any('inference' in b['capabilities'] for b in backends.values()):
                entrypoint(integration.get('inference'))
                entrypoint(integration.get('policy_check'))
            actions = integration.get('hand_actions', {})
            if not isinstance(actions, dict):
                raise ValueError('hand_actions must be an object')
            for action, preset in actions.items():
                if not re.fullmatch(r'[a-z][a-z0-9_]*', action) or not isinstance(preset, dict) or not isinstance(preset.get('label'), str):
                    raise ValueError('invalid hand action declaration')
                if 'target' in preset:
                    description.validate_vector(preset['target'])
                elif preset.get('operation') != 'sequence':
                    raise ValueError('hand action requires target or sequence operation')
            files = integration.get('configuration_files', {})
            if not isinstance(files, dict) or any(
                not isinstance(key, str) or not isinstance(spec, dict)
                or type(spec.get('required', True)) is not bool
                or ('default' in spec and not isinstance(spec['default'], str))
                for key, spec in files.items()
            ):
                raise ValueError('invalid configuration_files declarations')
            if integration.get("factory"):
                entrypoint(integration["factory"])
            package = row.get('package', {})
            if package:
                from .models import validate_package
                validate_package(package)
            result[description.device_id] = Registration(description, backends, retarget, row.get("integration", {}), path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest(), package, directory.resolve())
        except (KeyError, TypeError, ValueError) as exc:
            raise EffectorError("configuration", f"{path}: {exc}") from exc
    return result


def deployment(device="inspire", backend="sim", *, mounts=None, options=None, config_path=None, registry_dir=DEFAULT_REGISTRY):
    all_devices = registrations(registry_dir)
    source = "inline"
    if options is not None and not isinstance(options, dict):
        raise EffectorError("configuration", "options must be a JSON object", device=device)
    if config_path:
        source = str(Path(config_path).resolve())
        row = read_json(config_path)
        if row.get("schema") != "end_effector.deployment.v1":
            raise EffectorError("configuration", "unsupported deployment schema")
        if not isinstance(row.get("device"), str) or not isinstance(row.get("backend"), str):
            raise EffectorError("configuration", "deployment requires device and backend strings")
        if not isinstance(row.get("options", {}), dict):
            raise EffectorError("configuration", "deployment options must be a JSON object")
        device, backend = row["device"], row["backend"]
        if mounts is None:
            mounts = row.get("mounts")
        options = row.get("options", {}) | (options or {})
    if device not in all_devices:
        raise EffectorError("configuration", f"unregistered device: {device}")
    registration = all_devices[device]
    if backend not in registration.backends:
        raise EffectorError("capability", f"backend {backend} unavailable", device=device)
    # Mount defaults belong to deployment examples, not the device's joint definition.
    defaults = registration.package.get('defaults', {})
    options = defaults.get('options', {}) | (options or {})
    if mounts is None and defaults:
        mounts = defaults.get('mounts')
    if mounts is None:
        path = DATA / "deployments" / f"{device}.json"
        if not path.is_file():
            raise EffectorError("configuration", "supply mounts or a deployment file", device=device)
        example = read_json(path)
        mounts = example["mounts"]
        options = example.get("options", {}) | (options or {})
    if not isinstance(mounts, (list, tuple)) or not mounts or any(not isinstance(m, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", m) for m in mounts) or len(set(mounts)) != len(mounts):
        raise EffectorError("configuration", "mounts must be nonempty unique identifiers", device=device)
    if not isinstance(options or {}, dict):
        raise EffectorError("configuration", "options must be a JSON object", device=device)
    from .models import validate_options
    validate_options(registration.package, options or {}, backend)
    return Deployment(registration, backend, tuple(mounts), copy.deepcopy(options or {}), source)


def asset_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def deployment_for_mode(device='inspire', mode='sim', *, config_path=None, options=None, registry_dir=DEFAULT_REGISTRY):
    """Choose a registered default backend; an explicit deployment must match mode."""
    if config_path:
        selected = deployment(config_path=config_path, options=options, registry_dir=registry_dir)
    else:
        devices = registrations(registry_dir)
        if device not in devices:
            raise EffectorError('configuration', f'unregistered device: {device}')
        registration = devices[device]
        candidates = [key for key, spec in registration.backends.items() if spec['mode'] == mode]
        if not candidates:
            raise EffectorError('capability', f'device does not support {mode}', device=device)
        backend = registration.integration.get(f'default_{mode}_backend', candidates[0])
        selected = deployment(device, backend, options=options, registry_dir=registry_dir)
    if selected.registration.backends[selected.backend]['mode'] != mode:
        raise EffectorError('configuration', 'deployment backend does not match selected mode')
    return selected
