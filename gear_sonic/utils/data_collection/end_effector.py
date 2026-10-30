"""Device-described trajectory records and GR00T modality mappings.

The same writer accepts bench or streamed observations. It never opens a control
connection. Native records are transactional NPZ + lossless RGB PNG; an optional
LeRobot conversion uses the project's existing Gr00tDataExporter.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import time
import zlib

import numpy as np

from gear_sonic.utils.hand_control.interface import Channel, DeviceDescription, EffectorAction, EffectorError


def recording_metadata(deployment, *, task, outcome='unlabeled', source):
    """Shared device provenance for native trajectories and existing stream datasets."""
    if not task.strip() or outcome not in ('success', 'failure', 'unlabeled'):
        raise ValueError('task and valid episode outcome are required')
    snapshot = deployment.snapshot()
    from gear_sonic.utils.hand_control.registry import asset_path
    resources = {}
    for option, spec in deployment.registration.integration.get('configuration_files', {}).items():
        value = deployment.options.get(option, spec.get('default'))
        path = deployment.path(value) if value else None
        if path is None or not path.is_file():
            if spec.get('required', True):
                raise ValueError(f'missing configuration file: {option}={value}')
            resources[option] = {'path': str(path) if path else None, 'present': False}
            continue
        raw = path.read_bytes()
        resources[option] = {'path': str(path), 'present': True,
                             'sha256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode('utf-8')}
    from gear_sonic.utils.hand_control.models import package_fingerprint
    return {'task': task, 'outcome': outcome, 'source': source, 'backend': deployment.backend,
            'deployment': snapshot,
            'device_package': package_fingerprint(deployment.registration),
            'configuration_files': resources,
            'configuration_sha256': hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
            'retargeting_version': (deployment.registration.retargeting or {}).get('version'),
            'created_unix_ns': time.time_ns(),
            'validation': 'simulation' if deployment.registration.backends[deployment.backend]['mode'] == 'sim' else 'unverified_real',
            'checkpoint_compatibility': 'requires matching action/state schema and training configuration'}


def preprocessing_config(features, modality, snapshot):
    """Validate GR00T field slices and describe the training input/output contract.

    This is a dataset contract, not a trained policy or normalization statistics.
    The legacy stream adapter may supply its existing body representation.
    """
    fields = {}
    for group in ('state', 'action', 'video'):
        fields[group] = {}
        for name, item in modality[group].items():
            key = item['original_key']
            if key not in features:
                raise ValueError(f'missing preprocessing feature: {key}')
            feature = features[key]
            shape = list(feature['shape'])
            if group != 'video':
                start, end = item['start'], item['end']
                if (len(shape) != 1 or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= shape[0]
                        or len(feature['names']) != shape[0]):
                    raise ValueError(f'invalid preprocessing slice: {group}.{name}')
                names = list(feature['names'][start:end])
            else:
                if feature['dtype'] != 'video' or len(shape) != 3:
                    raise ValueError(f'invalid video feature: {key}')
                names = list(feature['names'])
            fields[group][name] = dict(item, dtype=feature['dtype'], source_shape=shape, names=names)
    description = snapshot['registration']
    return {'schema': 'end_effector.preprocessing.v1', 'fields': fields,
            'end_effector': {key: description[key] for key in
                             ('device_id', 'control_mode', 'state_channels', 'action_channels')},
            'mounts': snapshot['mounts'], 'nominal_fps': description['command_hz'],
            'normalization': 'fit using the training split; not provided by this schema',
            'checkpoint_compatibility': 'matching schema is necessary, not proof of policy compatibility'}


def validate_preprocessing(actual, expected):
    """Reject a dataset whose selected fields or action semantics changed."""
    if any(not isinstance(value, dict) or value.get('schema') != 'end_effector.preprocessing.v1'
           for value in (actual, expected)):
        raise ValueError('unsupported preprocessing schema')
    for key in ('schema', 'fields', 'end_effector', 'mounts', 'nominal_fps'):
        if key not in actual or key not in expected or actual[key] != expected[key]:
            raise ValueError(f'preprocessing schema mismatch: {key}')


def png_bytes(image):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("camera image must be uint8 HxWx3 RGB")
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    height, width, _ = array.shape
    pixels = b"".join(b"\0" + row.tobytes() for row in array)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def modality_config(description, mounts, *, body_state_dim=0, body_action_dim=0):
    """Per-field slices, never a fixed padded vector or hardcoded native6 layout."""
    state, action = {}, {}
    if body_state_dim:
        state["body"] = {"start": 0, "end": body_state_dim, "original_key": "observation.body"}
    if body_action_dim:
        action["body"] = {"start": 0, "end": body_action_dim, "original_key": "action.body"}
    for mount in mounts:
        state[mount] = {"start": 0, "end": description.state_dim, "original_key": f"observation.effectors.{mount}"}
        action[mount] = {"start": 0, "end": description.action_dim, "original_key": f"action.effectors.{mount}"}
    return {"state": state, "action": action, "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}}}


class TrajectoryWriter:
    def __init__(self, output, deployment, *, task, source, outcome="unlabeled", body_state_names=(), body_action_names=()):
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        if not task.strip() or outcome not in ("success", "failure", "unlabeled"):
            raise ValueError("task and valid episode outcome are required")
        self.deployment, self.description = deployment, deployment.registration.description
        self.mounts = deployment.mounts
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=f".{self.output.name}-", dir=self.output.parent))
        (self.staging / "images").mkdir()
        self.rows, self.images = [], []
        self.body_state_names, self.body_action_names = tuple(body_state_names), tuple(body_action_names)
        self.meta = recording_metadata(deployment, task=task, outcome=outcome, source=source) | {
                     "schema": "end_effector.trajectory.v1",
                     "body_state_names": list(body_state_names), "body_action_names": list(body_action_names),
                     "timestamp_clock": "host_monotonic_ns"}
        self._finished = False
        self._image_stamp = -1
        self._image_shape = None

    def append(self, *, state, receipt, intent, image, image_timestamp_ns, body_state=(), body_action=(), tracking_valid=True, sample_ns=None):
        if self._finished:
            raise RuntimeError("writer is closed")
        now = time.monotonic_ns() if sample_ns is None else sample_ns
        # Missing/stale measurements are explicit; they are never replaced by targets.
        state.require_fresh(now, self.description)
        if not 0 <= now-receipt.submitted_monotonic_ns <= self.description.stale_after_s*1e9:
            raise EffectorError("action_stale", "recording command is stale", device=self.description.device_id)
        if not 0 <= now-image_timestamp_ns <= self.description.stale_after_s*1e9:
            raise EffectorError("camera_stale", "recording image is stale", device=self.description.device_id)
        if self.rows and now <= self.rows[-1]["timestamp_ns"]:
            raise ValueError("sample timestamps must increase")
        if self.rows and (receipt.submitted_monotonic_ns < self.rows[-1]['submitted_ns']
                          or receipt.sequence < self.rows[-1]['command_sequence']):
            raise ValueError("command submission order regressed")
        if set(state.values) != set(self.mounts) or set(receipt.action.values) != set(self.mounts) or set(intent.values) != set(self.mounts):
            raise ValueError("frame mounts do not match deployment")
        row = {"timestamp_ns": now, "feedback_received_ns": state.received_monotonic_ns,
               "feedback_source_ns": state.source_timestamp_ns if state.source_timestamp_ns is not None else -1,
               "submitted_ns": receipt.submitted_monotonic_ns, "source_ns": receipt.action.source_timestamp_ns,
               "command_sequence": receipt.sequence, "tracking_valid": bool(tracking_valid),
               "camera_timestamp_ns": image_timestamp_ns, "fault": state.fault or ""}
        for mount in self.mounts:
            if not state.valid.get(mount) or state.values[mount] is None:
                raise EffectorError("feedback_invalid", "missing measurement", device=self.description.device_id, mount=mount)
            row[f"observation.effectors.{mount}"] = self.description.validate_vector(state.values[mount], action=False, mount=mount)
            row[f"action.effectors.{mount}"] = self.description.validate_vector(receipt.action.values[mount], mount=mount)
            row[f"intent.effectors.{mount}"] = self.description.validate_vector(intent.values[mount], mount=mount)
            row[f"valid.{mount}"] = bool(state.valid[mount])
        for key, values, names in (("observation.body", body_state, self.body_state_names), ("action.body", body_action, self.body_action_names)):
            vector = np.asarray(values, dtype=np.float32)
            if vector.shape != (len(names),) or not np.isfinite(vector).all():
                raise ValueError(f"invalid {key}, expected dimension {len(names)}")
            row[key] = tuple(map(float, vector))
        if image_timestamp_ns < self._image_stamp:
            raise ValueError("image timestamp regressed")
        if image_timestamp_ns != self._image_stamp:
            path = f"images/{len(self.images):06d}.png"
            encoded = png_bytes(image)
            if self._image_shape is not None and tuple(image.shape) != self._image_shape:
                raise ValueError("image shape changed")
            self._image_shape = tuple(image.shape)
            (self.staging / path).write_bytes(encoded)
            self.images.append(path)
            self._image_stamp = image_timestamp_ns
        row["image_index"] = len(self.images)-1
        self.rows.append(row)

    def commit(self, diagnostics=None):
        if self._finished or not self.rows:
            raise ValueError("cannot commit empty/closed episode")
        data = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        np.savez_compressed(self.staging / "trajectory.npz", **data)
        self.meta.update(frames=len(self.rows), images=self.images, image_shape=self._image_shape,
                         diagnostics=diagnostics or {}, nominal_fps=self.description.command_hz)
        (self.staging / "metadata.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))
        (self.staging / "modality.json").write_text(json.dumps(modality_config(self.description, self.mounts, body_state_dim=len(self.body_state_names), body_action_dim=len(self.body_action_names)), indent=2))
        audit_trajectory(self.staging)
        if self.output.exists():
            raise FileExistsError(self.output)
        self.staging.rename(self.output)
        self._finished = True
        return self.output

    def abort(self):
        if not self._finished:
            shutil.rmtree(self.staging)
            self._finished = True


def audit_trajectory(path):
    """Validate actual files, dimensions, timing and modality slices before conversion."""
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta["schema"] != "end_effector.trajectory.v1":
        raise ValueError("unsupported trajectory schema")
    if not isinstance(meta.get('task'), str) or not meta['task'].strip() or meta.get('outcome') not in ('success', 'failure', 'unlabeled'):
        raise ValueError("invalid episode task/outcome")
    description = meta["deployment"]["registration"]
    snapshot = meta["deployment"]
    if hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest() != meta["configuration_sha256"]:
        raise ValueError("configuration hash mismatch")
    for name, resource in meta.get('configuration_files', {}).items():
        if resource['present'] and hashlib.sha256(resource['text'].encode('utf-8')).hexdigest() != resource['sha256']:
            raise ValueError(f'configuration file hash mismatch: {name}')
    spec = {key: description[key] for key in DeviceDescription.__dataclass_fields__}
    for key in ('state_channels', 'action_channels'):
        spec[key] = tuple(Channel(**c) for c in spec[key])
    spec['physical_joints'] = tuple(spec['physical_joints'])
    device = DeviceDescription(**spec)
    if description['action_dim'] != device.action_dim or description['state_dim'] != device.state_dim:
        raise ValueError("stored dimensions disagree with channels")
    mounts = snapshot['mounts']
    if not mounts or any(not isinstance(m, str) for m in mounts) or len(set(mounts)) != len(mounts):
        raise ValueError("invalid stored mounts")
    with np.load(path / "trajectory.npz", allow_pickle=False) as data:
        count = meta["frames"]
        for key in data.files:
            if len(data[key]) != count:
                raise ValueError(f"wrong frame count: {key}")
            if data[key].dtype.kind in "f" and not np.isfinite(data[key]).all():
                raise ValueError(f"nonfinite data: {key}")
        if count < 2 or np.any(np.diff(data["timestamp_ns"])<=0):
            raise ValueError("episode needs increasing sample timestamps")
        for key in ('timestamp_ns', 'submitted_ns', 'feedback_received_ns', 'camera_timestamp_ns', 'source_ns', 'command_sequence'):
            values = data[key]
            if values.shape != (count,) or values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values[1:] < values[:-1]):
                raise ValueError(f"invalid or regressed {key}")
        for key in ('submitted_ns', 'feedback_received_ns', 'camera_timestamp_ns'):
            age = data['timestamp_ns'] - data[key]
            if np.any(age < 0) or np.any(age > device.stale_after_s * 1e9):
                raise ValueError(f"stale/future recording source: {key}")
        if np.any(data['fault'] != ''):
            raise ValueError("faulted samples cannot be training observations")
        for mount in snapshot["mounts"]:
            for prefix, dim in (("observation", description["state_dim"]), ("action", description["action_dim"]), ("intent", description["action_dim"])):
                if data[f"{prefix}.effectors.{mount}"].shape != (count, dim):
                    raise ValueError(f"wrong {prefix} dimension for {mount}")
                for vector in data[f'{prefix}.effectors.{mount}']:
                    device.validate_vector(vector, action=prefix != 'observation', mount=mount)
            if data[f'valid.{mount}'].shape != (count,) or data[f'valid.{mount}'].dtype.kind != 'b' or not data[f'valid.{mount}'].all():
                raise ValueError(f"invalid measured feedback: {mount}")
        for key, names in (('observation.body', meta['body_state_names']), ('action.body', meta['body_action_names'])):
            if data[key].shape != (count, len(names)):
                raise ValueError(f"body channel dimensions mismatch: {key}")
        modality = json.loads((path / "modality.json").read_text())
        expected = modality_config(device, mounts, body_state_dim=len(meta['body_state_names']), body_action_dim=len(meta['body_action_names']))
        if modality != expected:
            raise ValueError("modality slice/field mapping mismatch")
        for group in ("state", "action"):
            for spec in modality[group].values():
                if spec["start"] != 0 or spec["end"] != data[spec["original_key"]].shape[1]:
                    raise ValueError("modality slice mismatch")
        for image in meta["images"]:
            file = (path / image).resolve()
            if not file.is_relative_to((path / 'images').resolve()):
                raise ValueError("image path escapes recording")
            validate_png(file.read_bytes(), meta['image_shape'])
        if data["image_index"].min()<0 or data["image_index"].max()>=len(meta["images"]):
            raise ValueError("invalid image references")
        return {"frames": count, "device": description["device_id"], "action_dim_per_mount": description["action_dim"],
                "state_dim_per_mount": description["state_dim"], "body_state_dim": data["observation.body"].shape[1],
                "images": len(meta["images"]), "outcome": meta["outcome"], "validation": meta["validation"]}


def validate_png(raw, shape):
    """Check lossless RGB payloads, dimensions and CRCs without image dependencies."""
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('invalid PNG signature')
    offset, pixels, header, ended = 8, bytearray(), None, False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError('truncated PNG chunk')
        length = struct.unpack('>I', raw[offset:offset+4])[0]
        tag = raw[offset+4:offset+8]
        content = raw[offset+8:offset+8+length]
        end = offset + 12 + length
        if end > len(raw) or zlib.crc32(tag+content) & 0xffffffff != struct.unpack('>I', raw[end-4:end])[0]:
            raise ValueError('invalid PNG chunk CRC/length')
        if tag == b'IHDR':
            if header is not None or offset != 8 or length != 13:
                raise ValueError('invalid PNG header')
            header = struct.unpack('>IIBBBBB', content)
        elif tag == b'IDAT':
            pixels.extend(content)
        elif tag == b'IEND':
            ended = length == 0 and end == len(raw)
        offset = end
    if not ended or header is None or list(shape) != [header[1], header[0], 3] or header[2:] != (8, 2, 0, 0, 0):
        raise ValueError('PNG shape/encoding mismatch')
    try:
        decompressor = zlib.decompressobj()
        expected = shape[0]*(1+shape[1]*3)
        decoded = decompressor.decompress(pixels, expected+1)
        if len(decoded) != expected or not decompressor.eof or decompressor.unused_data:
            raise ValueError('invalid PNG pixel data length')
        if any(decoded[i*(1+shape[1]*3)] != 0 for i in range(shape[0])):
            raise ValueError('unsupported PNG row filter')
    except zlib.error as exc:
        raise ValueError('invalid PNG compressed pixels') from exc


def export_lerobot(source, output):
    """Publish a complete conversion atomically; never overwrite an existing dataset."""
    audit_trajectory(source)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        _export_lerobot(source, staging / 'dataset')
        if output.exists():
            raise FileExistsError(output)
        (staging / 'dataset').rename(output)
        return output
    finally:
        shutil.rmtree(staging)


def _export_lerobot(source, output):
    """Use the recorded schema rather than the device's current configuration."""
    from PIL import Image
    from gear_sonic.data.exporter import Gr00tDataExporter
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((source / "metadata.json").read_text())
    modality = json.loads((source / "modality.json").read_text())
    desc = meta["deployment"]["registration"]
    features = {"observation.images.ego_view": {"dtype":"video", "shape":tuple(meta["image_shape"]), "names":["height","width","channels"]}}
    with np.load(source / "trajectory.npz", allow_pickle=False) as data:
        diagnostic_fields = ('timestamp_ns', 'submitted_ns', 'source_ns', 'feedback_received_ns',
                             'feedback_source_ns', 'camera_timestamp_ns', 'command_sequence', 'tracking_valid')
        for key in diagnostic_fields:
            features[f'diagnostics.{key}'] = {'dtype': 'int64', 'shape': (1,), 'names': [key]}
        for key in data.files:
            if key.startswith(("observation.","action.","intent.")) and data[key].shape[1]:
                if key.endswith(".body"):
                    names = meta["body_state_names" if key.startswith("observation.") else "body_action_names"]
                else:
                    names = [c["name"] for c in desc["state_channels" if key.startswith("observation.") else "action_channels"]]
                features[key] = {"dtype":"float32", "shape":(data[key].shape[1],), "names":names}
        preprocessing = preprocessing_config(features, modality, meta['deployment'])
        writer = Gr00tDataExporter.create(save_root=output, fps=int(meta["nominal_fps"]), features=features,
                                         modality_config=modality, task=meta["task"],
                                         script_config={"source_schema":meta["schema"], "device":desc["device_id"]},
                                         robot_type=desc["device_id"], overwrite_existing=False)
        try:
            for i in range(meta["frames"]):
                frame = {k: np.asarray(data[k][i], dtype=np.float32) for k in features if k.startswith(('observation.', 'action.', 'intent.')) and k != "observation.images.ego_view"}
                frame.update({f'diagnostics.{key}': np.array([data[key][i]], dtype=np.int64) for key in diagnostic_fields})
                with Image.open(source / meta["images"][int(data["image_index"][i])]) as image:
                    frame["observation.images.ego_view"] = np.array(image.convert("RGB"))
                writer.add_frame(frame)
            writer.save_episode()
        finally:
            for video_writer in getattr(writer, "video_writers", {}).values():
                video_writer.cancel()
            writer.video_writers = {}
    (output / "meta/end_effector.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (output / "meta/preprocessing.json").write_text(json.dumps(preprocessing, ensure_ascii=False, indent=2))
    return output


def record_episode(args, config):
    """Common recording entry: registered stream adapter or owned local simulation."""
    from gear_sonic.utils.hand_control.management import DeviceLease
    with DeviceLease(config, 'capture'):
        return _record_episode(args, config)


def _record_episode(args, config):
    import os
    import subprocess
    import sys
    from gear_sonic.utils.hand_control.registry import resolve_entry

    spec = config.registration.backends[config.backend]
    if spec['capture_source'] == 'streams':
        if not spec.get('recorder'):
            raise EffectorError('capability', 'stream recorder is not registered')
        return config.registration.resolve_entry(spec['recorder'])(args, config)
    if args.wait_seconds:
        print(f'Waiting {args.wait_seconds:g} seconds before capture.', flush=True)
        time.sleep(args.wait_seconds)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f'.{output.name}.capture-', dir=output.parent))
    try:
        # Resolve deployment once. A separate simulation interpreter avoids requiring
        # the data-export environment to contain the physics or headset SDK.
        config_file = root/'deployment.json'
        config_file.write_text(json.dumps({'schema':'end_effector.deployment.v1',
                                          'device':config.registration.description.device_id,
                                          'backend':config.backend, 'mounts':config.mounts, 'options':config.options}))
        command = [getattr(args,'capture_python',None) or sys.executable, '-B', '-u', '-m',
                   'gear_sonic.scripts.run_end_effector', '--deployment', str(config_file),
                   '--registry-dir', str(config.registration.registry_dir),
                   '--operation', getattr(args,'capture_input','sequence'),
                   '--seconds', str(args.seconds if args.seconds is not None else args.frames/config.registration.description.command_hz),
                   '--record', str(root/'native'), '--task', args.task_prompt,
                   '--outcome', getattr(args,'outcome','unlabeled')]
        env = dict(os.environ)
        if getattr(args,'scene',None):
            command += ['--scene', str(Path(args.scene).resolve())]
        if getattr(args,'view',False):
            command += ['--view']
        else:
            env['MUJOCO_GL'] = 'egl'
        print('Recording started. Local simulation owns control for this episode.', flush=True)
        child = subprocess.Popen(command, env=env)
        try:
            if child.wait() != 0:
                raise RuntimeError('local simulation capture failed; no dataset published')
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            raise
        export_lerobot(root/'native', root/'dataset')
        (root/'native').rename(root/'dataset/raw')
        if output.exists():
            raise FileExistsError(output)
        (root/'dataset').rename(output)
        print(f'Saved dataset: {output}', flush=True)
        return output
    finally:
        shutil.rmtree(root)
