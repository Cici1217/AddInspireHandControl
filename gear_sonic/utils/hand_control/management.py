"""Runtime process records and atomic report writes for local robot tasks."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from .registry import DEFAULT_REGISTRY, ROOT, read_json


def state_dir(directory=DEFAULT_REGISTRY):
    directory = Path(directory).resolve()
    return ROOT / 'outputs/device_registry' if directory == DEFAULT_REGISTRY.resolve() else directory / '.registry'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def locked(directory=DEFAULT_REGISTRY):
    root = state_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield root


def process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def active_leases(directory=DEFAULT_REGISTRY, device=None):
    result = []
    for path in (state_dir(directory) / 'leases').glob('*.json'):
        row = read_json(path)
        processes = [row] + row.get('children', [])
        if not any(p.get('process_start') is not None and process_identity(p['pid']) == p['process_start'] for p in processes):
            continue
        if device is None or row['device'] == device:
            result.append(row)
    return result


class DeviceLease:
    def __init__(self, config, purpose='end_effector'):
        self.config, self.purpose, self.path = config, purpose, None
        self.row = None

    def start(self):
        registration = self.config.registration
        with locked(registration.registry_dir) as root:
            if self.path:
                return self
            self.row = {'id': uuid.uuid4().hex, 'pid': os.getpid(), 'process_start': process_identity(os.getpid()),
                        'device': registration.description.device_id, 'purpose': self.purpose,
                        'status': 'initializing', 'started_unix_ns': time.time_ns(),
                        'deployment': self.config.snapshot()}
            self.path = root / 'leases' / (self.row['id'] + '.json')
            atomic_json(self.path, self.row)
        return self

    def status(self, value):
        if self.path and self.row['status'] != value:
            self.row['status'] = value
            atomic_json(self.path, self.row)

    def add_process(self, pid):
        if self.path:
            self.row.setdefault('children', []).append({'pid': pid, 'process_start': process_identity(pid)})
            atomic_json(self.path, self.row)

    def close(self):
        if self.path:
            self.path.unlink(missing_ok=True)
            self.path = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()


def managed_device(device, config):
    """Attach lifecycle bookkeeping without changing adapter identity or behavior."""
    lease = DeviceLease(config)
    connect, close, send, stop = device.connect, device.close, device.send_action, device.stop

    def connected():
        lease.start()
        try:
            result = connect()
            lease.status('connected')
            return result
        except BaseException:
            lease.close()
            raise

    def sent(action):
        result = send(action)
        lease.status('controlling')
        return result

    def stopped():
        result = stop()
        lease.status('connected')
        return result

    def closed():
        result = close()
        lease.close()
        return result

    device.connect, device.close, device.send_action, device.stop = connected, closed, sent, stopped
    return device
