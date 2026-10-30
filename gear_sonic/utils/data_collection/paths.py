"""Shared dataset locations, independent of the recording or replay backend."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = 'outputs/datasets'


def dataset_root(project_root=PROJECT_ROOT, configured=DEFAULT_DATASET_ROOT):
    return (Path(project_root) / Path(configured).expanduser()).resolve()


def capture_directory(sim, *, project_root=PROJECT_ROOT, configured=DEFAULT_DATASET_ROOT):
    """Classify by recording source, never by the later replay destination."""
    return dataset_root(project_root, configured) / ('sim' if sim else 'real')


def output_directories(project_root=PROJECT_ROOT, configured=DEFAULT_DATASET_ROOT):
    root = Path(project_root).resolve() / 'outputs'
    return {
        'sim': str(capture_directory(True, project_root=project_root, configured=configured)),
        'real': str(capture_directory(False, project_root=project_root, configured=configured)),
        'logs': str(root / 'console_logs'),
        'calibration': str(root / 'inspire_calibrations'),
        'replay': str(root / 'replay_runs'),
        'vla': str(root / 'vla_runs'),
        'pretrained': str(root / 'checkpoints/pretrained'),
        'finetuned': str(root / 'checkpoints/finetuned'),
    }


def describe_dataset(path):
    """Read recording provenance; folder names never override the recorded source."""
    root = Path(path).expanduser().resolve()
    if (root / 'INCOMPLETE').exists():
        return None
    try:
        info = json.loads((root / 'meta/info.json').read_text())
        capture = root / 'meta/inspire_capture.json'
        simulation = root / 'meta/simulation_replay.json'
        kind, source = 'unsupported', 'unknown'
        if capture.is_file():
            metadata = json.loads(capture.read_text())
            source = {'SIM_ONLY': 'sim', 'REAL_CAPTURE_READ_ONLY': 'real'}.get(metadata.get('validation_level'), 'unknown')
            if metadata.get('capture_schema') in ('inspire.lerobot.capture.v3', 'inspire.lerobot.capture.v4'):
                kind = 'actions'
        elif simulation.is_file():
            metadata = json.loads(simulation.read_text())
            if metadata.get('schema') == 'inspire.simulation_replay.v1':
                kind, source = 'simulation', 'sim'
        count = info['total_episodes']
        if type(count) is not int or count < 1:
            return None
        return {'path': str(root), 'source': source, 'playback': kind, 'episodes': count,
                'frames': info['total_frames'], 'modified': (root / 'meta/info.json').stat().st_mtime}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_datasets(project_root=PROJECT_ROOT, configured=DEFAULT_DATASET_ROOT):
    root = dataset_root(project_root, configured)
    result = []
    for file in root.glob('**/meta/info.json'):
        if any(part.startswith('.') for part in file.relative_to(root).parts):
            continue
        item = describe_dataset(file.parent.parent)
        if item is not None:
            relative = file.parent.parent.relative_to(root)
            if relative.parts[0] in ('sim', 'real') and len(relative.parts) > 1:
                relative = Path(*relative.parts[1:])
            item['label'] = str(relative)
            result.append(item)
    return sorted(result, key=lambda item: (-item['modified'], item['path']))
