"""Console batch wrapper around the new checkout's independent recorder.

Each batch is one dataset with independently audited episodes. Shared-world/hardware capture only
subscribes to streams. Local capture explicitly owns its simulation per episode.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY
from gear_sonic.utils.data_collection.paths import capture_directory


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", help="Explicit parent directory; default: project outputs/datasets/sim or real")
    parser.add_argument("--name", default="inspire_capture")
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--episodes", type=int, default=1, help="0 captures continuously")
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--depth", choices=("off", "required"), default="off")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--hand-config")
    parser.add_argument("--device", default="inspire")
    parser.add_argument("--deployment")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--capture-input", choices=("sequence", "teleop"), default="sequence")
    parser.add_argument("--capture-python", help="Interpreter for local MuJoCo control")
    parser.add_argument("--scene", help="Custom standalone MuJoCo scene XML")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--outcome", choices=("unlabeled", "success", "failure"), default="unlabeled")
    parser.add_argument("--label-after-each", action="store_true", help="Ask for each episode's outcome after saving it, before recording the next")
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--start-immediately", action="store_true", help="Skip the console's initial Enter gate")
    parser.add_argument("--dry-run", action="store_true")
    return parser


@contextmanager
def reserve_prefix(root, name):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ("." + name + ".capture.lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another recorder owns this output prefix") from exc
        yield


def next_output(root, name):
    if not (root / name).exists():
        return root / name
    index = 1
    while (root / f"{name}_{index:03d}").exists():
        index += 1
    return root / f"{name}_{index:03d}"


def capture_event(stage, **values):
    print('CAPTURE_' + stage + ' ' + json.dumps(values, ensure_ascii=False), flush=True)


def save_outcome(output, outcome, episode_index=0):
    """Label a completed episode; never alter recorded frames or control data."""
    from gear_sonic.utils.data_collection.batch import label_outcome
    label_outcome(output, episode_index, outcome)


def label_episode(output, episode):
    print('Recording saved. Choose success, failure, or unlabeled for this episode.', flush=True)
    while True:
        capture_event('LABEL', episode=episode, output=str(output))
        outcome = input().strip().lower()
        if outcome in ('success', 'failure', 'unlabeled'):
            save_outcome(output, outcome, episode - 1)
            capture_event('LABELED', episode=episode, output=str(output), outcome=outcome)
            return
        print('Enter success, failure, or unlabeled.', flush=True)


def run(args):
    if not re.fullmatch(r"[\w.-]{1,100}", args.name) or args.name in (".", ".."):
        raise ValueError("invalid dataset name")
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 1800:
        raise ValueError("seconds must be between 1 and 1800")
    if not math.isfinite(args.wait_seconds) or not 0 <= args.wait_seconds <= 3600:
        raise ValueError("wait-seconds must be between 0 and 3600")
    if args.episodes < 0 or not args.task_prompt.strip() or not 1 <= args.camera_port <= 65535:
        raise ValueError("invalid episodes, task or camera port")
    root = Path(args.output_root).expanduser().resolve() if args.output_root else capture_directory(args.sim)
    from gear_sonic.scripts.record_pico_lerobot import capture_configuration
    config = capture_configuration(args)
    local_capture = config.registration.backends[config.backend]['capture_source'] == 'local'
    if args.dry_run:
        print(
            json.dumps(
                vars(args)
                | {
                    "output_pattern": str(root / args.name),
                    "next_output": str(next_output(root, args.name)),
                    "control_writes": local_capture,
                    "connections_opened": False,
                }
            )
        )
        return
    from gear_sonic.scripts.record_pico_lerobot import run as record

    with reserve_prefix(root, args.name):
        from gear_sonic.utils.data_collection.batch import append_episode
        output = next_output(root, args.name)
        completed = 0
        while args.episodes == 0 or completed < args.episodes:
            if completed or not args.start_immediately:
                total = str(args.episodes) if args.episodes else 'unlimited'
                print(f"Capture ready. Episode {completed + 1}/{total}: prepare the scene, then press Enter to record.", flush=True)
                capture_event('READY', episode=completed + 1, output=str(output))
                input()
            pending = Path(tempfile.mkdtemp(prefix=f'.{output.name}.pending-', dir=root))
            single = argparse.Namespace(
                **vars(args), output=str(pending / 'episode'), frames=int(args.seconds * 50), timeout=args.seconds + 30
            )
            if args.label_after_each:
                single.outcome = 'unlabeled'
            capture_event('RECORDING', episode=completed + 1, output=str(output))
            try:
                record(single)
                index = append_episode(single.output, output)
                if index != completed:
                    raise RuntimeError('batch episode index changed unexpectedly')
            except BaseException:
                print(f'Uncommitted recording diagnostics retained: {pending}', flush=True)
                raise
            else:
                shutil.rmtree(pending)
            completed += 1
            if args.label_after_each:
                label_episode(output, completed)
            print(f"Completed {completed} episode(s) in dataset: {output}", flush=True)
        capture_event('COMPLETE', episodes=completed)


def main():
    parser = build_parser()
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        parser.exit(130, "Capture stopped; saved datasets and teleoperation are retained. Pending labels remain unlabeled.\n")
    except (OSError, ValueError, RuntimeError, EOFError) as exc:
        parser.exit(1, f"Capture failed: {exc}\n")


if __name__ == "__main__":
    main()
