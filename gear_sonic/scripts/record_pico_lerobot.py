"""Record a registered end effector: read-only streams or an owned local simulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment_for_mode


def build_parser():
    parser = argparse.ArgumentParser(description="Registered end-effector episode recorder")
    parser.add_argument("--output", required=True, help="Explicit new dataset directory, e.g. outputs/datasets/sim/demo_001; never overwritten")
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument(
        "--frames", type=int, default=600, help="Valid frames to record; 600 is 12 nominal seconds"
    )
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--seconds", type=float, help="Wall-clock capture duration; overrides --frames")
    parser.add_argument("--wait-seconds", type=float, default=0, help="Delay before subscribing/recording")
    parser.add_argument("--depth", choices=("off", "required", "auto"), default="off")
    parser.add_argument("--sim", action="store_true", help="Label this dataset as simulation capture")
    parser.add_argument("--hand-config", help="Legacy Inspire configuration override")
    parser.add_argument("--device", default="inspire")
    parser.add_argument("--deployment")
    parser.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--capture-input", choices=("sequence", "teleop"), default="sequence")
    parser.add_argument("--capture-python", help="Interpreter for local MuJoCo control")
    parser.add_argument("--scene", help="Custom standalone MuJoCo scene XML")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--outcome", choices=("unlabeled", "success", "failure"), default="unlabeled")
    parser.add_argument("--camera-host", default="127.0.0.1")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    return parser


def capture_configuration(args):
    config = deployment_for_mode(getattr(args, 'device', 'inspire'), 'sim' if args.sim else 'real',
                                config_path=getattr(args, 'deployment', None),
                                registry_dir=getattr(args, 'registry_dir', DEFAULT_REGISTRY),
                                options={'hand_config': args.hand_config} if args.hand_config else None)
    spec = config.registration.backends[config.backend]
    if 'capture' not in spec['capabilities']:
        raise ValueError('selected device/backend does not support recording')
    if getattr(args, 'scene', None):
        if spec['capture_source'] != 'local' or spec['mode'] != 'sim':
            raise ValueError('--scene is only for owned local simulation; select shared scenes in run_sim_loop')
        if not Path(args.scene).is_file():
            raise ValueError(f'scene does not exist: {args.scene}')
    if spec['capture_source'] == 'local' and args.depth != 'off':
        raise ValueError('local end-effector capture currently supports RGB only')
    return config


def run(args):
    if args.seconds is not None and (not math.isfinite(args.seconds) or not 1 <= args.seconds <= 1800):
        raise ValueError("seconds must be between 1 and 1800")
    if not math.isfinite(args.wait_seconds) or not 0 <= args.wait_seconds <= 3600:
        raise ValueError("wait-seconds must be between 0 and 3600")
    if not 1 <= args.camera_port <= 65535:
        raise ValueError("invalid camera port")
    if args.frames < 40:
        raise ValueError("an episode needs at least 40 frames")
    if args.timeout <= 0 or not args.task_prompt.strip():
        raise ValueError("timeout must be positive and task-prompt non-empty")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config = capture_configuration(args)
    spec = config.registration.backends[config.backend]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "fps": config.registration.description.command_hz,
                    "frames": args.frames,
                    "output": str(output),
                    "control_writes": spec["capture_source"] == "local",
                    "device": config.registration.description.device_id,
                    "connections_opened": False,
                    "seconds": args.seconds,
                    "wait_seconds": args.wait_seconds,
                    "depth": args.depth,
                }
            )
        )
        return
    from gear_sonic.utils.data_collection.end_effector import record_episode
    return record_episode(args, config)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130, "Recording cancelled; owned local simulation is stopped; external teleoperation remains independent.\n")
    except (OSError, ValueError, RuntimeError, TimeoutError, EOFError) as exc:
        parser.exit(1, f"Recording failed: {exc}\n")


if __name__ == "__main__":
    main()
