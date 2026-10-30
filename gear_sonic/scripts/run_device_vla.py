"""Run the registered device's VLA adapter with the existing SONIC backend."""
import argparse
import math

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment_for_mode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['sim', 'real'], default='sim')
    parser.add_argument('--device', default='inspire')
    parser.add_argument('--registry-dir', default=str(DEFAULT_REGISTRY))
    parser.add_argument('--deployment')
    parser.add_argument('--hand-config')
    parser.add_argument('--policy-host', default='127.0.0.1')
    parser.add_argument('--policy-port', type=int, default=5550)
    parser.add_argument('--camera-host', default='127.0.0.1')
    parser.add_argument('--camera-port', type=int, default=5555)
    parser.add_argument('--body-endpoint', default='tcp://127.0.0.1:5556')
    parser.add_argument('--state-endpoint', default='tcp://127.0.0.1:5557')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--duration', type=float, default=30, help='Seconds; 0 runs until stopped')
    parser.add_argument('--startup-timeout', type=float, default=60)
    parser.add_argument('--ready-pose-values', help='29 quoted body angles for preparation in sim or real')
    parser.add_argument('--initial-pose', help='Optional simulation-only body29 reference file')
    parser.add_argument('--shadow', action='store_true', help='Read observations and check one policy reply without control writes')
    parser.add_argument('--execute', action='store_true', help='Connect; Enter gates start standing and then policy control')
    parser.add_argument('--auto-start', action='store_true', help='Local simulation tests only')
    return parser


def run(args):
    if not args.prompt.strip() or not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError('prompt must be nonempty and duration finite and nonnegative')
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        raise ValueError('startup timeout must be positive')
    if any(not 1 <= port <= 65535 for port in (args.policy_port, args.camera_port)):
        raise ValueError('invalid policy/camera port')
    if args.auto_start and (args.target != 'sim' or not args.execute or args.shadow):
        raise ValueError('--auto-start is only for executed local simulation tests')
    config = deployment_for_mode(args.device, args.target, registry_dir=args.registry_dir,
                                config_path=args.deployment,
                                options={'hand_config': args.hand_config} if args.hand_config else None)
    entry = config.registration.integration.get('inference')
    if 'inference' not in config.registration.backends[config.backend]['capabilities'] or not entry:
        raise ValueError('selected device/backend does not support VLA inference')
    return config.registration.resolve_entry(entry)(args, config)


def main():
    parser = build_parser()
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print('VLA_STOPPED 已停止推理。', flush=True)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(1, f'VLA_FAILED {exc}\n')


if __name__ == '__main__':
    main()
