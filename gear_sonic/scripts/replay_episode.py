"""Inspect or replay a recorded G1 + Inspire episode on simulation or hardware."""
import argparse

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment_for_mode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='Local dataset directory containing meta/info.json')
    parser.add_argument('--target', choices=('sim', 'real'), default='sim')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frames', type=int, default=0, help='0 plays the whole episode')
    parser.add_argument('--body-reference', choices=('smpl', 'joints'), default='smpl',
                        help='Body reference from the same episode: SMPL or exported measured G1 joints')
    parser.add_argument('--playback', choices=('auto', 'actions', 'simulation'), default='auto',
                        help='auto uses recorded commands when available, otherwise saved simulation states')
    parser.add_argument('--headless', action='store_true', help='Validate simulation state playback without a viewer')
    parser.add_argument('--execute', action='store_true', help='Connect; Enter starts standing, then Enter starts replay')
    parser.add_argument('--auto-start', action='store_true', help='Simulation tests only')
    parser.add_argument('--device', default='inspire')
    parser.add_argument('--registry-dir', default=str(DEFAULT_REGISTRY))
    parser.add_argument('--deployment')
    parser.add_argument('--hand-config')
    parser.add_argument('--body-endpoint', default='tcp://127.0.0.1:5556')
    parser.add_argument('--state-endpoint', default='tcp://127.0.0.1:5557')
    return parser


def run(args):
    config = deployment_for_mode(args.device, args.target, config_path=args.deployment,
                                registry_dir=args.registry_dir,
                                options={'hand_config': args.hand_config} if args.hand_config else None)
    spec = config.registration.backends[config.backend]
    entry = config.registration.integration.get('dataset_replay')
    if 'dataset_replay' not in spec['capabilities'] or not entry:
        raise ValueError('this device/backend does not support recorded full-body replay')
    return config.registration.resolve_entry(entry)(args, config)


def main():
    parser = build_parser()
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        print('REPLAY_STOPPED 回放已停止。', flush=True)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f'REPLAY_FAILED {exc}\n')


if __name__ == '__main__':
    main()
