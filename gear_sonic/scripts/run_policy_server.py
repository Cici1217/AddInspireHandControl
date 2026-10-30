"""Validate a registered policy contract, then run the configured GR00T server."""
import argparse
import json
import os
from pathlib import Path

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment_for_mode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--groot-root', required=True)
    parser.add_argument('--python', required=True, help='GR00T PolicyServer interpreter')
    parser.add_argument('--compute-device', default='cuda:0')
    parser.add_argument('--port', type=int, default=5550)
    parser.add_argument('--device', default='inspire')
    parser.add_argument('--registry-dir', default=str(DEFAULT_REGISTRY))
    parser.add_argument('--check', action='store_true', help='Offline files/contract check; no model load or sockets')
    return parser


def run(args):
    if not 1 <= args.port <= 65535:
        raise ValueError('invalid PolicyServer port')
    config = deployment_for_mode(args.device, 'sim', registry_dir=args.registry_dir)
    entry = config.registration.integration.get('policy_check')
    if not entry:
        raise ValueError('device has no declared policy contract')
    result = config.registration.resolve_entry(entry)(args.model)
    root = Path(args.groot_root).expanduser().resolve()
    script = root/'gr00t/eval/run_gr00t_server.py'
    if not script.is_file() or not Path(args.python).is_file():
        raise ValueError('GR00T source or PolicyServer Python missing; check 公共配置 → VLA 运行环境')
    policy = root/'gr00t/policy/gr00t_policy.py'
    if not policy.is_file() or 'rtc_previous_actions' not in policy.read_text():
        raise ValueError('GR00T runtime lacks the original RTC action-history implementation')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        print('VLA_MODEL_CHECK_OK 模型文件及设备动作定义检查通过。', flush=True)
        return result
    os.chdir(root)
    os.execv(args.python, [args.python, '-B', '-u', str(script), '--model-path', result['model'],
                         '--embodiment-tag', result['embodiment'], '--device', args.compute_device,
                         '--host', '127.0.0.1', '--port', str(args.port)])


def main():
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'VLA_MODEL_FAILED {exc}\n')


if __name__ == '__main__':
    main()
