"""Registered SONIC body launch hook; device packages own hand transport adapters."""
import argparse
import signal
import subprocess

from gear_sonic.utils.hand_control.registry import DEFAULT_REGISTRY, deployment_for_mode
from gear_sonic.utils.hand_control.management import DeviceLease


def run_process(command, lease=None):
    """Own just the body process; preserve terminal input and forward termination."""
    process = subprocess.Popen(command)
    if lease: lease.add_process(process.pid)
    previous = {}
    def forward(number, frame):
        if process.poll() is None:
            process.send_signal(number)
    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, forward)
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for number, handler in previous.items(): signal.signal(number, handler)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='dex3')
    parser.add_argument('--registry-dir', default=str(DEFAULT_REGISTRY))
    parser.add_argument('--deployment')
    parser.add_argument('--mode', choices=['sim', 'real'], required=True)
    parser.add_argument('--config', help='Legacy hand mapping file')
    parser.add_argument('--driver'); parser.add_argument('--remote', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--native-hand', action='store_true', help='Print the registered native transport, without importing drivers')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    return parser


def run(args):
    config = deployment_for_mode(args.device, args.mode, registry_dir=args.registry_dir,
                                 config_path=args.deployment,
                                 options={'hand_config': args.config} if args.config else None)
    registration = config.registration
    if 'body' not in registration.backends[config.backend]['capabilities'] or registration.package.get('model', {}).get('body_dof') != 29:
        raise ValueError('device has no G1 body29 SONIC integration')
    native = registration.integration.get('native_hand')
    if native not in ('dex3', 'external'):
        raise ValueError('SONIC requires a declared dex3 or external native hand transport')
    if args.native_hand:
        print(native); return 0
    integration = config.body_integration()
    integration.body_preflight(args)
    if args.check: return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: raise ValueError('body executable is required')
    with DeviceLease(config, 'sonic_body') as lease:
        return integration.run_body(args, command, lease=lease)


def main():
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Device body deployment: {exc}\n')


if __name__ == '__main__': main()
