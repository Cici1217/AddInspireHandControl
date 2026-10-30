"""Record successful simulated basket grasps and prepare N1.7 data; never train."""

import argparse
from gear_sonic.utils.hand_control.registry import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/checkpoints/pretrained/GR00T-N1.7-3B"))
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-root", default=str(ROOT / "outputs/datasets"))
    p.add_argument("--sim-python", default=str(ROOT / ".venv_sim/bin/python"))
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument(
        "--resume", action="store_true", help="Continue an incomplete local batch without replacing recordings"
    )
    a = p.parse_args()
    if not 2 <= a.episodes <= 100:
        p.error("episodes must be 2..100, including a held-out validation split")
    from gear_sonic.devices.inspire.training.prepare import prepare

    prepare(a)


if __name__ == "__main__":
    main()
