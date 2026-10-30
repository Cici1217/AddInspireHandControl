"""Check the G1 launcher without DDS, model inference or robot commands."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def launch(tmp_path):
    root = tmp_path / "sonic with spaces"
    root.mkdir()
    source = Path(__file__).resolve().parents[2] / "gear_sonic_deploy/run_body.sh"
    shutil.copy2(source, root / "run_body.sh")
    for name in (
        "policy/low_latency/model_decoder.onnx",
        "policy/low_latency/model_encoder.onnx",
        "policy/low_latency/observation_config.yaml",
        "planner/target_vel/V2/planner_sonic.onnx",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    (root / "reference/example").mkdir(parents=True)
    backend = root / "target/release/g1_deploy_onnx_ref"
    backend.parent.mkdir(parents=True)
    backend.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > launched-args\n')
    backend.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ip = fake_bin / "ip"
    ip.write_text(
        "#!/bin/sh\nprintf '%s\\n' '1: lo inet 127.0.0.1/8' '2: wifi inet 10.42.0.1/24' "
        "'3: enP8p1s0 inet 192.168.123.164/24'\n"
    )
    ip.chmod(0o755)

    def run(*args):
        return subprocess.run(
            ["bash", str(root / "run_body.sh"), *args],
            env={**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
            capture_output=True,
            text=True,
        )

    return run, root, ip


@pytest.mark.parametrize("args", [(), ("--keyboard",)])
def test_check_never_launches_body(launch, args):
    run, root, _ = launch
    result = run(*args, "--check")
    assert result.returncode == 0, result.stderr
    assert "interface=enP8p1s0" in result.stdout
    assert not (root / "launched-args").exists()


@pytest.mark.parametrize("mode, args", [("zmq_manager", ()), ("keyboard", ("--keyboard",))])
def test_launch_uses_internal_dds_and_independent_inspire(launch, mode, args):
    run, root, _ = launch
    result = run(*args)
    assert result.returncode == 0, result.stderr
    argv = (root / "launched-args").read_text().splitlines()
    assert argv[0] == "enP8p1s0"
    assert argv[argv.index("--input-type") + 1] == mode
    assert argv[argv.index("--hand") + 1] == "inspire"
    assert argv[argv.index("--zmq-host") + 1] == "127.0.0.1"
    assert "--disable-crc-check" not in argv


def test_missing_internal_network_does_not_fall_back_to_wifi(launch):
    run, root, ip = launch
    ip.write_text("#!/bin/sh\necho '2: wifi inet 10.42.0.1/24'\n")
    result = run()
    assert result.returncode != 0
    assert "internal DDS interface" in result.stderr
    assert not (root / "launched-args").exists()


def test_custom_motion_directory_is_one_argument_and_missing_path_never_launches(launch):
    run, root, _ = launch
    motions = root / "my 'motions'"
    motions.mkdir()
    result = run("--keyboard", "--motion-data", str(motions), "--check")
    assert result.returncode == 0, result.stderr
    assert not (root / "launched-args").exists()
    result = run("--motion-data", str(motions), "--keyboard")
    assert result.returncode == 0, result.stderr
    assert (root / "launched-args").read_text().splitlines()[2] == str(motions)
    (root / "launched-args").unlink()
    for args in (("--motion-data",), ("--motion-data", "/missing/motions")):
        assert run(*args).returncode != 0
        assert not (root / "launched-args").exists()


def test_dex3_hand_option_and_invalid_selection(launch):
    run, root, _ = launch
    result = run("--keyboard", "--hand", "dex3", "--check")
    assert result.returncode == 0, result.stderr
    assert not (root / "launched-args").exists()
    result = run("--keyboard", "--hand", "dex3")
    assert result.returncode == 0, result.stderr
    argv = (root / "launched-args").read_text().splitlines()
    assert argv[argv.index("--hand") + 1] == "dex3"
    (root / "launched-args").unlink()
    for args in (("--hand",), ("--hand", "gripper")):
        assert run(*args).returncode != 0
        assert not (root / "launched-args").exists()


def test_ready_pose_refuses_an_old_native_binary_before_control(launch):
    run,root,_=launch
    result=run('--ready-pose',' '.join(['0']*29),'--check')
    assert result.returncode != 0
    assert 'rebuild on G1' in result.stderr
    # Fixture writes argv for every invocation: only the read-only probe ran.
    assert (root/'launched-args').read_text().splitlines()[0]=='--check-ready-pose'


@pytest.mark.parametrize('check', [True, False])
def test_ready_pose_probe_is_separate_from_control_launch(launch, check):
    run, root, _ = launch
    backend = root / 'target/release/g1_deploy_onnx_ref'
    backend.write_text('''#!/bin/sh
if [ "$1" = --check-ready-pose ]; then
    printf '%s\\n' "$2" > checked-pose
    echo READY_POSE_VALID_NO_CONTROL
    exit 0
fi
printf '%s\\n' "$@" > launched-args
''')
    values = ' '.join(['0'] * 29)
    result = run('--keyboard', '--ready-pose', values, *(['--check'] if check else []))
    assert result.returncode == 0, result.stderr
    assert (root / 'checked-pose').read_text().strip() == values
    assert (root / 'launched-args').exists() is (not check)
    if not check:
        argv = (root / 'launched-args').read_text().splitlines()
        assert argv[argv.index('--ready-pose') + 1] == values
