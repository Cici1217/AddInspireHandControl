"""Inspectable, resumable initialization steps; never delete an environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from gear_sonic.utils.data_collection.paths import output_directories
from .config import fields, relative_path, resolve_paths, validate


def inspect_configuration(value):
    """Fast filesystem checks for onboarding: no imports, subprocesses or SSH."""
    c = validate(value, check_devices=False)
    labels = {f['key']: f['label'] for f in fields()}
    groups = {
        'base': [('root', 'directory'), ('base_python', 'python')],
        'sim': [('sim_python', 'python')],
        'scenes': [('scene_root', 'directory')],
        'hands': [('hand_config', 'file')],
        'sonic': [('sonic_root', 'directory'), ('tensorrt', 'directory'),
                  ('sonic_checkpoint', 'sonic'), ('sonic_obs', 'file'),
                  ('sonic_planner', 'file'), ('sonic_motion', 'directory')],
        'pico': [('teleop_python', 'python'), ('xrt_root', 'directory'),
                 ('xrt_program', 'file'), ('calibration', 'calibration')],
        'recording': [('data_python', 'python')],
        'policy': [('groot_root', 'directory'), ('policy_python', 'python'),
                   ('model', 'model'), ('vla_sonic_checkpoint', 'sonic'), ('vla_sonic_obs', 'file')],
    }
    rows = []
    for group, entries in groups.items():
        for name, kind in entries:
            key = 'local.' + name
            path = Path(c[key])
            status, detail = 'found', '路径已找到'
            try:
                if kind == 'sonic':
                    missing = [str(path) + suffix for suffix in ('_encoder.onnx', '_decoder.onnx')
                               if not Path(str(path) + suffix).is_file()]
                    if missing:
                        status, detail = 'missing', '缺少模型文件：' + '、'.join(Path(p).name for p in missing)
                elif kind == 'model':
                    if not path.is_dir() or not (path / 'config.json').is_file():
                        status, detail = 'missing', '请选择包含 config.json 和权重的具体 checkpoint 目录'
                    elif not any(path.glob('*.safetensors')) and not any(path.glob('pytorch_model*.bin')):
                        status, detail = 'missing', '目录内未找到模型权重'
                elif kind == 'calibration' and not path.exists():
                    status, detail = 'optional', '首次 PICO 遥操时按提示标定，保存到此位置'
                elif not (path.is_dir() if kind == 'directory' else path.is_file()):
                    status, detail = 'missing', '目录不存在' if kind == 'directory' else '文件不存在'
                elif kind == 'python' and not os.access(path, os.X_OK):
                    status, detail = 'missing', 'Python 文件不可执行'
            except OSError as exc:
                status, detail = 'missing', str(exc)
            rows.append(dict(group=group, key=key, label=labels[key], status=status,
                             path=relative_path(c[key], Path(c['local.root'])), detail=detail))
    return dict(checks=rows, missing=sum(row['status'] == 'missing' for row in rows),
                found=sum(row['status'] == 'found' for row in rows),
                remote_checked=False)


def initialize_outputs(c):
    """Create missing local output folders; never move or overwrite recordings."""
    c = resolve_paths(c)
    directories = output_directories(c['local.root'])
    for path in directories.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return directories

IMPORTS = {
    "simulation": ["mujoco", "cyclonedds", "unitree_sdk2py", "cv2", "yaml", "pinocchio"],
    "teleop": ["xrobotoolkit_sdk", "zmq", "pinocchio", "yaml"],
    "capture": ["lerobot", "datasets", "av", "zmq", "pinocchio", "cv2",
                "gear_sonic.data.exporter", "gear_sonic.devices.inspire.replay"],
    "inference": ["gr00t", "torch", "transformers"],
}


def run(argv):
    import shlex

    print("$ " + shlex.join(map(str, argv)), flush=True)
    subprocess.run(argv, check=True)


def check(c):
    c = resolve_paths(c)
    role = c["setup.role"]
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for program in ("git", "ssh", "rsync", "cmake", "ffmpeg"):
        add(program, shutil.which(program), shutil.which(program) or "未找到；安装对应系统工具")
    keys = ["root"]
    keys += {
        "simulation": [
            "sim_python",
            "hand_config",
            "sonic_root",
            "tensorrt",
            "sonic_obs",
            "sonic_planner",
            "sonic_motion",
        ],
        "teleop": ["teleop_python", "xrt_program"],
        "capture": ["data_python", "hand_config"],
        "inference": ["policy_python", "model", "groot_root"],
        "g1": [],
    }[role]
    for key in keys:
        path = Path(c["local." + key])
        add(key, path.exists(), str(path))
    if role == "inference":
        from .config import selected_effector
        try:
            device = selected_effector(c, 'sim').registration
            result = device.resolve_entry(device.integration['policy_check'])(c['local.model'])
            add('设备模型合同', True, result['model'])
        except (OSError, ValueError, KeyError) as exc:
            add('设备模型合同', False, str(exc))
    if role in ("teleop", "capture") and Path(c["local.calibration"]).is_file():
        try:
            document = json.loads(Path(c["local.calibration"]).read_text())
            add("光学手标定 JSON", isinstance(document, dict), c["local.calibration"])
        except (ValueError, OSError) as exc:
            add("光学手标定 JSON", False, str(exc))
    if role in IMPORTS:
        key = {
            "simulation": "sim_python",
            "teleop": "teleop_python",
            "capture": "data_python",
            "inference": "policy_python",
        }[role]
        python = c["local." + key]
        if Path(python).is_file():
            for module in IMPORTS[role]:
                try:
                    result = subprocess.run(
                        [python, "-c", f"import {module}"], cwd=c['local.root'],
                        capture_output=True, text=True, timeout=60
                    )
                    add(module, result.returncode == 0, result.stderr[-1500:] if result.returncode else "导入通过")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    add(module, False, str(exc))
    for key, value in output_directories(c['local.root']).items():
        path = Path(value)
        parent = path
        while not parent.exists():
            parent = parent.parent
        import os

        add(key + " 可写", parent.is_dir() and os.access(parent, os.W_OK), str(path))
    print(
        json.dumps(
            {"role": role, "checks": checks, "passed": all(i["ok"] for i in checks)}, ensure_ascii=False, indent=2
        ),
        flush=True,
    )
    return 0 if all(i["ok"] for i in checks) else 1


def ensure_venv(python, base):
    target = Path(python)
    if target.name not in ("python", "python3") or target.parent.name != "bin":
        raise ValueError("新虚拟环境 Python 路径应为 /绝对目录/bin/python")
    root = target.parent.parent
    if (root / "pyvenv.cfg").exists() and target.exists():
        print(f"已有虚拟环境，保留：{root}")
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"目录非空且不是完整虚拟环境，不覆盖：{root}")
    run([base, "-m", "venv", str(root)])


def install_commands(role, python, root, groot_root):
    if role == "inference":
        return [[python, "-m", "pip", "install", "-e", groot_root]]
    extra = {"simulation": "sim", "capture": "data_collection", "teleop": "teleop"}[role]
    commands = [[python, "-m", "pip", "install", "-e", f"{root}/gear_sonic[{extra}]"]]
    if role == "simulation":
        commands.append(
            [python, "-m", "pip", "install", "-e", f"{root}/external_dependencies/unitree_sdk2_python"]
        )
    if role == "teleop":
        commands.extend(
            [
                [python, "-m", "pip", "install", "cmake", "pybind11", "setuptools"],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "-e",
                    f"{root}/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64",
                ],
            ]
        )
    return commands


def install(role, python, root, groot_root):
    if not (Path(python).parent.parent / "pyvenv.cfg").is_file():
        raise ValueError("只允许在明确的虚拟环境安装依赖")
    # Importable dependencies can still belong to another checkout/version.
    # An editable install also repairs that binding without deleting the venv.
    if subprocess.run([python, "-m", "pip", "--version"], capture_output=True).returncode:
        run([python, "-m", "ensurepip"])
    for argv in install_commands(role, python, root, groot_root):
        run(argv)
    run([python, "-c", "; ".join(f"import {m}" for m in IMPORTS[role])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "directories", "venv", "install"])
    parser.add_argument("--config")
    parser.add_argument("--python")
    parser.add_argument("--base-python", default=sys.executable)
    parser.add_argument("--role", choices=list(IMPORTS))
    parser.add_argument("--root")
    parser.add_argument("--groot-root")
    args = parser.parse_args()
    if args.action == 'directories':
        from .config import validate
        print(json.dumps(initialize_outputs(validate(json.loads(args.config))), ensure_ascii=False, indent=2))
        return
    if args.action == "check":
        raise SystemExit(check(json.loads(args.config)))
    if args.action == "venv":
        ensure_venv(args.python, args.base_python)
    else:
        install(args.role, args.python, args.root, args.groot_root)


if __name__ == "__main__":
    main()
