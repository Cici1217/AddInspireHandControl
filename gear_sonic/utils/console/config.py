"""Project-relative settings, runtime path resolution and profile persistence."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import threading

from gear_sonic.utils.hand_control.registry import deployment, deployment_for_mode, registrations
from gear_sonic.utils.hand_control.interface import EffectorError
from gear_sonic.utils.data_collection.paths import capture_directory

ROOT = Path(__file__).resolve().parents[3]
STATE_DIRECTORY = ROOT / 'configs/console'


def fields(root: Path | None = None) -> list[dict]:
    root = ROOT if root is None else root
    sibling = root.parent
    definitions = [
        (
            "local",
            "本机路径",
            [
                ("root", "项目目录", str(root)),
                ("scene_root", "场景目录", "scenes"),
                ("effector_registry", "末端执行器注册目录", str(root / 'gear_sonic/devices')),
                ("effector_deployment", "自定义末端部署 JSON（可空）", ""),
                ("sim_python", "仿真 Python", str(root / ".venv_sim/bin/python")),
                ("data_python", "采集／回放 Python", str(root / ".venv_data_collection/bin/python")),
                ("teleop_python", "遥操 Python", str(root / ".venv_teleop/bin/python")),
                ("base_python", "新环境基础 Python", "/usr/bin/python3"),
                ("groot_root", "Isaac-GR00T 目录", str(sibling / "Isaac-GR00T")),
                (
                    "policy_python",
                    "PolicyServer Python",
                    str(sibling / "Isaac-GR00T/.venv/bin/python"),
                ),
                ("model", "GR00T 模型目录", str(root / "outputs/checkpoints/finetuned/your-checkpoint")),
                ("vla_sonic_checkpoint", "VLA 仿真 SONIC 模型前缀", str(root / "gear_sonic_deploy/policy/low_latency/model")),
                ("vla_sonic_obs", "VLA 仿真 SONIC 观测配置", str(root / "gear_sonic_deploy/policy/low_latency/observation_config.yaml")),
                ("vla_initial_pose", "VLA 仿真初始姿态（可空）", ""),
                ("ready_pose", "准备姿态文件（可空，电脑路径）", ""),
                ("sonic_root", "SONIC 部署目录", str(root / "gear_sonic_deploy")),
                ("tensorrt", "TensorRT 目录", str(Path.home() / "TensorRT")),
                (
                    "sonic_checkpoint",
                    "本机 SONIC 模型前缀（不带 _decoder.onnx）",
                    str(root / "gear_sonic_deploy/policy/release/model"),
                ),
                (
                    "sonic_obs",
                    "本机 SONIC observation 配置",
                    str(root / "gear_sonic_deploy/policy/release/observation_config.yaml"),
                ),
                (
                    "sonic_planner",
                    "本机 SONIC planner 模型",
                    str(root / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"),
                ),
                ("sonic_motion", "本机 SONIC reference 目录", str(root / "gear_sonic_deploy/reference/example")),
                (
                    "hand_config",
                    "本机 Inspire 手配置 YAML（关节、端点、场景）",
                    str(root / "gear_sonic/devices/inspire/assets/hand_config.yaml"),
                ),
                ("xrt_root", "XRoboToolkit 目录", "/opt/apps/roboticsservice"),
                ("xrt_program", "XRoboToolkit 启动脚本", "/opt/apps/roboticsservice/run3D.sh"),
                ("calibration", "PICO 光学手标定", str(root / "outputs/inspire_calibrations/pico_hands.json")),
                ("input", "回放数据集目录", str(root / "outputs/datasets/sim/inspire_capture_000")),
            ],
        ),
        (
            "g1",
            "G1 路径与连接",
            [
                ("ssh_host", "SSH 主机 / IP", "g1.example.invalid"),
                ("ssh_user", "SSH 用户", "unitree"),
                ("ssh_port", "SSH 端口", 22),
                ("root", "G1 个人部署目录", "/home/unitree/inspire_hand_bridge"),
                (
                    "camera_root",
                    "G1 相机项目目录（含 gear_sonic）",
                    "/home/unitree/inspire_hand_bridge/camera_server",
                ),
                (
                    "camera_python",
                    "G1 相机 Python",
                    "/home/unitree/inspire_hand_bridge/camera_server/.venv/bin/python",
                ),
                ("camera_driver", "相机驱动", "realsense"),
                ("camera_device", "相机序列号 / USB 编号（单 RealSense 可空）", ""),
                ("body_root", "G1 身体程序目录", "/home/unitree/inspire_hand_bridge/sonic"),
                ("body_script", "G1 身体启动脚本", "/home/unitree/inspire_hand_bridge/sonic/run_body.sh"),
                ("sonic_motion", "G1 动作目录（留空使用自带动作）", ""),
                ("hand_root", "G1 手桥接目录（含 bridge.env）", "/home/unitree/inspire_hand_bridge/hand_bridge"),
                ("hand_script", "G1 手桥接启动脚本", "/home/unitree/inspire_hand_bridge/hand_bridge/run.sh"),
            ],
        ),
        (
            "ports",
            "连接端口（本机端点；SSH 映射到 G1）",
            [
                ("camera", "相机端口", 5555),
                ("policy", "策略端口", 5550),
                ("body_target", "身体目标（官方后端固定 5556）", 5556),
                ("body_state", "身体状态（官方后端固定 5557）", 5557),
                ("hand_command", "手部指令", 5561),
                ("hand_state", "手部状态", 5562),
            ],
        ),
        (
            "task",
            "实验参数",
            [
                ("name", "数据集名称", "inspire_capture"),
                ("prompt", "任务描述", "While standing in place, grasp the box with both hands and lift it."),
                ("seconds", "每条采集秒数（实际时间窗口）", 15.0),
                ("episodes", "采集条数（0 表示连续采集）", 1),
                ("wait", "每条采集开始前等待秒数", 10.0),
                ("depth", "保存深度（必须有 RGB-D 输入）", False),
                ("fps", "数据频率（合同固定 50 Hz）", 50),
                ("body_hand", "末端执行器", "inspire"),
                ("real_hand", "真机末端执行器", "inspire"),
                ("effector_targets", "各设备动作目标", {}),
                ("capture_input", "仿真采集动作来源", "sequence"),
                ("outcome", "本批任务结果", "unlabeled"),
                ("pose", "手指操作", "targets"),
                ("hand_action", "测试动作", "sequence"),
                ("sim_scenes", "各设备仿真场景", {}),
                ("with_body", "手指功能同时启动站立身体控制", True),
                ("left_target", "左手六个目标角度 rad（小指→拇指旋转，空格分隔）", "0 0 0 0 0 0"),
                ("right_target", "右手六个目标角度 rad（小指→拇指旋转，空格分隔）", "0 0 0 0 0 0"),
                ("hand_seconds", "手姿过渡秒数", 2.0),
                ("episode", "回放 episode 编号", 0),
                ("replay_mode", "身体回放方式", "smpl"),
                ("replay_frames", "回放帧数（0 表示完整 episode）", 0),
                ("vla_seconds", "推理时长（0 持续运行）", 30.0),
                ("vla_keyframe", "仿真场景起始 keyframe（可空）", ""),
                ("headless", "仿真不显示窗口", False),
                ("device", "模型推理设备", "cuda:0"),
            ],
        ),
        (
            "setup",
            "初始化与传输",
            [
                ("role", "电脑角色", "simulation"),
                ("transfer_source", "传输来源（本机目录）", str(root / "gear_sonic/camera")),
                (
                    "transfer_target",
                    "传输目标（G1 目录）",
                    "/home/unitree/inspire_hand_bridge/camera_server/gear_sonic/camera",
                ),
                ("backup_target", "更新前备份目录（G1）", "/home/unitree/inspire_hand_bridge/camera_backups"),
            ],
        ),
    ]
    choices = {
        "task.capture_input": ["sequence", "teleop"],
        "task.outcome": ["unlabeled", "success", "failure"],
        "g1.camera_driver": ["realsense", "usb"],
        "task.pose": ["targets", "sequence"],
        "task.replay_mode": ["smpl", "joints"],
        "setup.role": ["simulation", "teleop", "capture", "inference", "g1"],
    }
    result = [
        dict(
            key=f"{group}.{key}",
            group=title,
            label=label,
            default=value,
            type="object" if isinstance(value, dict) else "boolean" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "text",
            choices=choices.get(f"{group}.{key}"),
        )
        for group, title, entries in definitions
        for key, label, value in entries
    ]
    for field in result:
        if field['key'] in local_path_keys():
            field['default'] = relative_path(field['default'], root)
    return result


def local_path_keys():
    return (
        'local.root', 'local.scene_root', 'local.effector_registry', 'local.effector_deployment',
        'local.sim_python', 'local.data_python', 'local.teleop_python', 'local.base_python',
        'local.groot_root', 'local.policy_python', 'local.model', 'local.vla_sonic_checkpoint',
        'local.vla_sonic_obs', 'local.vla_initial_pose', 'local.ready_pose', 'local.sonic_root',
        'local.tensorrt', 'local.sonic_checkpoint', 'local.sonic_obs', 'local.sonic_planner',
        'local.sonic_motion', 'local.hand_config', 'local.xrt_root', 'local.xrt_program',
        'local.calibration', 'local.input', 'setup.transfer_source',
    )


def relative_path(value, root):
    """Keep external paths explicit; do not dereference venv or dependency symlinks."""
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return value


def resolve_paths(config):
    """Resolve against this checkout / configured root, never the server's cwd."""
    result = dict(config)
    root = Path(os.path.abspath(ROOT / result.get('local.root', '.')))
    result['local.root'] = str(root)
    for key in local_path_keys():
        value = result.get(key)
        if value and key != 'local.root':
            result[key] = os.path.abspath(root / value)
    return result


def portable_config(config):
    """Store project files relatively; retain separately configured external tools."""
    result = resolve_paths(config)
    root = Path(result['local.root'])
    for key in local_path_keys():
        if key in result and key != 'local.root':
            result[key] = relative_path(result[key], root)
    if root == ROOT:
        result['local.root'] = '.'
    return result


def defaults() -> dict:
    return {f["key"]: f["default"] for f in fields()}


def validate(value: dict, *, check_devices=True) -> dict:
    if not isinstance(value, dict):
        raise ValueError("配置必须是 JSON 对象")
    # Output directories now follow the project. Keep old recording inputs intact.
    value = {key: item for key, item in value.items()
             if key not in ("local.logs", "local.output", "task.capture_directory", "task.replay_source")
             and not key.startswith('future.')}
    if value.get('task.replay_mode') in ('source', 'token', 'hands'):
        value['task.replay_mode'] = 'smpl'  # Former unused replay choices.
    root = value.get('local.root', str(ROOT))
    if isinstance(root, str):
        root = Path(os.path.abspath(ROOT / root))
        for key, sim in [('local.input', True)]:
            old = value.get(key)
            if old in (str(root / 'outputs/inspire_capture_000'), str(ROOT / 'outputs/inspire_capture_000'),
                       'outputs/inspire_capture_000') and not (root / old).exists():
                value[key] = str(capture_directory(sim, project_root=root) / 'inspire_capture_000')
    if value.get("task.pose") == "feedback":
        value["task.pose"] = "targets"  # Old read-only UI choice; CLI feedback remains available.
    spec = {f["key"]: f for f in fields()}
    if set(value) - set(spec):
        raise ValueError("未知配置字段：" + ", ".join(sorted(set(value) - set(spec))))
    result = defaults() | value
    for key, item in result.items():
        field = spec[key]
        if field["type"] == "object":
            if not isinstance(item, dict):
                raise ValueError(f"{field['label']} 必须是 JSON 对象")
        elif field["type"] == "boolean":
            if type(item) is not bool:
                raise ValueError(f"{field['label']} 必须是布尔值")
        elif field["type"] == "number":
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError(f"{field['label']} 必须是有限数字")
            if isinstance(field["default"], int) and int(item) != item:
                raise ValueError(f"{field['label']} 必须是整数")
            if isinstance(field["default"], int):
                result[key] = int(item)
        elif not isinstance(item, str) or any(c in item for c in "\x00\n\r") or len(item) > 4096:
            raise ValueError(f"{field['label']} 必须是单行文字")
        if field["choices"] and item not in field["choices"]:
            raise ValueError(f"{field['label']} 选项无效")
    for key in [k for k in result if k.startswith("ports.")] + ["g1.ssh_port"]:
        if not 1 <= result[key] <= 65535:
            raise ValueError(f"{key} 必须在 1–65535")
    for key, port in {
        "body_target": 5556,
        "body_state": 5557,
    }.items():
        if result[f"ports.{key}"] != port:
            raise ValueError(f"当前 SONIC / Body Gateway 固定 {key}={port}，需修改后端协议后才能更换")
    if len({result[k] for k in result if k.startswith("ports.")}) != len(
        [k for k in result if k.startswith("ports.")]
    ):
        raise ValueError("服务端口不能相互重复")
    if result["task.fps"] != 50:
        raise ValueError("现有数据合同要求 50 Hz")
    if not 1 <= result["task.seconds"] <= 1800 or not 0 <= result["task.wait"] <= 3600:
        raise ValueError("采集时长应为 1–1800 秒；等待应为 0–3600 秒")
    for key in ("task.episodes", "task.episode", "task.vla_seconds"):
        if result[key] < 0:
            raise ValueError(f"{key} 不能小于 0")
    if result["task.hand_seconds"] <= 0 or result["task.replay_frames"] < 0:
        raise ValueError("手指时长必须大于 0；回放帧数不能小于 0")
    if not re.fullmatch(r"[\w.-]{1,100}", result["task.name"]) or result["task.name"] in (".", ".."):
        raise ValueError("数据集名只能包含字母、数字、下划线、点和短横线")
    if not result["task.prompt"].strip():
        raise ValueError("任务描述不能为空")
    for device, scene in result["task.sim_scenes"].items():
        if not isinstance(device, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", device) or not isinstance(scene, str):
            raise ValueError("场景选择必须按设备保存场景名")
        if scene and (Path(scene).is_absolute() or len(Path(scene).parts) != 1 or scene in (".", "..")):
            raise ValueError("场景名应是 scenes/设备名 下的文件夹名称")
    for key in ("g1.ssh_host", "g1.ssh_user"):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", result[key]):
            raise ValueError(f"{key} 请输入主机名、IPv4 或用户名，不要填 SSH 参数")
    path_keys = [f["key"] for f in fields() if f["group"] == "本机路径"]
    path_keys += [
        "g1." + k
        for k in (
            "root",
            "camera_root",
            "camera_python",
            "body_root",
            "body_script",
            "hand_root",
            "hand_script",
        )
    ]
    path_keys += ["setup.transfer_source", "setup.transfer_target", "setup.backup_target"]
    for key in path_keys:
        if key in ('local.effector_deployment', 'local.vla_initial_pose', 'local.ready_pose') and not result[key]:
            continue
        if key in local_path_keys():
            if not result[key].strip() or result[key].startswith('~'):
                raise ValueError(f'{spec[key]["label"]}请填写项目相对路径或完整绝对路径')
            continue
        if not result[key].startswith("/"):
            raise ValueError(f"{key} 请填写所属机器上的绝对路径")
    if result["g1.sonic_motion"] and not result["g1.sonic_motion"].startswith("/"):
        raise ValueError("G1 动作目录请填写机器人上的绝对路径，或留空使用自带动作")
    result = resolve_paths(result)
    if check_devices:
        _validate_device_fields(result)
    return result


def _validate_device_fields(result):
    try:
        devices = registrations(result['local.effector_registry'])
        for key, mode in (('task.body_hand', 'sim'), ('task.real_hand', 'real')):
            if not any(b['mode'] == mode for d in devices.values() for b in d.backends.values()):
                continue
            if result[key] not in devices or not any(b['mode'] == mode for b in devices[result[key]].backends.values()):
                raise ValueError(f"{key} 选项无效：设备未注册或不支持 {mode}")
        for registration in devices.values():
            for mount, key in registration.integration.get('legacy_target_fields', {}).items():
                if key in result:
                    registration.description.validate_vector(list(map(float, result[key].split())), mount=mount)
        for device, targets in result['task.effector_targets'].items():
            if not isinstance(device, str) or not isinstance(targets, dict):
                raise ValueError(f"动作目标设备无效：{device}")
            if device not in devices:
                continue  # Retain settings for another registry; never execute them here.
            for mount, vector in targets.items():
                if not isinstance(mount, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', mount):
                    raise ValueError('动作目标安装位置无效')
                devices[device].description.validate_vector(vector, mount=mount)
    except EffectorError as exc:
        raise ValueError(str(exc)) from exc


def effector_catalog(config):
    """UI descriptions come from registration/deployment files, never driver imports."""
    config = resolve_paths(config)
    devices = registrations(config['local.effector_registry'])
    custom = None
    if config.get('local.effector_deployment'):
        custom = deployment(config_path=config['local.effector_deployment'], registry_dir=config['local.effector_registry'])
    result = []
    for device, registration in devices.items():
        row = registration.public()
        from gear_sonic.utils.mujoco_sim.scenes import list_scenes, scene_directory
        row['scenes'] = list_scenes(config['local.root'], device, config.get('local.scene_root', 'scenes'))
        row['scene_directory'] = str(scene_directory(config['local.root'], config.get('local.scene_root', 'scenes')) / device)
        row['deployments'] = {}
        for mode in ('sim', 'real'):
            if custom is not None:
                if custom.registration.description.device_id != device or registration.backends[custom.backend]['mode'] != mode:
                    continue
                selected = custom
            else:
                candidates = [name for name, spec in registration.backends.items() if spec['mode'] == mode]
                if not candidates:
                    continue
                backend = registration.integration.get(f'default_{mode}_backend', candidates[0])
                selected = deployment(device, backend, registry_dir=config['local.effector_registry'])
            row['deployments'][mode] = {'backend': selected.backend, 'mounts': list(selected.mounts),
                                        'options': selected.options,
                                        'capabilities': registration.backends[selected.backend]['capabilities']}
        result.append(row)
    return result


def selected_effector(config, mode):
    """Resolve the selected hand without enumerating UI scenes and other devices."""
    config = resolve_paths(config)
    device = config['task.real_hand' if mode == 'real' else 'task.body_hand']
    options = {'hand_config': config['local.hand_config']} if device == 'inspire' and not config.get('local.effector_deployment') else {}
    return deployment_for_mode(device, mode, options=options,
                               config_path=config.get('local.effector_deployment') or None,
                               registry_dir=config['local.effector_registry'])


def effector_targets(config, selected):
    device = selected.registration.description.device_id
    targets = config['task.effector_targets'].get(device)
    if targets is None:
        legacy = selected.registration.integration.get('legacy_target_fields', {})
        channels = selected.registration.description.action_channels
        targets = {mount: list(map(float, config[legacy[mount]].split())) if mount in legacy
                   else [min(c.upper, max(c.lower, 0)) for c in channels] for mount in selected.mounts}
    if set(targets) != set(selected.mounts):
        raise ValueError('动作目标必须包含当前部署的全部安装位置')
    for mount, values in targets.items():
        selected.registration.description.validate_vector(values, mount=mount)
    return targets


class Profiles:
    def __init__(self, directory: Path | None = None):
        use_legacy = directory is None
        directory = STATE_DIRECTORY if directory is None else Path(directory)
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "profiles.json"
        self.lock = threading.RLock()
        legacy = Path.home() / '.config/gear-sonic-newbranch-console/profiles.json'
        source = legacy if use_legacy and not self.path.exists() and legacy.is_file() else self.path
        self.data = (
            json.loads(source.read_text())
            if source.exists()
            else {"active": "默认", "profiles": {"默认": defaults()}}
        )
        self.data["profiles"] = {name: portable_config(validate(value, check_devices=False))
                                 for name, value in self.data["profiles"].items()}
        self._persist()

    def _persist(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data))

    def save(self, name, config):
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise ValueError("配置名称应为 1–100 字符")
        config = portable_config(validate(config))
        with self.lock:
            self.data["profiles"][name] = config
            self.data["active"] = name
            self._persist()
        return self.snapshot()
