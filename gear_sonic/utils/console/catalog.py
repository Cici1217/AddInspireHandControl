"""Runbook-backed plans. These exact commands are both displayed and executed."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from .config import validate, selected_effector, effector_targets

FEATURES = [
    {
        "id": "body",
        "title": "身体控制",
        "icon": "00",
        "description": "SONIC 键盘控制：慢走、走路、跑步",
        "modes": ["sim", "real"],
    },
    {
        "id": "hands",
        "title": "手指控制",
        "icon": "01",
        "description": "指定角度，或执行张开、合拢、逐指测试",
        "modes": ["sim", "real"],
    },
    {
        "id": "scene",
        "hidden": True,  # Keep existing command plans compatible; scenes are selected per task.
        "title": "仿真场景",
        "icon": "02",
        "description": "选择默认场景或 scenes/ 下的自定义场景",
        "modes": ["sim"],
    },
    {
        "id": "teleop",
        "title": "PICO 遥操作",
        "icon": "03",
        "description": "身体与双手，按步骤连接",
        "modes": ["sim", "real"],
    },
    {
        "id": "capture",
        "title": "数据采集",
        "icon": "04",
        "description": "RGB / 深度、动作与时间对应",
        "modes": ["sim", "real"],
    },
    {
        "id": "replay",
        "title": "SONIC 动作回放",
        "icon": "05",
        "description": "回放自己的录制数据：SMPL 人体参考或实测关节角参考",
        "modes": ["sim", "real"],
    },
    {
        "id": "vla",
        "title": "VLA 推理",
        "icon": "06",
        "description": "GR00T 图像与状态推理，SONIC 身体与 Inspire 双手执行",
        "modes": ["sim", "real"],
    },
    {
        "id": "setup",
        "title": "新电脑初始化",
        "icon": "07",
        "description": "按角色检查环境、安装和传输",
        "modes": ["local"],
    },
]
PREFIX = "gear_sonic.scripts."
BOOTSTRAP = "gear_sonic.utils.console.bootstrap"


def ssh(c, tty=False):
    return [
        "ssh",
        "-tt" if tty else "-T",
        "-p",
        str(c["g1.ssh_port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        f"{c['g1.ssh_user']}@{c['g1.ssh_host']}",
    ]


class Builder:
    def __init__(self, config, feature, mode):
        self.c = validate(config)
        self.feature, self.mode = feature, mode
        self.device = selected_effector(self.c, mode) if mode in ('sim', 'real') and feature in ('body','hands','scene','teleop','capture','replay','vla') else None
        self.integration = self.device.registration.integration if self.device else {}
        self.hand_type = self.device.registration.description.device_id if self.device else None
        hand_config = self.device.options.get('hand_config') if self.device else None
        self.hand_config = str(self.device.path(hand_config)) if hand_config else None
        self.capabilities = self.device.registration.backends[self.device.backend]['capabilities'] if self.device else []
        self.scene_path = None
        if self.device and mode == 'sim':
            name = self.c['task.sim_scenes'].get(self.hand_type)
            if name:
                from gear_sonic.utils.mujoco_sim.scenes import resolve_scene
                self.scene_path = str(resolve_scene(self.c['local.root'], self.hand_type, name, self.c['local.scene_root']))
        self.ready_pose_values = None
        if (self.device and 'body' in self.capabilities and feature in ('body','hands','teleop','capture','vla')
                and self.c['local.ready_pose']):
            from gear_sonic.utils.inference.ready_pose import load_ready_pose
            self.ready_pose_values = ' '.join(map(str, load_ready_pose(self.c['local.ready_pose'])))
            if mode == 'sim' and feature == 'vla' and (self.c['local.vla_initial_pose'] or self.c['task.vla_keyframe']):
                raise ValueError('使用准备姿态时，请清空旧的 VLA 仿真姿态文件和场景 keyframe；两套设置不能同时使用。')
        self.steps = []
        self.simulation_replay = False
        if feature == 'replay':
            from gear_sonic.utils.data_collection.paths import describe_dataset
            recorded = describe_dataset(self.c['local.input'])
            self.simulation_replay = bool(recorded and recorded['playback'] == 'simulation')
            if self.simulation_replay and mode != 'sim':
                raise ValueError('这份数据保存的是仿真物理状态，请选择仿真查看；真机回放需要录制的控制指令。')

    def ready_pose_args(self):
        return ["--ready-pose", self.ready_pose_values] if self.ready_pose_values is not None else []

    def scene_args(self):
        return ["--scene", self.scene_path] if self.scene_path else []

    def device_args(self, *, backend=True):
        args = ['--device', self.device.registration.description.device_id,
                '--registry-dir', self.c['local.effector_registry']]
        if backend:
            args += ['--backend', self.device.backend]
        if self.c['local.effector_deployment']:
            args += ['--deployment', self.c['local.effector_deployment']]
        if self.integration.get('hand_config'):
            args += ['--hand-config', str(self.hand_config)]
        return args

    def add(
        self,
        key,
        title,
        argv,
        *,
        machine="local",
        cwd=None,
        env=None,
        ready="",
        note="",
        inputs=None,
        service=False,
        outputs=None,
        stop_order=50,
        resources=None,
        deps=None,
    ):
        c = self.c
        cwd = cwd or c["local.root" if machine == "local" else "g1.root"]
        env = {"PYTHONUNBUFFERED": "1", **(env or {})}
        argv = list(map(str, argv))
        command = (
            "cd "
            + shlex.quote(cwd)
            + " && exec "
            + shlex.join(["env", *[f"{k}={v}" for k, v in env.items()], *argv])
        )
        actual = ["bash", "-c", command] if machine == "local" else ssh(c, True) + [command]
        self.steps.append(
            dict(
                id=key,
                title=title,
                machine=machine,
                cwd=cwd,
                env=env,
                argv=argv,
                command=command,
                launch_argv=actual,
                launch_command=shlex.join(actual),
                ready=ready,
                note=note,
                inputs=inputs or [],
                service=service,
                outputs=outputs or [],
                stop_order=stop_order,
                resources=resources or [f"{machine}:{key}"],
                depends=deps if deps is not None else ([self.steps[-1]["id"]] if self.steps else []),
                effector_only=self.device is not None and 'body' not in self.capabilities,
                dataset_replay=self.feature == 'replay',
                simulation_replay=self.simulation_replay,
                vla_control=self.feature == 'vla',
                device_context=({'registry_dir': str(self.device.registration.registry_dir),
                                 'snapshot': self.device.snapshot()} if self.device else None),
            )
        )

    def py(self, key, title, module, args=(), python="sim_python", **kw):
        self.add(key, title, [self.c[f"local.{python}"], "-B", "-u", "-m", PREFIX + module, *args], **kw)

    def endpoint(self, key):
        return f"tcp://127.0.0.1:{self.c['ports.' + key]}"

    def scene(self):
        c = self.c
        args = [
            "--interface",
            "sim",
            "--hand",
            self.hand_type,
            '--device-registry', c['local.effector_registry'],
            *(['--effector-deployment', c['local.effector_deployment']] if c['local.effector_deployment'] else []),
            *(["--hand-config", self.hand_config] if self.integration.get('hand_config') else []),
            "--enable-image-publish",
            "--enable-offscreen",
            "--camera-port",
            c["ports.camera"],
        ]
        args += self.scene_args()
        if self.feature == "vla" and c["task.vla_keyframe"]:
            args += ["--initial-keyframe", c["task.vla_keyframe"]]
        if c["task.headless"]:
            args += ["--no-enable-onscreen"]
        if self.feature == "capture" and c["task.depth"]:
            args += ["--enable-depth-publish"]
        self.py(
            "scene",
            "启动身体 / " + self.device.registration.description.label + " 仿真与相机",
            "run_sim_loop",
            args,
            service=True,
            stop_order=30,
            resources=["local:dds", "local:hand", f"local:port:{c['ports.camera']}"],
            note=(
                "run_sim_loop 启动 MuJoCo 身体与手部仿真。--hand 选择末端；"
                "--enable-image-publish 发布相机图像，--enable-offscreen 启用离屏渲染，"
                "--camera-port 指定图像端口。"
                + ("--hand-config 指定手部配置。" if self.integration.get('hand_config') else "")
                + ("--scene 加载本次选择的场景，保留设备模型与关节映射。" if self.scene_path else "")
                + ("--no-enable-onscreen 关闭仿真窗口。" if c["task.headless"] else "")
                + ("--enable-depth-publish 同时发布深度。" if self.feature == "capture" and c["task.depth"] else "")
            ),
        )

    def hand(self):
        c = self.c
        service = self.integration.get('console_hand_bridge')
        if not service:
            return
        values = dict(c)
        def flatten(prefix, data):
            for key, value in data.items():
                if isinstance(value, dict): flatten(prefix + key + '.', value)
                else: values[prefix + key] = str(value)
        flatten('device.', self.device.options)
        def expand(value):
            # Dot-containing profile/option keys are literal placeholders, not attributes.
            import re
            return re.sub(r'\{([^{}]+)\}', lambda match: values[match[1]], value)
        self.add(
            "hand",
            "G1 手桥接",
            [expand(arg) for arg in service['argv']],
            machine=service.get('machine', 'g1'),
            cwd=expand(service['cwd']),
            service=True,
            stop_order=60,
            resources=["g1:hand"],
            note=expand(service.get('note', '启动注册设备的桥接服务。')),
        )

    def camera(self):
        c = self.c
        args = [
            c["g1.camera_python"],
            "-u",
            "-m",
            "gear_sonic.camera.composed_camera",
            "--ego-view-camera",
            c["g1.camera_driver"],
            "--port",
            c["ports.camera"],
        ]
        if c["g1.camera_device"]:
            args += ["--ego-view-device-id", c["g1.camera_device"]]
        self.add(
            "camera",
            "G1 相机服务",
            args,
            machine="g1",
            cwd=c["g1.camera_root"],
            env={"PYTHONPATH": c["g1.camera_root"]},
            service=True,
            stop_order=10,
            resources=["g1:camera"],
            ready="Sensor server running",
            note="composed_camera 在 G1 采集并发布图像。--ego-view-camera 选择相机驱动，--ego-view-device-id 指定设备，--port 指定发布端口；RealSense 支持 RGB-D。",
        )

    def tunnel(self, body=False, camera=False):
        c = self.c
        ports = list(self.integration.get('tunnel_ports', [])) + (["body_state"] if body else []) + (["camera"] if camera else [])
        argv = ssh(c)[:-1] + ["-N", "-o", "ExitOnForwardFailure=yes"]
        for name in ports:
            port = c["ports." + name]
            argv += ["-L", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        if body:
            port = c["ports.body_target"]
            argv += ["-R", f"127.0.0.1:{port}:127.0.0.1:{port}"]
        argv += [f"{c['g1.ssh_user']}@{c['g1.ssh_host']}"]
        self.add(
            "tunnel",
            "SSH 数据转发",
            argv,
            service=True,
            stop_order=20,
            resources=[f"local:port:{c['ports.' + p]}" for p in ports],
            note="ssh -N 只建立转发，不启动远端命令。-L 将 G1 的手部、身体反馈或相机端口映射到电脑；遥操使用 -R 将电脑的身体目标端口映射到 G1。",
        )

    def body(self, keyboard=False):
        c = self.c
        if self.mode == "sim":
            self.add(
                "body",
                "SONIC 身体控制",
                [
                    "bash",
                    c["local.sonic_root"] + "/deploy.sh",
                    "sim",
                    "--hand",
                    self.hand_type,
                    '--device-registry', c['local.effector_registry'],
                    *(['--effector-deployment', c['local.effector_deployment']] if c['local.effector_deployment'] else []),
                    *(["--hand-python", c["local.sim_python"]] if self.integration.get('hand_python') else []),
                    "--cp",
                    c["local.vla_sonic_checkpoint"] if self.feature == 'vla' else c["local.sonic_checkpoint"],
                    "--obs-config",
                    c["local.vla_sonic_obs"] if self.feature == 'vla' else c["local.sonic_obs"],
                    *(["--sim-initial-pose", c["local.vla_initial_pose"]] if self.feature == "vla" and c["local.vla_initial_pose"] else []),
                    *self.ready_pose_args(),
                    "--planner",
                    c["local.sonic_planner"],
                    "--motion-data",
                    c["local.sonic_motion"],
                    *(["--hand-config", self.hand_config] if self.integration.get('hand_config') else []),
                    "--input-type",
                    "keyboard" if keyboard else "zmq_manager",
                    "--output-type",
                    "zmq",
                ],
                cwd=c["local.sonic_root"],
                env={"TensorRT_ROOT": c["local.tensorrt"],
                     "LD_LIBRARY_PATH": c["local.tensorrt"] + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")},
                service=True,
                ready="" if keyboard else "Init Done",
                stop_order=80,
                inputs=["\n", "y\n", "]", "O"],
                resources=["local:body"],
                note=(
                    "deploy.sh sim 启动仿真 SONIC 身体控制。--hand 选择末端，--cp 指定策略模型，"
                    "--obs-config 指定观测配置，--planner 指定行走规划模型，--motion-data 指定参考动作目录。"
                    "--input-type 选择 keyboard 键盘输入或 zmq_manager 遥操输入，--output-type zmq 发布状态。"
                ),
            )
        else:
            self.add(
                "body",
                "G1 SONIC 身体控制",
                ["bash", c["g1.body_script"]]
                + (["--keyboard"] if keyboard else [])
                + self.ready_pose_args()
                + (["--motion-data", c["g1.sonic_motion"]] if c["g1.sonic_motion"] else []),
                machine="g1",
                cwd=c["g1.body_root"],
                service=True,
                ready="" if keyboard else "Init Done",
                stop_order=80,
                inputs=["]", "\n", "O"],
                resources=["g1:body"],
                note="通过 SSH 在 G1 运行 run_body.sh，使用内部 DDS 网卡控制身体；手型由 G1 脚本指定。--keyboard 切换为键盘输入。--ready-pose 传入电脑姿态文件中的 29 个角度，开始控制后腰和双臂约 3 秒过渡到目标，腿部由 SONIC 保持平衡。--motion-data 可指定 G1 上的动作目录。",
            )

        if keyboard:
            self.steps[-1]["control_mode"] = "planner"
            self.steps[-1]["inputs"] += list("1234567890wasd, .qejlrnp-=itzfh".replace(" ", ""))
            if self.feature == "hands":
                self.steps[-1]["title"] += "（可选）"
                self.steps[-1]["note"] += "此可选步骤与独立身体控制入口使用同一控制程序，手指指令由下一步骤单独发送。"

    def xrt(self):
        c = self.c
        self.add(
            "xrt",
            "XRoboToolkit 服务",
            ["bash", c["local.xrt_program"]],
            cwd=c["local.xrt_root"],
            service=True,
            stop_order=10,
            resources=["local:xrt"],
            deps=[],
            note="bash 运行本机 XRoboToolkit PC Service 启动脚本，为 PICO 头显与本机遥操程序提供跟踪数据连接。",
        )
    def teleop(self):
        c = self.c
        self.xrt()
        args = self.device_args(backend=False) + ["--calibration", c["local.calibration"]]
        if self.ready_pose_values is not None:
            args += ["--ready-pose-values", self.ready_pose_values]
        if self.mode == "sim":
            args += ["--sim"]
        if 'body' not in self.capabilities:
            args += self.scene_args()
            if not c['task.headless']:
                args += ['--view']
        self.py(
            "teleop",
            "PICO 身体与末端遥操" if 'body' in self.capabilities else 'PICO 末端遥操',
            "run_pico_teleop",
            args,
            python="teleop_python",
            service=True,
            inputs=["\n"],
            ready="PICO ready.",
            stop_order=90,
            deps=(["body"] if 'body' in self.capabilities else []) + ["xrt"],
            resources=["local:hand_producer", "local:pico", "local:body_target"],
            note=(
                "run_pico_teleop 读取头显跟踪，通过注册的 Retargeting 转成设备动作。"
                "--device/--deployment 选择设备及后端；--sim 指定仿真。"
                + ("此设备同时发送 SONIC 身体目标；--calibration 指定光学手标定文件。"
                   if 'body' in self.capabilities else "此设备只控制独立末端仿真。")
            ),
        )

    def dataset_replay(self, *, inspect=False):
        c = self.c
        args = [c['local.input'], '--target', self.mode, '--episode', c['task.episode'],
                '--frames', c['task.replay_frames'], *self.device_args(backend=False)]
        if not self.simulation_replay:
            args += ['--body-reference', c['task.replay_mode']]
        if not inspect: args += ['--execute']
        if self.simulation_replay:
            args += ['--playback', 'simulation']
            if c['task.headless']: args += ['--headless']
        self.py('replay_check' if inspect else 'dataset_replay',
                '检查录制数据' if inspect else ('仿真场景回放' if self.simulation_replay else '身体与双手回放'), 'replay_episode', args,
                python='sim_python' if self.simulation_replay else 'data_python',
                deps=[] if inspect else (['replay_check'] if self.simulation_replay else ['body']),
                inputs=[] if inspect or self.simulation_replay else ['\n'], stop_order=100,
                resources=['local:replay_check'] if inspect else (['local:simulation_replay'] if self.simulation_replay else ['local:hand_producer', 'local:body_target']),
                outputs=[] if inspect else [c['local.root'] + '/outputs/replay_runs'],
                note=('读取录制时的完整场景及机器人、篮子状态，直接播放。无需启动身体控制，关闭窗口即可结束。'
                      if self.simulation_replay else '读取电脑上的 LeRobot episode，检查身体参考、native6 通道和 50 Hz 时间；不连接控制。'
                      if inspect else '所选身体参考经过 SONIC 控制器，双手同步播放录制动作。--body-reference 选择 smpl 人体参考或 joints 实测关节角参考；--target 选择仿真或真机；'
                      '--episode 选择条目，--frames 设为 0 播放完整条目。完成后身体保持站立；异常或手动停止会退出身体控制。'))

    def policy_server(self, *, check=False):
        c = self.c
        self.py('vla_check' if check else 'policy', '检查 VLA 模型与运行环境' if check else 'GR00T 模型服务',
                'run_policy_server', ['--model', c['local.model'], '--groot-root', c['local.groot_root'],
                 '--python', c['local.policy_python'], '--compute-device', c['task.device'],
                 '--port', c['ports.policy'], '--device', self.hand_type,
                 '--registry-dir', c['local.effector_registry'], *(['--check'] if check else [])],
                python='data_python', deps=[] if check else ['vla_check'], service=not check,
                ready='' if check else 'Server ready', stop_order=10,
                resources=['local:vla_check'] if check else [f'local:port:{c["ports.policy"]}'],
                note='检查完整权重、G1/Inspire 动作维度和 RTC 支持。模型服务在电脑加载 GR00T；此步骤不连接机器人。')

    def vla(self):
        c = self.c
        args = ['--target', self.mode, '--prompt', c['task.prompt'], '--duration', c['task.vla_seconds'],
                '--policy-port', c['ports.policy'], '--camera-port', c['ports.camera'], '--execute',
                *self.device_args(backend=False)]
        if self.ready_pose_values is not None:
            args += ['--ready-pose-values', self.ready_pose_values]
        if self.mode == 'sim' and c['local.vla_initial_pose']:
            args += ['--initial-pose', c['local.vla_initial_pose']]
        self.py('vla', 'VLA 身体与双手控制', 'run_device_vla', args, python='data_python',
                deps=['body', 'policy'], inputs=['\n'], stop_order=100,
                resources=['local:hand_producer', 'local:body_target'],
                outputs=[c['local.root']+'/outputs/vla_runs'],
                note='读取当前图像和身体/手部实测状态，异步请求 GR00T，按 50 Hz 发送 SONIC token 和原生手动作。'
                     '--duration 为执行秒数，0 持续到停止；反馈断流、动作越界或推理超时会停止本轮控制。')

    def capture(self):
        c = self.c
        from gear_sonic.utils.data_collection.paths import capture_directory
        output_root = str(capture_directory(self.mode == 'sim', project_root=c['local.root']))
        local_capture = self.device.registration.backends[self.device.backend].get('capture_source') == 'local'
        args = [
            "--output-root",
            output_root,
            "--name",
            c["task.name"],
            "--task-prompt",
            c["task.prompt"],
            "--label-after-each",
            "--seconds",
            c["task.seconds"],
            "--episodes",
            c["task.episodes"],
            "--wait-seconds",
            c["task.wait"],
            "--depth",
            "required" if c["task.depth"] and not local_capture else "off",
            *self.device_args(backend=False),
            "--camera-host",
            "127.0.0.1",
            "--camera-port",
            c["ports.camera"],
        ]
        if self.mode == "sim":
            args += ["--sim"]
        if local_capture:
            args += self.scene_args()
            args += ['--capture-input', c['task.capture_input'], '--capture-python',
                     c['local.sim_python'] if c['task.capture_input'] == 'sequence' else c['local.teleop_python']]
            if c['task.capture_input'] == 'teleop':
                self.xrt()
            if not c['task.headless']:
                args += ['--view']
        self.py(
            "capture",
            "按时长录制数据",
            "capture_session",
            args,
            python="data_python",
            outputs=[output_root + "/" + c["task.name"]],
            stop_order=100,
            inputs=["\n", "success\n", "failure\n", "unlabeled\n"],
            resources=["local:recorder", "output:" + output_root + "/" + c["task.name"]]
                      + (['local:hand_producer'] if local_capture else [])
                      + (['local:pico'] if local_capture and c['task.capture_input'] == 'teleop' else []),
            deps=(['xrt'] if c['task.capture_input'] == 'teleop' else []) if local_capture else ["teleop"],
            note=('启动独立末端仿真，执行所选测试动作或 PICO 映射并同步采集；每条结束关闭此仿真实例，保存原始轨迹和 LeRobot 数据。' if local_capture else "capture_session 只读录制相机、身体和手部数据，停止录制不停止遥操。") +
                 ' 每条保存后，在终端选择成功、失败或暂不标注，再继续下一条。--seconds 指定每条时长，--episodes 指定条数，--wait-seconds 指定等待；数据在电脑上按采集来源分入 sim/real，--output-root 与 --name 决定最终目录。',
        )

    def setup(self):
        c = self.c
        role = c["setup.role"]
        if role == "g1":
            code = "from pathlib import Path\nimport importlib.util, json\n"
            paths = [c['g1.' + key] for key in ('root', 'body_root', 'body_script', 'hand_root',
                     'hand_script', 'camera_root', 'camera_python')]
            paths += [c['g1.hand_root'] + '/bridge.env', c['g1.camera_root'] + '/gear_sonic/camera/composed_camera.py']
            code += f"paths={paths!r}\n"
            code += "result={p:Path(p).exists() for p in paths}\n"
            code += 'result["RealSense SDK"]=importlib.util.find_spec("pyrealsense2") is not None\n'
            code += 'try:\n from gear_sonic.camera.sensor_server import ImageMessageSchema\n result["RGB-D metadata"]="metadata" in ImageMessageSchema.__dataclass_fields__\nexcept ImportError:\n result["RGB-D metadata"]=False\n'
            code += "print(json.dumps(result, indent=2)); raise SystemExit(0 if all(result.values()) else 1)"
            self.add(
                "check",
                "检查 G1 个人程序与深度依赖",
                [c["g1.camera_python"], "-c", code],
                machine="g1",
                cwd=c["g1.camera_root"],
                deps=[],
                env={"PYTHONPATH": c["g1.camera_root"]},
                note="在 G1 只读检查已配置的个人部署目录、身体与手部启动文件、bridge.env、相机代码及 Python、RealSense SDK 与 RGB-D 消息支持；不启动控制程序。",
            )
            self.add(
                "camera_sdk",
                "安装 G1 RealSense Python SDK",
                [c["g1.camera_python"], "-m", "pip", "install", "pyrealsense2"],
                machine="g1",
                cwd=c["g1.camera_root"],
                deps=[],
                note="在 G1 相机 Python 环境中通过 pip 安装 pyrealsense2，用于访问 RealSense 相机。",
            )
            remote = f"{c['g1.ssh_user']}@{c['g1.ssh_host']}"
            backup = c["setup.backup_target"] + "/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            self.add(
                "camera_update",
                "备份并更新 G1 相机代码",
                [
                    "rsync",
                    "-a",
                    "--protect-args",
                    "--backup",
                    "--backup-dir=" + backup,
                    "--exclude=__pycache__",
                    "-e",
                    shlex.join(ssh(c)[:-1]),
                    c["setup.transfer_source"].rstrip("/") + "/",
                    remote + ":" + c["setup.transfer_target"].rstrip("/") + "/",
                ],
                deps=[],
                outputs=["G1:" + c["setup.transfer_target"], "G1:" + backup],
                note="rsync -a 同步相机源码并保留文件属性；--backup 与 --backup-dir 保存被替换文件，--exclude 跳过 Python 缓存；-e 指定 SSH 连接。",
            )
            return
        self.add(
            "check",
            "检查环境（只读）",
            [sys.executable, "-m", BOOTSTRAP, "check", "--config", json.dumps(c)],
            deps=[],
            note="bootstrap check 按配置检查本机路径、Python 依赖和输出目录权限，只报告结果，不安装软件。",
        )
        self.add(
            'directories', '初始化本机输出目录',
            [sys.executable, '-m', BOOTSTRAP, 'directories', '--config', json.dumps(c)],
            deps=[],
            note='创建缺少的仿真/真机数据集、日志、标定和回放记录目录；保留已有文件，不连接 G1。',
        )
        packages = [
            "build-essential",
            "cmake",
            "git",
            "git-lfs",
            "rsync",
            "ffmpeg",
            "python3-venv",
            "libgl1",
            "libegl1",
            "libzmq3-dev",
            "cppzmq-dev",
            "libmsgpack-dev",
            "libusb-1.0-0-dev",
            "espeak",
        ]
        install_system = (
            "task_missing=(); for task_package in " + shlex.join(packages) + "; do "
            "if [[ $(dpkg-query -W -f='${Status}' \"$task_package\" 2>/dev/null) != 'install ok installed' ]]; "
            'then task_missing+=("$task_package"); fi; done; '
            "if ((${#task_missing[@]})); then sudo -n apt-get update "
            '&& sudo -n apt-get install -y "${task_missing[@]}"; '
            "else echo '系统依赖已安装，跳过'; fi"
        )
        self.add(
            "system_packages",
            "补齐 Ubuntu / Debian 系统依赖",
            ["bash", "-c", install_system],
            deps=[],
            note="dpkg-query 检查系统依赖，apt-get 只安装缺少的软件包；sudo -n 使用已有免交互权限。此命令不安装 GPU 驱动、TensorRT 或 XRoboToolkit。",
        )
        python = c[
            "local."
            + {
                "simulation": "sim_python",
                "teleop": "teleop_python",
                "capture": "data_python",
                "inference": "policy_python",
            }[role]
        ]
        self.add(
            "venv",
            "创建缺少的 Python 环境",
            [sys.executable, "-m", BOOTSTRAP, "venv", "--python", python, "--base-python", c["local.base_python"]],
            deps=[],
            outputs=[str(Path(python).parent.parent)],
            note="bootstrap venv 用 --base-python 创建 --python 所属虚拟环境；目标环境已存在时保留。",
        )
        self.add(
            "install",
            "安装角色依赖",
            [
                sys.executable,
                "-m",
                BOOTSTRAP,
                "install",
                "--role",
                role,
                "--python",
                python,
                "--root",
                c["local.root"],
                "--groot-root",
                c["local.groot_root"],
            ],
            deps=["venv"],
            note="bootstrap install 根据 --role 将依赖安装到 --python 指定的环境；--root 和 --groot-root 指定源码目录，下方列出实际安装命令。",
        )
        from .bootstrap import install_commands

        self.steps[-1]["nested_commands"] = [
            shlex.join(argv) for argv in install_commands(role, python, c["local.root"], c["local.groot_root"])
        ]
        self.add(
            "verify",
            "重新检查环境",
            [sys.executable, "-m", BOOTSTRAP, "check", "--config", json.dumps(c)],
            deps=["install"],
            note="再次运行 bootstrap check，检查安装后的路径与依赖是否可用，只报告结果。",
        )

    def build(self):
        info = next((f for f in FEATURES if f["id"] == self.feature), None)
        if info is None or self.mode not in info["modes"]:
            raise ValueError("功能或模式无效")
        c = self.c
        required = {'hands':'control', 'body':'body', 'scene':'scene', 'teleop':'teleop', 'capture':'capture', 'replay':'dataset_replay', 'vla':'inference'}.get(self.feature)
        if required and required not in self.capabilities:
            raise ValueError(f'{self.device.registration.description.label} 不支持当前模式的 {info["title"]}')
        if self.feature == "setup":
            self.setup()
        elif self.feature == 'vla':
            self.policy_server(check=True)
            self.policy_server()
            if self.mode == 'sim':
                self.scene()
            else:
                self.hand()
                self.camera()
                self.tunnel(body=True, camera=True)
            self.body()
            self.vla()
        elif self.feature == 'replay':
            self.dataset_replay(inspect=True)
            if self.simulation_replay:
                self.dataset_replay()
                return self.result(info)
            if self.mode == 'sim':
                self.scene()
            else:
                self.hand()
                self.tunnel(body=True)
            self.body()
            self.dataset_replay()
        elif 'body' not in self.capabilities:
            if self.feature == 'hands':
                self.control()
            elif self.feature == 'scene':
                self.py('scene', '末端执行器仿真', 'run_end_effector',
                        self.device_args()+self.scene_args()+['--operation','feedback','--seconds','1800']+([] if c['task.headless'] else ['--view']),
                        service=True, resources=['local:hand_producer'], deps=[],
                        note='打开注册的独立 MuJoCo 模型并读取实际反馈，不启动 SONIC 身体控制。')
            elif self.feature == 'teleop':
                self.teleop()
            elif self.feature == 'capture':
                self.capture()
        elif self.feature == "scene":
            self.scene()
        elif self.feature == "body":
            if self.mode == "sim":
                self.scene()
            self.body(keyboard=True)
        else:
            if (
                self.feature == "capture"
                and self.mode == "real"
                and c["task.depth"]
                and c["g1.camera_driver"] != "realsense"
            ):
                raise ValueError("开启深度需选择 realsense 相机驱动")
            if self.mode == "sim":
                self.scene()
            else:
                self.hand()
                if self.feature == "capture":
                    self.camera()
                self.tunnel(body=self.feature != "hands", camera=self.feature == "capture")
            if self.feature == "hands":
                if c["task.with_body"]:
                    self.body(keyboard=True)
                self.control()
            else:
                self.body()
                self.teleop()
                if self.feature == "capture":
                    self.capture()
        return self.result(info)

    def control(self):
        c = self.c
        operation = 'target'
        title = '执行指定角度'
        if c['task.pose'] == 'sequence':
            actions = self.integration.get('hand_actions', {})
            preset = actions.get(c['task.hand_action'])
            if not preset:
                raise ValueError('请选择当前设备声明的测试动作')
            title = preset['label']
            operation = preset.get('operation', 'target')
            targets = {mount: preset['target'] for mount in self.device.mounts} if 'target' in preset else None
        else:
            targets = effector_targets(c, self.device)
        module = 'run_end_effector'
        args = self.device_args() + ['--operation', operation, '--seconds', c['task.hand_seconds']]
        if operation == 'target':
            args += ['--targets', json.dumps(targets)]
        sequence = self.integration.get('sequence_cli')
        if operation == 'sequence' and sequence:
            module = sequence['module']
            values = {'backend': sequence['backend_names'][self.device.backend],
                      'hand_config': str(self.hand_config), 'seconds': str(c['task.hand_seconds'])}
            args = [part.format_map(values) for part in sequence['arguments']]
        elif 'body' not in self.capabilities:
            args += self.scene_args()
            if not c['task.headless']:
                args += ['--view']
        self.py('send_hand', title, module, args, stop_order=100, resources=['local:hand_producer'],
                deps=[step['id'] for step in self.steps if step['id'] in ('scene', 'hand', 'tunnel')],
                note=('逐个手指弯曲再张开：先左手再右手，每只手从拇指到小指。动作时间设置每阶段的最短时间。'
                      if operation == 'sequence' and sequence else
                      '按当前设备的通道、单位和限位发送目标，从实测位置起步并限制速度；运行指定时长后结束。'))

    def result(self, info):
        from gear_sonic.utils.data_collection.paths import output_directories
        value = dict(
            feature=self.feature,
            mode=self.mode,
            title=info["title"],
            pending=False,
            config=self.c,
            steps=self.steps,
            log_root=str(Path(self.c["local.root"]) / "outputs/console_logs"),
            output_directories=output_directories(self.c['local.root']),
        )
        value["id"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]
        return value


def build_plan(config, feature, mode):
    return Builder(config, feature, mode).build()
