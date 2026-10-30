"use strict";
// Configuration ownership is explicit; unrelated fields are never appended to a function form.
const ConsoleSettings = {
  environments: [
    {role:'simulation',label:'仿真环境',groups:['sim','sonic','scenes']},
    {role:'teleop',label:'PICO 环境',groups:['pico']},
    {role:'capture',label:'采集／回放环境',groups:['recording']},
    {role:'inference',label:'VLA 环境',groups:['policy']},
    {role:'g1',label:'G1 文件与依赖',groups:['g1','real_camera']},
  ],
  downloads: {
    base: [
      {title:"需要安装：uv 和 Python 3.10", text:"本机联网执行一次。安装后，将最后一行输出的 Python 路径填入「创建新环境使用的 Python」，再在本页「安装与检查」创建仿真、遥操和采集环境。",
       links:[["uv 官方安装说明","https://docs.astral.sh/uv/getting-started/installation/"]],
       commands:[{label:"安装 uv / Python",lines:['curl -LsSf https://astral.sh/uv/install.sh | sh','source "$HOME/.local/bin/env"','uv python install 3.10','uv python find --no-project 3.10']}],
       location:"uv 和 Python 安装在当前用户的工具目录；三个运行环境在本页创建到项目的 .venv_sim、.venv_teleop、.venv_data_collection。"},
      {title:"项目文件：使用实验室提供的版本", text:"需要包含 8765 控制台、Inspire 和 RTC 补丁的这份项目。Git 获取的项目还需拉取 LFS 文件；使用压缩包时，请向提供者确认机器人模型和网格文件齐全。",
       commands:[{label:"Git 版本补齐大文件（项目根目录）",lines:['git lfs install','git lfs pull']}],
       location:"手模型、场景和准备姿态随项目提供，保留项目目录结构。"}
    ],
    sim: [{title:"需要联网安装：MuJoCo 与仿真依赖",text:"在本页「安装与检查」选择「仿真环境」，执行系统依赖、创建环境、安装角色依赖、重新检查。程序会下载 Python 包，不需要另找 MuJoCo 安装包。",location:".venv_sim/bin/python → 填入「仿真 Python」。"}],
    recording: [{title:"需要联网安装：采集／回放依赖",text:"在本页「安装与检查」选择「采集／回放环境」，创建环境并安装角色依赖。该环境同时供 04 采集、05 回放和 06 VLA 控制端使用。",location:".venv_data_collection/bin/python → 填入「采集／回放 Python」。"}],
    scenes: [{title:"随项目提供：仿真场景",text:"默认机器人场景和 table_box、basket_chair 已在这份项目中。自定义场景连同引用的模型、网格一起复制。",location:"scenes/inspire/<场景名>/scene.xml；公共配置的「场景目录」通常保持 scenes。"}],
    hands: [{title:"随项目提供：Inspire 手配置与模型",text:"无需单独下载手模型。真机串口参数在 G1 的 hand_bridge/bridge.env 中，由实验室部署提供。",location:"gear_sonic/devices/inspire/assets/hand_config.yaml → 填入「本机 Inspire 手配置」。"}],
    sonic: [
      {title:"需要下载：SONIC release 权重与 planner",text:"先安装上面的 uv。在本项目根目录运行；脚本按当前部署需要的目录结构保存 encoder、decoder、观测配置和 planner。",
       links:[["NVIDIA GEAR-SONIC 模型","https://huggingface.co/nvidia/GEAR-SONIC"]],
       commands:[{label:"下载 SONIC 身体模型",lines:['uv run --no-project --with huggingface_hub python download_from_hf.py']}],
       location:"模型前缀：gear_sonic_deploy/policy/release/model；观测配置：同目录 observation_config.yaml；planner：gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx。"},
      {title:"需要下载：桌面电脑 TensorRT",text:"打开 NVIDIA 下载页，选择 Linux x86_64 的 TensorRT 10.13.x TAR 包，并匹配电脑 CUDA 版本；当前桌面验证版本为 10.13.3。下载后解压到自己选择的工具目录。GPU 驱动和 CUDA 按 NVIDIA 对应版本说明安装。",
       links:[["TensorRT 10.x 下载","https://developer.nvidia.com/tensorrt/download/10x"],["CUDA 历史版本","https://developer.nvidia.com/cuda-toolkit-archive"]],
       location:"将解压后同时包含 include/、lib/、bin/ 的目录填入「TensorRT 目录」，例如 $HOME/tools/TensorRT-10.13.3.9。此项配置电脑；实验室 G1 的 TensorRT 沿用其部署环境。"}
    ],
    pico: [
      {title:"需要下载：XRoboToolkit PC Service",text:"电脑端服务和 Python SDK 是两项依赖。先按 Ubuntu 版本选择下面一组命令安装 PC Service，再在本页「PICO 环境」安装 Python SDK。",
       links:[["PC Service v1.0.0 下载页","https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0"]],
       commands:[
         {label:"Ubuntu 22.04 / x86_64",lines:['mkdir -p "$HOME/Downloads"','curl -fL https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb -o "$HOME/Downloads/XRoboToolkit_PC_Service_ubuntu_22.04_amd64.deb"','sudo apt install "$HOME/Downloads/XRoboToolkit_PC_Service_ubuntu_22.04_amd64.deb"']},
         {label:"Ubuntu 24.04 / x86_64",lines:['mkdir -p "$HOME/Downloads"','curl -fL https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb -o "$HOME/Downloads/XRoboToolkit_PC_Service_ubuntu_24.04_amd64.deb"','sudo apt install "$HOME/Downloads/XRoboToolkit_PC_Service_ubuntu_24.04_amd64.deb"']}
       ],
       location:"安装后确认 /opt/apps/roboticsservice/run3D.sh 存在。「XRoboToolkit 目录」填 /opt/apps/roboticsservice，「启动脚本」填该 run3D.sh；遥操 Python 填 .venv_teleop/bin/python。"},
      {title:"需要安装到头显：PICO 客户端",text:"在 PICO 浏览器打开发布页，下载 XRoboToolkit-PICO-1.1.1.apk 并安装，按项目 PICO 说明开启开发者模式、身体和手部跟踪。它装在头显，不填写电脑目录。",
       links:[["PICO 客户端 v1.1.1","https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1"]],
       location:"光学手标定不从网上下载；首次遥操按提示生成 outputs/inspire_calibrations/pico_hands.json。"}
    ],
    policy: [
      {title:"需要获取：兼容本项目的 Isaac-GR00T 源码",text:"当前 VLA 需要 RTC 动作历史接口。下面在新目录克隆已核对的官方版本，再应用随本项目提供的实验室补丁；已有实验室兼容版本可直接填写路径。命令从本项目根目录开始，groot_source_dir 可改为自己的安装位置。",
       links:[["Isaac-GR00T 官方源码","https://github.com/NVIDIA/Isaac-GR00T"],["对应官方版本","https://github.com/NVIDIA/Isaac-GR00T/commit/65cc4a192e6d084650d97747308b6a8deb790722"]],
       commands:[{label:"新建兼容的 Isaac-GR00T 环境（x86_64 / Python 3.10）",lines:[
         '(',
         'set -e',
         'sonic_project_dir="$PWD"',
         'groot_source_dir="$HOME/projects/Isaac-GR00T"',
         'test ! -e "$groot_source_dir"',
         'mkdir -p "$HOME/projects"',
         'GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "$groot_source_dir"',
         'git -C "$groot_source_dir" checkout --detach 65cc4a192e6d084650d97747308b6a8deb790722',
         'git -C "$groot_source_dir" submodule update --init --recursive',
         'git -C "$groot_source_dir" apply --check "$sonic_project_dir/external_dependencies/patches/isaac-groot-rtc.patch"',
         'git -C "$groot_source_dir" apply "$sonic_project_dir/external_dependencies/patches/isaac-groot-rtc.patch"',
         'uv --directory "$groot_source_dir" sync --python 3.10',
         ')'
       ]}],
       location:"「Isaac-GR00T 目录」填该源码目录；「PolicyServer Python」填该目录的 .venv/bin/python。依赖安装按官方 dGPU 指南准备 CUDA / FFmpeg，再在本页检查环境，使用 06 检查模型。"},
      {title:"需要下载：VLA 配套的 SONIC low_latency",text:"在本项目根目录执行。它与普通身体控制的 release 分开保存。",
       links:[["low_latency 模型","https://huggingface.co/nvidia/GEAR-SONIC/tree/main/low_latency"]],
       commands:[{label:"下载 low_latency",lines:['uv run --no-project --with huggingface_hub python download_from_hf.py --low-latency']}],
       location:"「VLA 仿真 SONIC 模型前缀」填 gear_sonic_deploy/policy/low_latency/model；观测配置填同目录 observation_config.yaml。"},
      {title:"按需下载：原始 GR00T N1.7 模型",text:"这是预训练起点。06 执行实验室任务时应选择已针对 G1 + Inspire 微调的 checkpoint。先安装 uv，再从本项目根目录运行。",
       links:[["NVIDIA GR00T-N1.7-3B","https://huggingface.co/nvidia/GR00T-N1.7-3B"],["Hugging Face 下载说明","https://huggingface.co/docs/huggingface_hub/guides/cli"]],
       commands:[{label:"下载原始 N1.7",lines:['uvx --from huggingface_hub hf download nvidia/GR00T-N1.7-3B --local-dir outputs/checkpoints/pretrained/GR00T-N1.7-3B']}],
       location:"outputs/checkpoints/pretrained/GR00T-N1.7-3B/。若访问提示需要登录或许可，先在模型页面按提示操作，再用 uvx --from huggingface_hub hf auth login 登录。"},
      {title:"向实验室获取：任务 checkpoint",text:"实验室微调模型没有在这里提供公开下载地址。取得完整 checkpoint 文件夹或压缩包，保留 config、processor、统计信息、权重分片及索引等配套文件。",
       location:"放入 outputs/checkpoints/finetuned/<模型名>/；「GR00T 模型目录」选择这个具体 checkpoint 目录，例如 outputs/checkpoints/finetuned/your-checkpoint。"}
    ],
    g1: [{title:"实验室已有部署：确认 G1 文件位置",text:"新电脑连接现有 G1 时，向实验室确认 SSH 和部署路径即可。若机器人尚未部署，需要这份项目对应的身体、手桥接和相机程序；请使用实验室的部署包。本页「G1 文件与依赖」中的相机传输步骤只更新相机代码。",location:"G1 上默认 /home/unitree/inspire_hand_bridge/，其中 sonic/run_body.sh、hand_bridge/run.sh、hand_bridge/bridge.env 分别对应身体、手桥接入口与手配置。"}],
    real_camera: [{title:"按需安装到 G1：RealSense Python SDK",text:"相机程序随实验室部署提供。缺少 pyrealsense2 时，在本页「安装与检查」选择「G1 文件与依赖」，使用对应 SDK 安装步骤；安装位置是 G1 的相机 Python 环境。",location:"G1 的 camera_server/.venv/bin/python；电脑上的仿真相机由 MuJoCo 提供。"}]
  },
  groups: [
    {machine:"local", id:"base", title:"项目目录", scope:"所有功能都要确认", required:true,
      description:"默认 . 自动使用当前代码所在的项目根目录。项目内的路径统一相对于此目录，执行日志保存在 outputs/console_logs。",
      fields:["local.root","local.base_python"]},
    {machine:"local", id:"storage", title:"数据与输出目录", scope:"仿真与真机数据都保存在电脑",
      description:"所有输出固定保存在项目 outputs 下；数据集按采集来源分入 datasets/sim 或 datasets/real。",
      fields:[]},
    {machine:"local", id:"sim", title:"仿真运行环境", scope:"仅做仿真时需要",
      description:"用于启动 MuJoCo 和仿真手指程序。只使用真机 PICO 时无需配置。",
      fields:["local.sim_python"]},
    {machine:"local", id:"scenes", title:"仿真场景目录", scope:"所有仿真任务共用",
      description:"相对路径以项目目录为基准；也可填写绝对路径。",
      fields:["local.scene_root"]},
    {machine:"local", id:"hands", title:"手部配置", scope:"手指 / 场景 / PICO / 采集共用",
      description:"手配置 YAML 指定关节、手部端点和仿真场景。真机转发端口须与 YAML 和 G1 bridge.env 一致。",
      fields:["local.hand_config","ports.hand_command","ports.hand_state"]},
    {machine:"local", id:"sonic", title:"仿真身体控制", scope:"仿真中需要站立或遥操时填写",
      description:"PICO 仿真、仿真采集，以及勾选身体控制的手指功能使用。仅打开场景不需要这些模型。",
      fields:["local.sonic_root","local.tensorrt","local.sonic_checkpoint","local.sonic_obs","local.sonic_planner","local.sonic_motion"]},
    {machine:"local", id:"pico", title:"PICO 与光学手", scope:"PICO 遥操 / 数据采集需要",
      description:"仿真和真机共用本机头显服务、遥操解释器和手部标定。只做普通手指测试无需配置。",
      fields:["local.teleop_python","local.xrt_root","local.xrt_program","local.calibration","ports.body_target","ports.body_state"]},
    {machine:"g1", id:"g1", title:"G1 连接与身体、手部程序", scope:"01–06 真机功能共用；首次使用必须确认", required:true,
      description:"填写实验室 G1 的 SSH 连接和个人部署目录。sonic/run_body.sh 启动身体，hand_bridge/run.sh 启动手桥接；手串口与速度在机器人 hand_bridge/bridge.env 中设置。修改部署目录会同步调整仍在该目录内的启动路径。",
      fields:["g1.ssh_host","g1.ssh_user","g1.ssh_port","g1.root","g1.hand_root","g1.hand_script","g1.body_root","g1.body_script"]},
    {machine:"local", id:"recording", title:"采集与回放运行环境", scope:"数据采集 / 录制数据回放 / VLA 控制",
      description:"电脑上读取和保存 LeRobot 数据的 Python 环境；录制和回放各自的参数在功能内设置。",
      fields:["local.data_python"]},
    {machine:"local", id:"camera", title:"相机连接", scope:"仿真相机 / 数据采集共用",
      description:"相机发布和录制接收使用同一端口。仿真相机由场景程序启动。",
      fields:["ports.camera"]},
    {machine:"g1", id:"real_camera", title:"G1 相机", scope:"04 真机采集 / 06 真机 VLA 需要",
      description:"配置机器人上含 gear_sonic/camera 的项目目录、相机 Python 和设备。RealSense 使用序列号；仅一台可留空。相机程序需已部署到 G1，依赖和文件检查在本页「G1 文件与依赖」中完成。",
      fields:["g1.camera_root","g1.camera_python","g1.camera_driver","g1.camera_device"]},
    {machine:"local", id:"policy", title:"VLA 运行环境", scope:"Inspire 仿真 / 真机推理",
      description:"Isaac-GR00T 属于外部项目，按本机安装位置配置；checkpoint 使用 outputs/checkpoints 下的项目相对路径。",
      fields:["local.groot_root","local.policy_python","local.model","ports.policy","local.vla_sonic_checkpoint","local.vla_sonic_obs"]},
  ],
  choiceLabels: {
    "task.pose": {targets:"指定角度", sequence:"测试动作"},
    "task.capture_input": {sequence:"自动测试动作",teleop:"PICO 遥操"},
    "task.outcome": {unlabeled:"未标注",success:"成功",failure:"失败"},
    "task.replay_mode": {smpl:"SMPL 人体参考", joints:"实测关节角参考"},
    "setup.role": {simulation:"仿真电脑",teleop:"PICO 遥操电脑",capture:"数据采集电脑",inference:"VLA 推理电脑",g1:"G1 机器人"},
    "g1.camera_driver": {realsense:"RealSense（支持深度）",usb:"USB 相机（RGB）"}
  },
  fieldLabels: {
    "task.body_hand":"末端执行器", "task.real_hand":"末端执行器", "task.pose":"操作", "task.with_body":"同时启动身体控制（可选）",
    "task.seconds":"每条录制时长（秒）", "task.wait":"每条开始前等待（秒）",
    "task.episodes":"录制条数", "task.depth":"保存深度数据", "task.hand_seconds":"动作时长（秒）",
    "task.left_target":"左手目标角度（rad）", "task.right_target":"右手目标角度（rad）",
    "task.name":"数据集名称", "task.headless":"不显示仿真窗口",
    "local.sonic_motion":"电脑动作目录", "g1.sonic_motion":"G1 动作目录",
    "local.input":"电脑上的数据集目录", "task.episode":"Episode 编号", "task.replay_frames":"播放帧数（0 = 完整条目）",
    "local.calibration":"光学手标定文件", "local.base_python":"创建新环境使用的 Python"
  },
  hints: {
    "local.ready_pose":"仿真和真机共用，真机自动传入 29 维身体角度；留空沿用原版 SONIC 流程。项目相对路径示例：gear_sonic/data/poses/g1/box_ready.txt。",
    "local.base_python":"本页创建新环境时使用的 Python。项目安装脚本使用 Python 3.10；已有环境可直接填写对应的 bin/python。",
    "local.sim_python":"保留项目默认位置；未安装时，在本页「安装与检查」选择「仿真环境」创建环境并安装依赖。",
    "local.data_python":"04 采集、05 回放、06 VLA 控制端共用；未安装时，在本页「安装与检查」选择「采集／回放环境」。",
    "local.teleop_python":"03 PICO 遥操和 04 采集需要；未安装时，在本页「安装与检查」选择「PICO 环境」。",
    "local.policy_python":"Isaac-GR00T 的推理环境，填写该环境的 bin/python；与本项目的采集环境分别配置。",
    "local.calibration":"个人光学手标定文件；没有时，首次 PICO 遥操会提示完成标定并保存到这里。",
    "g1.root":"这是机器人上的路径。默认 /home/unitree/inspire_hand_bridge；请向实验室确认实际位置。",
    "g1.body_script":"G1 上实际存在的身体启动脚本，通常是个人部署目录下 sonic/run_body.sh。",
    "g1.hand_script":"G1 上实际存在的手桥接脚本，通常是个人部署目录下 hand_bridge/run.sh。",
    "local.tensorrt":"外部依赖，填写这台电脑实际安装的 TensorRT 目录。",
    "local.xrt_root":"外部依赖，填写这台电脑实际安装的 XRoboToolkit 服务目录。",
    "local.groot_root":"外部项目，填写这台电脑的 Isaac-GR00T 源码目录。",
    "local.model":"微调模型放在 outputs/checkpoints/finetuned/，原始模型放在 outputs/checkpoints/pretrained/；这里填写含 config.json 和模型权重的具体目录。",
    "g1.ssh_host":"填 G1 上 hostname -I / ip -4 -br addr show up 查到、电脑可访问的 IPv4；此处是机器人 IP。PICO 的 PC Service 填运行 XRoboToolkit 服务的电脑 IP。",
    "local.scene_root":"默认 scenes 是相对项目根目录的路径，也可填写绝对路径。目录内按 设备名/场景名/scene.xml 存放。",
    "local.sonic_motion":"默认使用 SONIC 自带动作；更换目录后重新启动身体程序。",
    "g1.sonic_motion":"填写 G1 上的动作目录绝对路径；留空使用 sonic/reference/example/。更换后重新启动身体程序。",
    "local.input":"选择含 meta/info.json 的录制目录；可来自 datasets/sim 或 datasets/real，与播放目标无关。",
    "task.episode":"从 0 开始；例如数据集有 3 条，就填写 0、1 或 2。",
    "task.replay_frames":"固定 50 Hz。可先填写 150，播放前 3 秒。",
    "task.replay_mode":"两种方式都经过 SONIC 控制身体，并同步回放双手。实测关节角方式需要新录制保存的 sonic_reference 文件。",
    "task.headless":"需要在 MuJoCo 窗口按 9 释放支撑时，保持此项关闭。",
    "task.with_body":"也可先在「身体控制」启动；已启动时关闭此项。",
    "task.episodes":"0 表示持续采集，直到手动停止。", "task.depth":"需要 RGB-D 输入；缺少深度时会报错。",
    "task.left_target":"6 个数，空格分隔：小指、无名指、中指、食指、拇指弯曲、拇指旋转。",
    "task.right_target":"顺序与左手相同。0 表示回零目标，不表示保持当前角度。",
    "local.vla_initial_pose":"仅仿真使用的 29 关节起始姿态文件；留空使用 SONIC 默认站立。应与 checkpoint 的采集设置一致。",
    "task.vla_keyframe":"必须是当前场景已有的预设名。basket_chair 支持 photo_arms_down；table_box 没有此预设，应留空。使用默认站立时，上面的姿态文件也留空。",
    "local.hand_config":"手部通道和连接参数；仿真场景在各功能内选择。",
    "g1.camera_device":"RealSense 填序列号，不填 USB 设备编号；只有一台时可留空。",
  },
  locked: ["task.fps","ports.body_target","ports.body_state"],
  sections(feature, mode, config, device) {
    const section = (title, keys, note="") => ({title,keys,note});
    const hasBody=device?.deployments[mode]?.capabilities.includes("body");
    const selector=section("设备",[mode==="real"?"task.real_hand":"task.body_hand"]);
    const display=mode==="sim"?[section("仿真场景",[...(config["task.sim_scenes"]!==undefined?["@scene"]:[]),"task.headless"])]:[];
    switch (feature) {
      case "body": return [selector,...display];
      case "hands": return [selector,...display,section("选择操作",["task.pose",...(hasBody?["task.with_body"]:[])]),
        ...(config["task.pose"] === "targets" ? [section("指定角度",["@targets"])] : (config["task.hand_action"]!==undefined?[section("测试动作",["@hand_action"])]:[])),
        section("动作时间",["task.hand_seconds"])];
      case "scene": return [selector,...display];
      case "teleop": return [selector,...display];
      case "capture": return [selector,...display,section("录制内容",["task.prompt",...(hasBody?["task.depth"]:["task.capture_input"])],"每条录制保存后，在运行终端选择成功、失败或暂不标注，再继续下一条。"),section("时间与条数",["task.seconds","task.episodes","task.wait"]),
        section("保存位置",["task.name"])];
      case "replay": return [section("录制数据",["@dataset","task.episode","task.replay_mode","task.replay_frames"]),selector,...display];
      case "vla": return [selector,...display,section("推理参数",["local.model","task.prompt","task.vla_seconds","task.device"]),...(mode==="sim"?[section("仿真初始姿态（可选）",["local.vla_initial_pose","task.vla_keyframe"])]:[])];
      case "setup": return [section("初始化对象",["setup.role"]),config["setup.role"] === "g1"
        ? section("本次相机代码传输",["setup.transfer_source","setup.transfer_target","setup.backup_target"])
        : section("创建环境",["local.base_python"],"目标环境的 Python 路径在公共配置中设置。已有环境会保留。"),
        ...(config["setup.role"] === "inference" ? [section("待检查的模型",["local.model"])] : [])];
      default: return [];
    }
  }
};
