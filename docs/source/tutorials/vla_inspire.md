# Inspire VLA 推理

入口是网页 **06 VLA 推理**，当前支持 G1 + Inspire；仿真和真机共用电脑上的模型服务与控制循环。只迁入推理，不包含训练、RL 或重建。

## 网页操作

1. 在「公共配置 → 本地电脑」确认采集／回放 Python 为本项目 `.venv_data_collection/bin/python`。它运行 VLA 控制程序；模型服务另用已有的 GR00T Python 环境。
2. 在「VLA 运行环境」填写 Isaac-GR00T 目录、PolicyServer Python 和端口。模型环境需要公共配置安装说明中的 RTC 补丁；仅有普通 `get_action` 接口不够。
3. 进入 **06 VLA 推理**，选择仿真或真机，填写 checkpoint、任务描述和时长。时长 `0` 表示运行到手动停止；仿真场景也在这里选择。
4. 逐步执行模型检查、模型服务和对应控制服务。模型服务等待 `Server ready`。仿真 SONIC 询问部署时，在右侧身体终端点击「确认部署 y↵」；看到 `Init Done` 后执行 VLA 步骤，身体终端不按 `]` 或 Enter。
5. VLA 终端显示 `VLA_READY` 后，点击「启动站立（Enter）」。显示 `VLA_STANDING` 后，确认稳定；仿真在 MuJoCo 窗口按 **9** 释放支撑，真机按现场方式让双脚承重并保留吊架保护。再点击 VLA 终端的「开始推理（Enter）」。
6. 右侧显示执行帧数和时间。达到设定时长后回到站立；点击「停止 Ctrl+C」或发生错误时，程序请求停止身体、释放手部，再退出。身体退出后，重新推理前需重新启动身体步骤。

真机步骤复用已有 **手桥接、相机、SSH 转发、G1 身体程序**。模型与控制循环仍在电脑，地址保持 `127.0.0.1`；不需要把 GR00T 模型搬到 G1。PICO 不参与 VLA 推理。

### 环境配置示例

| 项目 | 路径 |
|---|---|
| GR00T 项目 | `/absolute/path/to/Isaac-GR00T` |
| 模型 Python | 上述目录的 `.venv/bin/python` |
| 任务 checkpoint | `outputs/checkpoints/finetuned/your-checkpoint` |
| 仿真 SONIC 模型前缀 | 本项目 `gear_sonic_deploy/policy/low_latency/model` |
| SONIC 观测配置 | 本项目 `gear_sonic_deploy/policy/low_latency/observation_config.yaml` |

准备姿态现在可独立于场景设置：在「准备姿态（可选）」选择「盒子任务准备姿态」或自定义身体29角度文件；选择后清除旧 VLA 姿态和 keyframe 设置，仿真及真机共用此配置。INIT 沿用原版初始化姿态；第一次 Enter 启动 SONIC 控制后，腰和双臂才开始跟随准备目标，腿部由策略维持平衡。第二次 Enter 后，腰和双臂误差需在 **0.15 rad 内连续保持 0.5 秒**才开始执行。真机需要部署匹配版本的启动脚本和身体程序。文件格式见 `gear_sonic/data/poses/g1/README.md`。

旧仿真配置仍兼容：`local.vla_initial_pose` 只用于仿真；`photo_arms_down` keyframe 只存在于 `basket_chair` 场景。旧配置不能和新的准备姿态同时使用。

模型、场景、任务描述和起始状态应与训练数据匹配。模型维度检查通过不等于会完成抓取。

## 命令行入口

以下在本项目根目录执行；网页生成并执行的也是这些入口。

只检查模型及 RTC 运行环境，不启动 GPU 或机器人：

```bash
.venv_data_collection/bin/python -m gear_sonic.scripts.run_policy_server \
  --model outputs/checkpoints/finetuned/your-checkpoint \
  --groot-root /absolute/path/to/Isaac-GR00T \
  --python /absolute/path/to/Isaac-GR00T/.venv/bin/python \
  --check
```

去掉 `--check` 启动模型服务。先按网页启动仿真／真机控制服务，再运行：

```bash
.venv_data_collection/bin/python -m gear_sonic.scripts.run_device_vla \
  --target sim --device inspire \
  --prompt "While standing in place, grasp the box with both hands and lift it." \
  --duration 30 --execute
```

真机改为 `--target real --hand-config gear_sonic/devices/inspire/assets/hand_config.yaml`。不加 `--execute` 只显示配置，不连接控制；加 `--shadow --execute` 读取现有状态、验证一次模型回复，不发送启动或动作，要求对应状态与相机服务已运行。`--auto-start` 仅供本地仿真测试，真机拒绝该选项。

## 实现与边界

```text
当前 RGB + 身体实测 29 维 + 左手 6 维 + 右手 6 维
    → 注册设备的观测适配 → GR00T / RTC 异步推理
    → 40 帧动作块 → 50 Hz 调度
        ├── token64 → 共用 SONIC Session → 身体控制
        └── 左右 native6 → 现有 Inspire Adapter → 仿真或原手驱动
```

- `devices/inspire/policy.py` 管理设备动作维度、观测构造和 checkpoint 检查；`scheduling.py` 迁自旧 `live_vla.py`；`inference.py` 连接现有工厂和控制循环。
- 共用 `utils/inference/policy_client.py` 传输模型请求，`sonic_session.py` 由原录制回放的身体会话提取，回放与 VLA 共用。未另建手部驱动或搬入旧 1600 多行部署 CLI。
- 保留旧 RTC 参数：40 帧、通常重叠 20 帧、冻结 5 帧；必须收到 `rtc_applied=true`。运行中根据实际延迟丢弃过期帧，整个动作块过期则停止。首次静止预热单独处理。
- 身体 token 绝对值不超过 1，相邻帧变化不超过 0.25；状态超时 0.25 秒、动作块耗尽最多保持 0.1 秒、调度迟到上限 0.020 秒。手部继续使用原限位、限速、碰撞约束和会话检查；按实际发送间隔计算斜坡，不改驱动门限。
- 正常结束回到 planner 站立；异常先请求身体 STOP 和手部释放，再等待推理线程退出。CLI 不写任何 G1 文件。
- 真机接入使用公共配置中的 SONIC/SSH 路径，**没有迁入旧项目另一套原生 Body Gateway**。现有 G1 启动脚本使用 low_latency；远端程序版本和实际任务效果需要在部署后确认。

必要的原入口修改仅包括注册能力、控制台入口、回放共享会话，以及 SONIC 可选 `--sim-initial-pose` 的参数传递和 INIT 姿态、仿真 `--initial-keyframe` 的启动／reset 钩子；未指定时保持原默认行为。仿真模型与手驱动未改。初始 keyframe 必须存在，否则启动失败，不会默默回退。

每轮报告保存到 `outputs/vla_runs/*.json`，包含设备版本／配置快照、目标环境、动作帧数、频率、限速情况和失败原因。网页命令及模型加载日志在 `outputs/console_logs/`。历史测试记录、截图和临时仿真数据已于 2026-10-27 清理；未把测试轨迹标为抓取成功。
