# Inspire 手部功能

这里是 Inspire RH56DFX 的 Python 功能入口。硬件网关位于
[`gear_sonic_deploy/inspire/`](../../../gear_sonic_deploy/inspire/README.md)。

可以单独控制手，无需启动 G1 身体控制、SONIC、PICO 或 VLA。当前驱动按左右两只手管理，尚不支持只接一只实体手。代码仍使用本仓库公共接口，不是复制本目录即可安装的独立 Python 包。

## 先运行手部仿真

在仓库根目录，用已有 `.venv_sim` 环境运行：

```bash
.venv_sim/bin/python -m gear_sonic.devices.inspire.examples.hand_control --backend sim
.venv_sim/bin/python -m gear_sonic.devices.inspire.examples.hand_control --backend sim --demo
.venv_sim/bin/python -m gear_sonic.devices.inspire.examples.hand_control --backend sim --sequence
```

第一条只读取状态；第二条弯曲左食指并旋转右拇指；第三条按左手、右手顺序逐指开合。这个本地仿真不需要另起身体程序或共享场景。`sim-remote` 则连接已经启动的共享仿真网关，两者不要混用。

## 控制数据

每只手的目标与反馈都是六个数，项目内部单位为弧度：

```text
[小指, 无名指, 中指, 食指, 拇指弯曲, 拇指旋转]
```

例如 `[0, 0, 0, 0.8, 0, 0]` 表示食指目标为 0.8，其余通道目标为 0。零不是“不控制”。这些是本项目的设备坐标，C++ 驱动再转换成手的串口表示，不应直接作为硬件寄存器值发送。

真实手部流程是：

```text
示例／你的程序 → adapter → driver/client.py → ZMQ → C++ 网关 → 串口 → 手
                                      ← 实际位置反馈 ←
```

主要 API：

```python
from gear_sonic.utils.hand_control.registry import deployment
from gear_sonic.utils.hand_control.interface import EffectorAction

hand = deployment("inspire", "real", options={"hand_config": "/absolute/path/hand.yaml"}).create()
try:
    hand.connect()                # 连接网关并读取反馈；不自动发位置命令
    state = hand.get_state()
    # 首条目标必须靠近实际反馈。此处仅保持当前姿态，不直接跳到新姿态。
    hand.send_action(EffectorAction(state.values))
finally:
    hand.close()                 # 结束本客户端会话；不关闭外部网关
```

到其他姿态的运动应参考 `examples/hand_control.py`：从实际反馈开始插值，按 50 Hz 发送目标并检查反馈。`--left` 和 `--right` 必须各给六个数，`--duration` 控制过渡时间。首次上手优先读状态，再使用明确指定的动作。

## 代码在哪里

| 目录／文件 | 职责 |
|---|---|
| `examples/hand_control.py` | 指定目标、逐指开合、插值与读取反馈 |
| `adapter.py` | 对接项目公共 EndEffector 接口 |
| `contract.py`、`config.py`、`safety.py` | 数据格式、通道映射、配置及命令检查 |
| `driver/client.py`、`driver/session.py` | Python 通信客户端与控制会话 |
| `driver/supervisor.py` | 启动／监视网关；可选启动身体进程 |
| `simulation/`、`assets/` | 仿真及模型资源 |
| `retargeting.py`、`optical.py` | PICO 光学手部重定向 |
| `recording/`、`replay.py` | 采集与回放 |
| `training/`、`inference.py` | 训练准备与 VLA 集成 |

## 依赖与兼容边界

- 真机手部基本 Python 依赖：NumPy、PyYAML、pyzmq、msgpack；公共代码使用 `gear_sonic/utils/hand_control/`。在仓库根目录启动即可解析本地模块。普通 `pip install gear_sonic` 的完整依赖范围大于这条手部链路。
- 本地仿真额外需要 MuJoCo；装配模型检查需要 Pinocchio。PICO、相机和 VLA 依赖只在使用相关功能时需要。
- `manifest.json` 保存内置关节、模型和接口描述，`assets/hand_config.yaml` 保存手部映射与连接参数。G1 装配模型使用共享 G1 网格。旧 `data/end_effectors/` 路径仅保留文件兼容链接，不再用于发现或安装第三方设备。
- 原 `python -m gear_sonic.scripts.inspire_hand_example` 转发到同一个示例实现；G1 控制台可以继续使用它。
- 原 `gear_sonic_deploy/scripts/inspire_hand.py` 转发到同一个 supervisor。
- `driver/native` 仅为兼容符号链接；C++ 实现的唯一位置是 `gear_sonic_deploy/inspire/native/`。
- 本地源码位置不决定机器人上网关的安装目录。现有远端目录、SSH 配置与控制台操作方式保持兼容。

编译、串口配置和仅手部启动见[部署说明](../../../gear_sonic_deploy/inspire/README.md)；完整集成见[Inspire 使用教程](../../../docs/source/tutorials/inspire_hands.md)。
