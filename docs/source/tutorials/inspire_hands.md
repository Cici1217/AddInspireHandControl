# Inspire hand support

Inspire RH56DFX support for MuJoCo and serial hardware, integrated with the
existing deployment entry points. SONIC controls the G1 body; a separate stream
controls six finger channels per hand. Omitting `--hand` preserves the original
Dex3 deployment.

只控制手、不运行身体时，从 [Python 手部接口与示例](../../../gear_sonic/devices/inspire/README.md)
和 [独立网关部署](../../../gear_sonic_deploy/inspire/README.md) 开始。
Python 实现集中在 `gear_sonic/devices/inspire/`，C++ 网关集中在
`gear_sonic_deploy/inspire/native/`；旧命令及模型文件路径保留兼容入口，内部 Python 导入直接使用上述目录。
仅手部监视／启动可用 `python -m gear_sonic.devices.inspire.driver.supervisor --hand-only`，
配合 `--mode` 和 `--config` 使用。原来的身体联合部署方式保持不变。

PICO optical teleoperation and independent LeRobot recording are included below.
Dataset replay and VLA are available through the [console](console.md) and
[Inspire VLA guide](vla_inspire.md). Inspire is the supported custom hand;
the original Dex3 body simulation and motion replay remain available.
The project no longer installs or manages additional device plugins.

Use the existing `gear_sonic_deploy/deploy.sh` for both hands. Model selection,
network selection, build, deployment confirmation and body keyboard controls
use the same path. Leave out `--hand` for Dex3, or add `--hand inspire` for Inspire.
The launcher previews the exact command it will execute, including the Inspire
communication supervisor when selected; no separate Python deployment command
is needed.

本地网页入口见 [网页控制台](console.md)：先选功能，再选仿真/真机，配置路径并预览实际命令。

## Setup

Complete the existing [deployment setup](../getting_started/installation_deploy.md)
and [model download](../getting_started/download_models.md).
For the simulation Python environment, use the existing installer from the
repository root. Run it once when setting up a checkout; it recreates `.venv_sim`:

```bash
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
git lfs pull --include="gear_sonic/data/robots/g1/meshes/**,gear_sonic_deploy/reference/example/**" --exclude=""
```

Reference CSVs must contain numeric motion data, not Git LFS pointer text.
Keep planner models in their downloaded directory layout, including `V2`.
The short deployment command below uses the existing launcher defaults:

```text
gear_sonic_deploy/
├── policy/release/
│   ├── model_encoder.onnx
│   ├── model_decoder.onnx
│   └── observation_config.yaml
├── planner/target_vel/V2/planner_sonic.onnx
└── reference/example/
```

The environment and model binaries are local, ignored files. Do not commit
virtual environments, generated TensorRT engines or machine-specific paths.
Custom model locations can still be supplied with the upstream `--cp`,
`--obs-config`, `--planner` and `--motion-data` options.

## Full-body simulation

In each new terminal, start at the repository root and activate the environment
once with `source .venv_sim/bin/activate`. The launcher uses its `python3` by
default; `--hand-python` is only needed to select a different interpreter.
Then start these in separate terminals on the computer:

```bash
# Terminal 1, repository root: one shared body-and-hands world.
python gear_sonic/scripts/run_sim_loop.py --interface sim --hand inspire

# Terminal 2, repository root: SONIC body controller.
cd gear_sonic_deploy
./deploy.sh sim --hand inspire --input-type keyboard --output-type zmq
```

After `Init Done`, press `]` in the controller terminal to start body control.
Wait for `transitioning to CONTROL state`, then press Enter in that same terminal
to enable the idle planner. Keep the support band attached until
`[Keyboard] motion name is planner_motion` appears; allow three seconds for the
transition, then press `9` once in the MuJoCo window to release the band. Check
that the robot is standing before sending hand targets. Releasing the band
before switching to the planner can cause a fall during the time spent switching
windows. Do not enable the planner before starting control.

Send a small hand trajectory from a third terminal at the repository root:

```bash
python -m gear_sonic.scripts.inspire_hand_example --backend sim-remote --demo
```

Without `--demo`, the example only reads feedback. `--backend sim` instead uses
a standalone hand test scene with a fixed body. For custom endpoints, pass the
same configuration via `--hand-config` to the simulator and deployment launcher,
and via `--config` to the example.

## Hardware: computer controller, robot serial bridge

```text
Computer: SONIC + models ── DDS ──────────────────────> G1 body
          hand client ── ZMQ through SSH ──> Robot: C++ bridge ── serial ──> hands
```

The robot needs only the bridge executable, configuration and launcher. No
Python, ROS, Unitree SDK or model files are required by the bridge. Building
requires CMake, a C++20 compiler, libzmq/cppzmq and MessagePack headers; runtime
uses libzmq and the standard C++ libraries. Reuse installed dependencies.

For a wired setup, connect the computer to the G1 development Ethernet port and
give that computer interface an unused address in the robot's subnet (commonly
`192.168.123.0/24`). This connection does not need a default gateway. Use the
onboard computer's Ethernet address for SSH and verify that `rt/lowstate` is
received on the selected computer interface before starting control. The same
cable carries body DDS, SSH and the hand tunnel; connecting to the robot's Wi-Fi
is not required. Stop other body controllers and hand drivers before this run.

On the computer, package the existing driver source and copy it to a fresh
robot directory. Replace `USER@ROBOT` with your SSH destination:

```bash
python gear_sonic_deploy/inspire/native/bridge/package.py /tmp/inspire-bridge.tar.gz
ssh USER@ROBOT 'mkdir -p ~/inspire_hand_bridge'
scp /tmp/inspire-bridge.tar.gz USER@ROBOT:~/inspire_hand_bridge/
```

On the robot, build entirely inside that directory:

```bash
cd ~/inspire_hand_bridge
mkdir .build
tar -xzf inspire-bridge.tar.gz -C .build
TMPDIR="$PWD/.build" cmake -S .build/source -B .build/build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
TMPDIR="$PWD/.build" cmake --build .build/build --target inspire_direct_serial_hand_gateway -j2
install -m 755 .build/build/inspire_direct_serial_hand_gateway ./inspire_hand_bridge
install -m 755 .build/run.sh ./run.sh
cp .build/bridge.env.example ./bridge.env
cp .build/LICENSE .build/LICENSE-*.txt .build/NOTICE.md .build/source-manifest.json .
```

Edit `bridge.env` to match the two local serial devices, IDs and baud rate.
Defaults are 50 Hz, 57600 baud and device ID 1 per hand. Use plain `KEY=value`
without shell quotes. The generated `.build` and archive are not needed at runtime.

```bash
./run.sh --check  # Validate paths/configuration without opening serial devices.
./run.sh          # Start in foreground; sets and verifies the device speed.
```

On the computer, keep this tunnel running in a separate terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15561:127.0.0.1:5561 \
  -L 127.0.0.1:15562:127.0.0.1:5562 USER@ROBOT
```

Copy [hand_config.yaml](../../../gear_sonic/data/robots/g1/inspire/hand_config.yaml) to a computer-side configuration. Under
`hardware`, change only these endpoints; local device paths may remain null:

```yaml
command_endpoint: tcp://127.0.0.1:15561
state_endpoint: tcp://127.0.0.1:15562
```

Read feedback, then start SONIC on the computer:

```bash
python -m gear_sonic.scripts.inspire_hand_example --backend real --config /path/to/hand.yaml
cd gear_sonic_deploy
./deploy.sh COMPUTER_DDS_INTERFACE --hand inspire --hand-remote \
  --hand-config /path/to/hand.yaml --input-type keyboard --output-type zmq
```

Use the computer's interface connected to the robot's DDS network. The hand SSH
tunnel does not carry body DDS; successful Wi-Fi/SSH access alone is insufficient.
Follow the controller startup order above, without the simulator support-band key.
The launcher requires healthy, disarmed hand feedback before starting the body.
It monitors feedback freshness and gateway identity but does not start or stop
the remote bridge. `--hand-remote` cannot be used with sim, Dex3 or `--hand-driver`.

If body control and hand serial devices are on the same machine, omit
`--hand-remote`, configure the local device paths in YAML, and build the driver
with `cmake -S gear_sonic_deploy/inspire/native -B build/inspire-native`
and `cmake --build build/inspire-native`. The launcher then owns that driver.

### Body control on G1

For the prepared G1 deployment, the body controller runs in
`~/inspire_hand_bridge/sonic/` and the serial bridge in
`~/inspire_hand_bridge/hand_bridge/`. The computer runs PICO and forwards body
and hand streams over SSH; body DDS stays on G1's internal interface.

Keep the bridge endpoints at `127.0.0.1:5561/5562` on both machines. Run these
commands in separate computer terminals:

```bash
# Bridge plus body/hand forwarding.
ssh -t -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:5556:127.0.0.1:5556 \
  -L 127.0.0.1:5557:127.0.0.1:5557 \
  -L 127.0.0.1:5561:127.0.0.1:5561 \
  -L 127.0.0.1:5562:127.0.0.1:5562 \
  USER@ROBOT 'cd ~/inspire_hand_bridge/hand_bridge && ./run.sh'

# Body controller for PICO; add --keyboard for independent manual control.
ssh -t USER@ROBOT 'cd ~/inspire_hand_bridge/sonic && ./run_body.sh'
```

`run_body.sh` uses the existing G1 `low_latency` models and the internal DDS
interface. `--check` validates its paths without starting control. It launches
the native body controller directly; in keyboard mode a hand disconnect does
not automatically stop the body. Press `O` in the body terminal to stop it.

## Hand commands and stopping

Upper-level entry points now use the registered `EndEffector` interface:
`connect()`, `get_state()`, `send_action()`, `stop()`, `reset()` and `close()`.
`InspireAdapter` preserves the existing gateway protocol and simulation controller;
the old `HandBackend` remains an internal driver boundary. See the centralized
[device integration guide](../../../gear_sonic/utils/hand_control/README.md)
for registration, deployment and the MuJoCo parallel-gripper example.

Each target has six angles in radians, ordered
`pinky, ring, middle, index, thumb_bend, thumb_rotation`. Bounds are
`[0, 1.7/1.7/1.7/1.7/0.6/1.3]`; velocity limits are `[2, 2, 2, 2, 1, 1]` rad/s.
Start near measured feedback and ramp within these limits. The first valid target
acquires the session automatically; manual RESET/ARM commands are unnecessary.
`native6` in diagnostic output names the six-channel hand protocol, not a
separate program to start. ARM/DISARM are internal session states handled by
the client; the example acquires ownership on its first target and releases it
on exit.
The example accepts explicit `--left` and `--right` targets with `--duration`;
`--demo` is simulation-only.

To check every finger while the body remains under SONIC control, run this
explicit sequence from another computer terminal after the idle planner is ready:

```bash
# Hardware; the bridge and body controller must already be running.
python -m gear_sonic.scripts.inspire_hand_example --backend real \
  --config /path/to/hand.yaml --sequence

# Shared full-body simulation uses the same sequence.
python -m gear_sonic.scripts.inspire_hand_example --backend sim-remote --sequence
```

The sequence closes then opens the left thumb, index, middle, ring and pinky,
then repeats on the right hand. Only the selected finger's targets change.
It starts from measured positions, uses 80% of the configured joint velocity
limits. Each stage publishes for `--duration` (default 2 seconds), or for the
full transition if longer, holding its target after the ramp. Measurements are
recorded without adding an arrival threshold. A full four-finger joint
stroke takes about 1.06 seconds each way; thumb bending takes about 0.75 seconds.
Thumb closure uses bend `0.6` and rotation `0.3` rad to keep clear of the closed
index, rather than the rotation joint's full `1.3` rad limit. On completion the
example disarms the hands. It does not stop the separately launched body: press
`O` in the body terminal, then stop the bridge. No sequence runs automatically
when `--hand inspire` is selected.

Add `--log-file hand-sequence.jsonl` to save the sequence from the example.
The file must be new. Each submitted target is recorded with its stage, local
submission time, measured left/right positions and local feedback receipt time.
Repeated feedback keeps its original receipt time; a completed stage means its
publication interval ended, not that every joint physically reached the target.
For body data, the existing ZMQ output provides `last_action` (body position
commands) and `body_q` (measured joints), both in MuJoCo joint order;
`body_q_target` is the reference motion, not the motor command.
Body CSV recording is off by default. Add `--enable-csv-logs` to `deploy.sh`
to save the existing body state logs; optionally use `--logs-dir /path/to/logs`
to select the output directory. This does not enable or change finger commands.

The target-producing client must call `stop()`/`close()` and confirm DISARM
before exit. Stop the body with `O`, then the foreground bridge with Ctrl+C.
For a local bridge started by `deploy.sh`, the launcher stops its owned body
and bridge on exit, interruption or startup failure. With `--hand-remote`, the
independently started bridge remains running; close the hand client first and
stop that bridge in its own terminal. The simulation process also remains
separate, as in Dex3 deployment.
`close()` releases the connection; it does not close the fingers. Hardware stop
requests a measured-position hold; normal driver exit restores device speed.
The body's stop path applies damping, not active balance.

**Hardware command-stream timeout is disabled:** lost commands or a broken
network can leave the last target active; remote DISARM cannot be guaranteed
when communication is lost. Feedback faults inhibit writes and remain latched.
Simulation instead zeros hand control after 200 ms without a command. The 50 Hz
hardware setting is a target rate; serial overruns are reported without catch-up
bursts. No preset hand opening/closing is sent at startup or shutdown.

## Validation

```bash
ctest --test-dir build/inspire-native --output-on-failure
INSPIRE_TEST_GATEWAY="$PWD/build/inspire-native/inspire_direct_serial_hand_gateway" \
  python -m pytest -q gear_sonic/tests/test_inspire_*.py
```

Offline checks cover session ownership, measured feedback, sequence logging,
serial/process emulation and deployment entry points. The test suite passed
74 Python checks and seven C++ checks.

Full-body MuJoCo GUI validation exercised `deploy.sh`, default endpoints and
pauses between operator actions. The robot completed the 20-stage close/open
sequence while standing, followed by confirmed hand DISARM and normal body
Stop. Use the planner-before-band-release startup order above.

Hardware validation used a 29-DoF G1 with Inspire hands, computer-side SONIC
over wired DDS, and the serial bridge on the onboard computer. The operator
confirmed standing, both complete finger sequences and normal shutdown. Two
recorded runs each contain 20 stages, 2,000 hand samples over approximately
40 seconds, and 1,999 concurrent body position/action frames. Target submission
was approximately 49.99 Hz; this is not a measurement of serial feedback rate.
Stage completion records the end of publication, not a joint-arrival tolerance
check. These trials validate the tested standing and finger sequence, not
locomotion or the full sim-to-real motion envelope.

Model provenance is in [ASSETS.md](../../../gear_sonic/data/robots/g1/inspire/ASSETS.md);
retained credits and licenses are in [third-party notices](../../../legal/NOTICE-inspire-hands.md).

## PICO optical teleoperation

Inspire 使用 `run_pico_teleop.py`，不需要 ABXY、RESET 或手动 ARM。
头显先连接电脑的 XRoboToolkit PC Service，并开启身体和光学手跟踪。
首次安装环境：

```bash
bash install_scripts/install_pico.sh
```

仿真在三个终端运行。每个终端从仓库根目录开始。

```bash
# 终端一：仿真和相机图像，后面录制也用这一进程。
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py --interface sim --hand inspire \
  --enable-image-publish --enable-offscreen --camera-port 5555
```

```bash
# 终端二：沿用原部署入口。
source .venv_sim/bin/activate
cd gear_sonic_deploy
./deploy.sh sim --hand inspire --input-type zmq_manager --output-type zmq
```

```bash
# 终端三：PICO 身体和双手遥操。
source .venv_teleop/bin/activate
python -m gear_sonic.scripts.run_pico_teleop --sim
```

第一次没有标定文件时，程序直接提示采集四种手势，保存后自动复用。
等身体部署显示 `Init Done`，站好身体标定姿态，在 PICO 终端按 Enter。
程序自动启动身体并切到 PICO 流；确认控制已启动后，在仿真窗口按 `9` 释放支撑。
这条流程不使用 keyboard 模式的 `] → Enter`，也不使用旧 manager 的 ABXY。

真机按上面的“Body control on G1”启动 SSH 转发、手桥接和身体程序。
电脑的 PICO 命令去掉 `--sim`，加 `--hand-config /path/to/hand.yaml`；
手配置保持 `127.0.0.1:5561/5562`。不要同时运行逐指示例。

PICO 终端 Ctrl+C 停止遥操：发送身体 Stop，并检查手部 DISARM。
手目标独立每 20ms 提交，保留限速和碰撞约束。沿用旧 PICO 的光学丢失处理：手被遮挡时保持上一目标，重新识别后继续跟随。网关故障仍停止手部发送并提示，身体数据继续发送，不自动清故障。
手反馈短暂过期时显示 `PICO hand feedback paused`，暂停目标发送并持续接收反馈；同一网关、同一控制会话恢复健康反馈后自动继续，显示 `control resumed`。暂停期间不累计目标位移，恢复从最后成功提交的位置限速过渡；FAULT、网关重启或控制权丢失仍停止手部。普通手指测试保持原来的报错停止行为。
身体只处理新帧；短暂停更等待恢复，连续 5 秒没有新帧才停止遥操。退出时保留最初的错误，手部 DISARM 失败单独报告。
断网时无法保证远端停止送达，硬件 watchdog 配置仍沿原实现。
需要重新标定时才单独运行 `python -m gear_sonic.scripts.calibrate_pico_hands`。

## Independent LeRobot recording

遥操保持运行，录制另外启动。首次安装录制环境：

```bash
bash install_scripts/install_data_collection.sh
```

```bash
source .venv_data_collection/bin/activate
python -m gear_sonic.scripts.record_pico_lerobot \
  --sim --output outputs/datasets/sim/pico_demo_001 --task-prompt "拿起物体"
```

命令运行后直接采集，不再按 Enter。默认录制 600 个有效样本，名义 12 秒；
用 `--frames` 调整长度，每次换一个新输出目录。Ctrl+C 只取消录制，遥操继续运行。
真机去掉 `--sim`，加同一手配置和 `--camera-host` 指向机器人相机 IP。
相机须发布 640×480 的 `ego_view`，仿真已由终端一发布。

输出沿用 LeRobot 的 Parquet、MP4 和 metadata。41维实测状态、六路手实测、
六路提交目标和原始手部意图分别保存，含64维 SONIC token及来源时间。
成功检查后发布目录，失败留 INCOMPLETE，不覆盖已有数据。训练统计不在录制里运行。

新录制还会自动保存一份 SONIC 原生身体动作参考，原有 Parquet、视频、SMPL、
token 和双手数据格式保持不变：

```text
数据集目录/
└── sonic_reference/
    ├── episode_000000/
    │   ├── joint_pos.csv    # 29 个实测关节角，弧度，IsaacLab 顺序
    │   ├── joint_vel.csv    # 29 个实测关节速度，rad/s，同一顺序
    │   ├── body_quat.csv    # 实测身体根节点朝向，wxyz
    │   ├── body_pos.csv     # 零占位；没有记录世界位置
    │   ├── metadata.txt    # 根节点索引 [0]
    │   └── info.txt        # 数据来源和格式说明
    └── episode_000001/     # 同一批次下一条录制
```

在“05 SONIC 动作回放”选择同一个数据集和 Episode，再选“实测关节角参考”，
程序自动读取对应文件，使用 G1 编码模式，并同步播放原有双手目标。
当前 release/low_latency 的该模式不使用根节点 XYZ。
这份参考按实际采样时间整理为 50 Hz，丢帧间隔会插值，因此行数可能与训练数据不同。
它只包含身体动作，不包含场景或双手控制，不表示恢复机器人在世界中的行走轨迹。
原有录制不会被回写，默认仍使用 SMPL 人体参考。

当前已做离线控制和实际文件导出验证；真实头显控制完整仿真及实体 Wi-Fi 仍待现场测试。
50Hz 是目标提交和名义采样，不能等同于实体反馈频率。
