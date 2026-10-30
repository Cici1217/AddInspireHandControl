# Inspire 硬件部署

本目录集中 C++ 串口网关、测试和桥接部署脚本。Python 接口与示例在
[`gear_sonic/devices/inspire/`](../../gear_sonic/devices/inspire/README.md)。

网关运行在接有手串口的 Linux 机器上，可以是机器人上的电脑，也可以是独立电脑。只控制手不需要启动 SONIC 或 G1 身体。当前实现需要左右两只手。

## 编译

需要 CMake、C++20 编译器、ZeroMQ、cppzmq 和 msgpack 开发依赖。在仓库根目录执行：

```bash
cmake -S gear_sonic_deploy/inspire/native -B build/inspire-native -DBUILD_TESTING=ON
cmake --build build/inspire-native
ctest --test-dir build/inspire-native --output-on-failure
```

可执行文件仍为 `build/inspire-native/inspire_direct_serial_hand_gateway`，原调用方无需更改二进制名称。原 `gear_sonic_deploy/src/inspire_hand_bridge` 和 Python 设备包内的 `driver/native` 都是兼容符号链接，指向本目录的 `native/`，没有第二份实现。

## 本机仅手部部署

复制 Python 设备包的 `assets/hand_config.yaml` 到自己的运行配置目录，填写 `hardware.left_device`、`right_device`、对应设备 ID、波特率及命令／反馈地址。左右串口必须对应实际连接。不要改六通道顺序或把目标弧度当成串口原始值。

在已安装手部 Python 依赖的环境中，先验证配置：

```bash
python -m gear_sonic.devices.inspire.driver.supervisor \
  --mode real --hand-only --config /absolute/path/hand.yaml --check
```

`--check` 不打开串口、不发送控制命令。配置完成后，以下命令会连接硬件并启动网关：

```bash
python -m gear_sonic.devices.inspire.driver.supervisor \
  --mode real --hand-only --config /absolute/path/hand.yaml
```

该模式不启动身体进程，也不自动发送手指位置目标；C++ 网关仍执行原有硬件初始化。另一个终端读取反馈：

```bash
python -m gear_sonic.devices.inspire.examples.hand_control \
  --backend real --config /absolute/path/hand.yaml
```

要执行逐指动作，在上条命令后加 `--sequence`；这会让实体手运动。先结束控制客户端，再退出仅手部部署进程。部署进程只终止自己创建的网关，网关保留原有退出恢复逻辑。

`--hand-only` 不能同时指定身体命令。原有不带此参数、以 `-- BODY_COMMAND ...` 启动身体与手的方式保持不变。

## 远端串口网关

使用源码桥接包可在没有完整 Python 项目的机器上编译 C++ 网关：

```bash
python gear_sonic_deploy/inspire/native/bridge/package.py /tmp/inspire-bridge.tar.gz
```

输出文件必须尚不存在。包内包含 C++ 源码、`run.sh`、配置模板、许可和来源校验清单。在目标机器解包后：

```bash
cmake -S source -B build -DBUILD_TESTING=OFF
cmake --build build
install -m 755 build/inspire_direct_serial_hand_gateway ./inspire_hand_bridge
cp bridge.env.example bridge.env
```

填写 `bridge.env` 中的实际串口和通信参数，再运行 `./run.sh --check`，实际启动使用 `./run.sh`。脚本只读取与自身相邻的配置；不会自动修改控制台保存的远端路径。

计算机侧配置命令／反馈地址后，可以使用 `--mode real --remote --hand-only` 监视已启动的远端网关；该进程不拥有远端网关，退出时也不会停止它。客户端仍通过自己的手部会话进行控制。

## 不连接实体手的验证

原生 CTest 覆盖协议、数值转换、反馈检查和会话。Python 串口测试通过 PTY 模拟左右手，以真实 C++ 网关验证发送、反馈、停止和恢复：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
INSPIRE_TEST_GATEWAY="$PWD/build/inspire-native/inspire_direct_serial_hand_gateway" \
.venv_sim/bin/python -m pytest gear_sonic/tests/test_inspire_layout.py \
  gear_sonic/tests/test_inspire_serial_e2e.py gear_sonic/tests/test_inspire_deploy.py
```

这些测试不会访问真实手的串口。模拟验证通过不等于真机验收；真实手的接线、机械运动和实际负载需要在硬件阶段确认。
