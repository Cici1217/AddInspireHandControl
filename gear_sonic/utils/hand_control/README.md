# 身体与手部共用接口

项目仅内置 Inspire 和原版 Dex3，不再提供第三方设备安装、卸载或模块管理。

| 文件 | 用途 |
|---|---|
| `interface.py` | 目标、反馈、通道、限位和故障类型 |
| `registry.py` | 读取内置两种手的描述，选择仿真／真机后端 |
| `models.py` | 解析模型资源，检查关节、执行器与单位的一致性 |
| `management.py` | 记录任务进程、释放资源、原子写入报告 |

实际 Inspire 功能在 [`gear_sonic/devices/inspire/`](../../devices/inspire/README.md)：

- `adapter.py`、`driver/`：手部接口和网关通信。
- `simulation/`、`assets/`：仿真和模型。
- `retargeting.py`、`optical.py`：光学手重定向。
- `recording/`、`replay.py`：采集与回放。
- `inference.py`、`policy.py`、`scheduling.py`：VLA 控制、观测与动作调度。

硬件 C++ 网关位于 [`gear_sonic_deploy/inspire/`](../../../gear_sonic_deploy/inspire/README.md)。

内部代码直接导入实际实现，不再经过旧的 `utils/*/inspire/` 转发目录。已有命令行入口、模型路径和录制数据格式继续兼容；真机串口及运动控制协议未改变。
