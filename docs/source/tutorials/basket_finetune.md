# Basket 仿真数据准备

场景入口为 `scenes/inspire/basket_chair/scene.xml`。数据准备程序通过独立物理仿真，记录 G1 + Inspire 双手抓取篮子的轨迹；通过物理检查后才导出训练数据。生成的数据、个人云端账号、提交记录和模型不随仓库发布。

## 在本机生成数据

从项目根目录执行，使用新批次名称：

```bash
.venv_data_collection/bin/python -m gear_sonic.scripts.prepare_basket_finetune \
  --episodes 25 --output outputs/training/basket_capture
```

运行期间不要同时启动另一套仿真、SONIC 或 PICO。生成结果中的 `INCOMPLETE` 标志表示批次未完成，不能用于训练。模型和 Python 环境在公共配置中准备。

数据集包含 50 Hz 的机器人视角图像、身体和手部实测状态、SONIC 动作 token、双手提交目标及相应时间。原始轨迹、接触检查、导出数据和验证报告位于本次输出目录。通过检查的数据才会进入训练／验证集；不会为程序示教伪造 PICO 或 SMPL 参考。

物理检查要求双手同时接触、篮子持续提离支撑面，以及身体没有外部支撑力。检查通过只表示录制满足设定条件，不代表微调模型已学会抓取或可直接迁移到实机。

## 检查与训练

设置自己的 Isaac-GR00T 安装位置，然后执行生成的数据检查脚本：

```bash
GROOT_SOURCE_DIR=/absolute/path/to/Isaac-GR00T
"$GROOT_SOURCE_DIR/.venv/bin/python" \
  outputs/training/basket_capture/train.py \
  --groot-root "$GROOT_SOURCE_DIR"
```

不带 `--execute` 时只进行数据和模型前向检查；训练参数见生成的 `finetune.yaml` 和 `train.py --help`。本机训练输出放在 `outputs/checkpoints/finetuned/`，VLA 选择其中具体的 checkpoint 目录。

## 可选：Slurm / Apptainer

`cloud.slurm` 是模板。准备集群上的工作目录、Isaac-GR00T、Python 环境、模型缓存和容器镜像后，在解压的数据包目录提交：

```bash
export GROOT_CLOUD_ROOT=/absolute/path/to/cloud-workspace
export GROOT_CONTAINER_IMAGE=/absolute/path/to/pytorch.sif
sbatch --account=YOUR_ACCOUNT cloud.slurm check
```

工作目录与容器路径必须改成实际位置。模板不包含任何集群账号、登录主机或个人作业号。先检查集群的资源和环境要求，再按模板注释配置；使用 `train` 代替 `check` 才会训练。`GROOT_CLOUD_ROOT` 必须显式设置；还可通过 `BASKET_BUNDLE`、`BASKET_CHECKPOINT`、`BASKET_GROOT_ROOT` 指定数据、模型和源码位置。

日志默认写入提交目录的 `groot_basket_<job-id>.out`；模型写入云端工作目录的 `outputs/checkpoints/finetuned/`。需要分享的数据和权重单独分发，不加入 Git。

## 在控制台查看

在「05 SONIC 动作回放」选择仿真，选择或手动填写本次导出的数据集，设置 Episode，执行检查和回放。

含 `meta/simulation_replay.json` 的数据会还原录制时的完整场景和物理状态，要求兼容的 MuJoCo 版本；无需启动 SONIC 或 PICO，也不能当作实机控制命令。普通 PICO 动作录制的回放方式见控制台说明。
