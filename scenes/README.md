# 自定义仿真场景

默认放在项目的 `scenes/设备名/场景名/scene.xml`。`scenes` 是相对项目根目录的路径；不是相对浏览器、终端当前位置或设备包。

在网页「公共配置 → 本地电脑 → 仿真场景目录」查看或修改目录：默认填 `scenes`，也可以填绝对路径（如 `/absolute/path/to/project/scenes`）。页面显示解析后的实际位置。目录供所有仿真功能共用，更换目录不会搬动已有文件，XML 的资源引用需要有效。

在身体控制、手指控制、PICO 或采集等功能中选择仿真，再选择末端执行器和场景；侧栏不再单列场景入口。下拉框按当前设备的文件夹扫描场景。新增文件夹后重新进入功能页，列表会自动更新。

```text
scenes/
├── inspire/table_box/scene.xml       # G1 + Inspire、桌子、可移动方块
├── inspire/basket_chair/             # 迁入旧椅子、网篮、屏风场景
│   ├── scene.xml
│   ├── assets/                      # 原网格与贴图
│   └── SOURCES.md                   # 原尺寸、物理参数及来源
└── dex3/table_box/scene.xml          # G1 + Dex3、桌子、可移动方块
```

创建自己的 Inspire 场景，在项目目录执行：

```bash
cp -r scenes/inspire/table_box scenes/inspire/my_scene
nano scenes/inspire/my_scene/scene.xml
```

在 `worldbody` 中修改桌子和方块的 `pos`、`size`、`rgba`，也可以增加物体。MuJoCo 的 `size` 对方盒表示三个方向的半长，单位为米。场景专用资源也放在自己的场景文件夹。

保留示例的 `include` 和 `compiler meshdir`：它们复用设备包的机器人模型及网格，不需要复制模型。场景列表只扫描文件，不预先编译或比较机器人结构。点击执行后由 MuJoCo 正常加载；失败时在运行终端显示错误，换一个场景即可。现有控制程序仍需场景包含对应机器人的关节、执行器和传感器。

Inspire 和 Dex3 分别记住场景选择。身体、手指、遥操、采集和回放的仿真入口都会使用该选择；真机不使用它。选「设备默认场景」恢复原模型。切换选择只影响下次启动，已有仿真需要先停止再启动。

命令行也可指定：

```bash
.venv_sim/bin/python -m gear_sonic.scripts.run_sim_loop --interface sim --hand inspire --scene scenes/inspire/my_scene/scene.xml
```

删除自建场景前先停止使用它的任务，再切回默认场景、删除自己的场景文件夹；设备包和历史数据不受影响。

原椅子网篮场景：网页选择 Inspire → `basket_chair`。其项目相对路径为 `scenes/inspire/basket_chair/scene.xml`。机器人、手驱动和控制方式继续使用设备包及现有入口。
