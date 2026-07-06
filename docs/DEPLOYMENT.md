# tubeGrabber 部署与使用文档

本文面向第一次接手项目的人，目标是把代码部署到现场电脑，并按顺序完成硬件检查、视觉扫描、空跑和正式抓放。

## 1. 项目能做什么

tubeGrabber 用 Realman RM-65B 机械臂、Orbbec 336L 腕部 RGB-D 相机、RS485 平行夹爪和 YOLO 模型完成 24 孔试管架抓放任务。

主流程是：

1. 连接机械臂、相机、夹爪。
2. 机械臂移动到 `scan_pose`，拍 RGB-D 图。
3. YOLO 检测 `tube` / `empty`，映射成 `left.a1`、`right.b2` 这类槽位。
4. 用户输入搬运命令，例如 `move left.a1 right.b2`。
5. 到源槽上方二次拍照精定位，夹取，上提。
6. 到目标空槽上方二次拍照精定位，放置，上提。
7. 重新扫描并验证源槽变空、目标槽有管。

重要原则：槽位 ID 是逻辑编号，槽位实际坐标每次由扫描重新计算，不在配置里手写 24 个孔坐标。

## 2. 硬件和软件准备

硬件：

- Realman RM-65B 机械臂，默认地址 `192.168.1.18:8080`。
- Orbbec 336L 深度相机，默认序列号 `CP84B4100090`。
- RS485 平行夹爪，默认使用 Realman SDK 的 RM Plus 夹爪接口。
- 两个 12 孔试管架，共 24 槽。
- YOLO 权重文件：`data/models/best.pt`。

软件：

- Python 3.10 或更高版本。
- 推荐 Linux 现场电脑。相机 USB 权限、Realman SDK、Orbbec SDK 在 Linux 下更容易稳定。
- Python 依赖见 `requirements.txt`。

如果 `pyorbbecsdk2` 或 `Robotic_Arm` 不能直接通过 pip 安装，先按对应厂商 SDK 文档安装本机包，再回到本项目安装依赖。

## 3. 目录说明

```text
tubeGrabber/
├── main.py                 # 主入口，交互 / scan / dry-run / move
├── config/                 # 主配置、标定文件、示教位姿
├── data/models/best.pt     # YOLO 权重，必须存在
├── data/captures/          # 运行截图和深度图，运行时生成
├── drivers/                # 机械臂、相机、夹爪驱动
├── perception/             # YOLO、坐标变换、槽位映射、精定位
├── planning/               # 指令校验和运动路点规划
├── tasks/                  # 抓放状态机
├── world/                  # 24 槽状态表和结果验证
├── utils/                  # 配置、可视化、架面高度标定工具
├── scripts/                # 分层测试和现场辅助脚本
└── docs/                   # 部署和交接文档
```

`Log/` 和 `data/captures/` 里的文件是运行产物。重新部署时它们不是必需文件，但可以作为旧现场结果参考。

`calib/rm65b-handeye-calibration` 是手眼标定相关子模块。只运行抓取程序不一定需要初始化它；如果要重新做手眼标定，再执行子模块初始化。

## 4. 安装步骤

进入项目目录：

```bash
cd /path/to/tubeGrabber
```

如果是从 git 获取代码，并且需要标定子模块：

```bash
git submodule update --init --recursive
```

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

确认关键文件存在：

```bash
python -m compileall .
test -f data/models/best.pt
test -f config/default.yaml
```

`compileall` 只检查 Python 语法，不会连接硬件。

## 5. 配置检查

主配置文件是 `config/default.yaml`。第一次上机前至少检查这些字段：

| 配置段 | 必查项 |
|---|---|
| `arm` | `ip`、`port`、`default_speed`、`approach_speed` |
| `camera` | `serial`、`width`、`height`、`fps`、深度范围 |
| `gripper` | `enabled`、`backend`、`baudrate`、`tool_voltage`、各开合位置 |
| `yolo` | `model_path`、置信度阈值、类别映射 |
| `vision` | dashboard 尺寸、扫描历史数量、实时相机、是否保存最新标注图 |
| `motion` | approach、refine、pick/place 下探和上提高度 |
| `poses` | `scan_pose`、左右 region、`vertical`、`lift_pose` |
| `calib` | 手眼、相机内参、试管架布局文件路径 |

几个容易混淆的单位：

| 项 | 单位 / 含义 |
|---|---|
| 机械臂 `pose_6d` | 位置 mm，姿态 rad |
| `config/hand_eye.yaml` 平移 | 文件中是 m，代码加载后会转成 mm |
| 视觉 `base_xyz` | 基坐标系下夹爪 TCP 目标点，单位 mm |
| `gripper.tcp_offset_mm` | 法兰到夹爪 TCP 的末端坐标系偏移，单位 mm |
| SDK 夹爪位置 | 默认 `0=全闭`，`1000=全开` |

没有显示器或 SSH 运行时，把 `vision.display_enabled` 改为 `false`。首次联调建议把 `runtime.continuous_mode` 改为 `false`，这样抓取下探前仍需要手动按 Enter。

可视化只使用一个 `tubeGrabber Dashboard` 窗口：左列是最近几次全局扫描，中间是精定位和实时相机，右列是 24 槽状态表。dashboard 不等待按键确认，会随流程自动刷新。默认不会自动保存 scan/refine 标注图；需要保留最新标注图时，把 `vision.save_latest_views` 改为 `true`。

## 6. 首次上机验收顺序

按顺序执行，不要跳步。每一步通过后再进入下一步。

### 6.1 机械臂连接

```bash
python scripts/test_arm_connect.py
```

期望结果：

- 能连接 `arm.ip:arm.port`。
- 能读取当前位姿。
- 位姿位置是 mm 量级，不是 m 量级。

如果失败，先检查网线、机械臂 IP、控制器是否上电、SDK 是否安装。

### 6.2 慢速移动到扫描位

```bash
python scripts/test_move_scan.py
```

执行前确认工作空间清空、急停可用。脚本会移动到 `config/poses/scan_pose.json`。

### 6.3 相机取帧

```bash
python scripts/test_camera_capture.py
```

期望结果：

- 能识别 `camera.serial` 对应设备。
- 能输出 color/depth 尺寸。
- `data/captures/test_color.png` 和 `test_depth.png` 被保存。

如果相机找不到，检查 USB、序列号、Orbbec SDK、Linux 设备权限。

### 6.4 夹爪开合

```bash
python scripts/test_gripper.py
```

也可以指定开度：

```bash
python scripts/test_gripper.py --positions 120,130,135,140,115,110
```

确认哪几个开度适合套住、夹住、释放试管，然后同步到 `config/default.yaml` 的 `pick_open_position`、`grip_position`、`release_open_position`。

如果暂时不接夹爪，可以在主流程中使用 `--no-gripper`，或把 `gripper.enabled` 改成 `false`。

### 6.5 scan 位拍照

```bash
python scripts/test_scan_and_capture.py
```

这个脚本会移动到 `scan_pose` 后保存现场 RGB-D 图，是后续离线排查的基础。

### 6.6 坐标变换验证

真机：

```bash
python scripts/verify_coord_transform.py
```

离线：

```bash
python scripts/verify_coord_transform.py --offline data/captures/scan_xxx_color.png data/captures/scan_xxx_depth.png
```

检查输出的基坐标是否与实物位置方向一致。若偏差很大，优先检查相机内参、手眼矩阵、拍照时臂姿和深度图。

### 6.7 槽位映射和状态表

```bash
python scripts/test_slot_mapper_offline.py data/captures/scan_xxx_color.png --depth data/captures/scan_xxx_depth.png
python scripts/test_tube_registry.py data/captures/scan_xxx_color.png --depth data/captures/scan_xxx_depth.png
```

期望能看到 24 个槽位，并且 `tube` / `empty` 数量与现场接近。

### 6.8 标定架面高度

如果空槽无法获得稳定 `base_xyz`，需要标定架面 Z：

```bash
python scripts/calibrate_rack_height.py
```

离线标定：

```bash
python scripts/calibrate_rack_height.py --offline data/captures/scan_xxx_color.png data/captures/scan_xxx_depth.png
```

按窗口提示点击多个空孔位，确认后脚本会写入 `config/rack_layout.yaml` 的 `default_rack_plane_z_mm`。

### 6.9 空跑

先不使用夹爪：

```bash
python main.py dry-run left.a1 right.b2 --no-gripper
```

空跑仍会连接机械臂和相机、全局扫描、移动到源槽/目标槽上方并做精定位，但不会夹取和放置。

### 6.10 正式抓放

确认源槽是 `tube`、目标槽是 `empty` 后执行：

```bash
python main.py move left.a1 right.b2
```

执行完成后会重新扫描并验证：

- 抓取验证：源槽应变成 `empty`。
- 放置验证：目标槽应变成 `tube`。

## 7. 日常运行

交互模式：

```bash
python main.py
```

交互命令：

| 命令 | 作用 |
|---|---|
| `scan` | 重新扫描 24 槽 |
| `table` | 打印当前槽位状态表 |
| `move SRC DST` | 完整抓放 |
| `dry-run SRC DST` | 空跑，不夹取 |
| `SRC DST` | `move SRC DST` 简写 |
| `help` | 显示命令 |
| `quit` | 退出 |

单次命令：

```bash
python main.py scan
python main.py dry-run left.a1 right.b2 --no-gripper
python main.py move left.a1 right.b2
```

槽位格式固定为：

```text
left.a1 ... left.d3
right.a1 ... right.d3
```

大小写不敏感，建议统一小写。

## 8. 常见问题

### 机械臂连接失败

检查：

- 电脑和机械臂是否在同一网段。
- `config/default.yaml` 的 `arm.ip` 是否正确。
- 控制器是否上电、是否处于真实机模式。
- Realman Python SDK 是否安装成功。

### 相机找不到

检查：

- `camera.serial` 是否与设备实际序列号一致。
- USB 是否插稳，是否被其他程序占用。
- Linux 下是否配置了 USB 权限。
- 分辨率和 FPS 是否被相机支持。

### 夹爪初始化失败

检查：

- `tool_voltage` 是否打开工具端 24V。
- `backend` 是否适合当前夹爪，默认 `realman_sdk`。
- `baudrate` 是否与夹爪协议一致。
- 可以先用 `--no-gripper` 验证机械臂和视觉链路。

### 扫描后很多槽位是 unknown

检查：

- YOLO 权重是否存在，类别是否为 `empty` / `tube`。
- 光照、遮挡、相机视野是否覆盖两个试管架。
- `yolo.conf_threshold` 是否过高。
- `scan_pose` 是否仍然适合当前架子摆放。

### 无法估计 z_rack

原因通常是扫描中没有可用试管深度，且 `default_rack_plane_z_mm` 为空。

处理：

```bash
python scripts/calibrate_rack_height.py
```

或离线点击已有 RGB-D 图后写入架面高度。

### move 命令提示源槽或目标槽不合法

`move SRC DST` 要求：

- `SRC` 当前必须是 `tube`。
- `DST` 当前必须是 `empty`。
- 两个槽位都必须有 `base_xyz`。

先执行 `scan` 和 `table` 看状态表，再选择正确槽位。

### refine 失败

检查：

- 近距离图像中目标是否还在视野内。
- 深度范围 `camera.depth_min_mm` / `depth_max_mm` 是否合适。
- `registry.slot_match_max_dist_mm` 是否过小。
- 抓取 refine 失败时，默认不会盲目使用全局坐标；这是为了避免误抓。

### OpenCV 窗口没有显示或刷新慢

dashboard 不需要按 Enter。无显示环境下把：

```yaml
vision:
  display_enabled: false
```

实时相机刷新慢时，可以降低 `vision.live_preview_fps`；如果运动时不想取相机流，可以把 `vision.live_preview_enabled` 改为 `false`。

## 9. 交接清单

交给下一个人前，请确认：

- `data/models/best.pt` 存在。
- `config/default.yaml` 中 IP、相机序列号、夹爪开度符合现场设备。
- `config/hand_eye.yaml`、`config/camera_intrinsics.yaml` 是当前相机和夹爪安装方式对应的标定。
- `config/poses/*.json` 是当前工作台和试管架摆放可达的示教位姿。
- 已经跑通 `test_arm_connect.py`、`test_camera_capture.py`、`test_gripper.py`、`test_scan_and_capture.py`。
- 至少跑通过一次 `python main.py dry-run SRC DST --no-gripper`。
- 正式运行前现场人员知道急停位置，并确认工作空间无障碍。

## 10. 开发者速查

主要调用关系：

```text
main.py
└── tasks/fsm_factory.py
    └── PickPlaceFSM
        ├── drivers/arm_driver.py
        ├── drivers/camera_driver.py
        ├── drivers/gripper_driver.py
        ├── perception/yolo_detector.py
        ├── perception/slot_mapper.py
        ├── perception/refine.py
        ├── planning/command_validator.py
        ├── planning/motion_planner.py
        └── world/tube_registry.py
```

状态机主要阶段：

```text
CHECK_HW -> SCAN_GLOBAL -> PICK_TRANSIT -> PICK_REFINE -> PICK_GRASP
-> VERIFY_PICK -> PLACE_TRANSIT -> PLACE_REFINE -> PLACE_RELEASE
-> VERIFY_PLACE -> DONE
```

如果只改文档或配置，至少运行：

```bash
python -m compileall .
python scripts/test_command_validator.py
python scripts/test_operation_verifier.py
```

如果改了运动规划，至少再运行：

```bash
python scripts/test_motion_planner.py left.a1 right.b2
```
