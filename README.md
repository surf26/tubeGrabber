# tubeGrabber

**24 孔试管架视觉引导抓放系统**

Realman RM-65B · Orbbec 336L 腕部相机 · RS485 平行夹爪 · YOLO（`empty` / `tube`）

---

## 概述

`tubeGrabber` 是一套分层式机器人软件栈，用于在两个 12 孔试管架（共 24 槽）之间自动搬运试管。槽位 ID（如 `left.a1`、`right.b2`）为逻辑编号、长期稳定；**物理坐标在每次全局扫描时由 RGB-D 视觉、手眼标定与当前臂姿重新解算**，不写入配置文件。

**设计要点**

- **扫描驱动世界模型** — 配置中不硬编码槽位坐标
- **停稳—拍照—精定位** — 逼近阶段在臂停止后重新取帧；运动过程中不做 YOLO 追踪
- **TCP 感知运动规划** — 视觉输出夹爪 TCP 目标；规划器通过 `gripper.tcp_offset_mm` 换算法兰位姿
- **严格分层** — `drivers` → `perception` → `world` → `planning` → `tasks`

---

## 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│  main.py / PickPlaceFSM                                     │
├──────────────┬──────────────┬──────────────┬──────────────——┤
│  planning/   │  world/      │  perception/ │  drivers/      │
│  指令校验     │  状态表       │  YOLO         │  机械臂         │
│  路点规划     │  registry    │  槽位映射      │  相机           │
│              │              │  精定位       │  夹爪           │
└──────────────┴──────────────┴──────────────┴──────────────——┘
         config/          data/models/        utils/
```

| 层级 | 职责 |
|------|------|
| `drivers/` | 硬件 I/O（臂、相机、夹爪 Modbus） |
| `perception/` | 检测、像素→基坐标、24 槽映射、精定位 |
| `world/` | 24 槽状态持久化（`TubeRegistry`） |
| `planning/` | 指令校验、运动路点生成 |
| `tasks/` | 抓放状态机编排 |

---

## 硬件配置

| 设备 | 规格 |
|------|------|
| 机械臂 | Realman RM-65B，`192.168.1.18:8080` |
| 相机 | Orbbec 336L（eye-in-hand），640×480，序列号见配置 |
| 夹爪 | RS485 平行夹爪，经臂末端 Modbus（`port=1`） |
| 视觉模型 | YOLO 权重 `data/models/best.pt`，类别 `empty` / `tube` |

---

## 环境要求

- Python 3.10+
- Linux，深度相机 USB 可用

```bash
pip install -r requirements.txt
```

主要依赖：`numpy`、`opencv-python`、`pyyaml`、`ultralytics`、`pyorbbecsdk2`、`Robotic_Arm`。

从零部署、现场验收和排错流程见：[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)。

---

## 配置说明

主配置文件：[`config/default.yaml`](config/default.yaml)

| 配置段 | 说明 |
|--------|------|
| `arm` | IP、端口、运动速度 |
| `camera` | 序列号、分辨率、深度有效范围 |
| `gripper` | Modbus 参数、**`tcp_offset_mm`**（法兰→指尖，末端坐标系） |
| `yolo` | 模型路径、阈值、类别映射 |
| `vision` | 精定位置信度、可视化（`display_enabled`、`font_scale`） |
| `motion` | 逼近高度、插入/退避深度 |
| `poses` | 示教位姿：scan、region、vertical（`config/poses/*.json`） |
| `calib` | 手眼、内参、试管架布局 |

标定文件：

- [`config/hand_eye.yaml`](config/hand_eye.yaml) — eye-in-hand `T_ee_cam`
- [`config/camera_intrinsics.yaml`](config/camera_intrinsics.yaml)
- [`config/rack_layout.yaml`](config/rack_layout.yaml) — 槽位命名；`default_rack_plane_z_mm` 由架面标定写入

---

## 快速开始

### 交互模式

```bash
python main.py
```

启动后自动：连接硬件 → 全局扫描 → 进入命令行交互。

### 单次命令

```bash
python main.py scan                                    # 连接并扫描
python main.py dry-run left.a1 right.b2 --no-gripper   # 空跑：移动 + 精定位，不夹取
python main.py move left.a1 right.b2                   # 完整抓放
```

### 交互命令

| 命令 | 说明 |
|------|------|
| `scan` | 重新全局扫描 24 槽 |
| `table` | 打印当前状态表 |
| `move SRC DST` | 完整抓放（含 VERIFY） |
| `dry-run SRC DST` | 仅 transit + refine，不夹取/放置 |
| `SRC DST` | `move` 的简写 |
| `quit` | 退出 |

槽位格式：`{side}.{row}{col}`，如 `left.a1`、`right.b2`（建议小写）。

---

## 测试脚本

按顺序执行，勿跳步。

| 步骤 | 命令 | 目标 |
|------|------|------|
| 1 | `python scripts/test_arm_connect.py` | 臂连接与位姿回读 |
| 2 | `python scripts/test_camera_capture.py` | RGB-D 取流 |
| 3 | `python scripts/test_gripper.py` | 夹爪 Modbus 开合 |
| 4 | `python scripts/test_scan_and_capture.py` | scan 位拍照 |
| 5 | `python scripts/calibrate_rack_height.py` | 架面平面 Z 标定 |
| 6 | `python main.py dry-run … --no-gripper` | 路径与精定位验证 |
| 7 | `python main.py move …` | 端到端搬运 |

完整部署、检查项、安全须知与调参说明：**[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)**

---

## 坐标约定

| 量 | 坐标系 / 含义 |
|----|----------------|
| 臂 `get_pose_6d()` | 基坐标系下的 **法兰** 位姿（mm，rad） |
| 视觉 `base_xyz` | 基坐标系下的 **夹爪 TCP** 目标（试管抓取点） |
| `gripper.tcp_offset_mm` | **末端坐标系**下 TCP 相对法兰的平移；`tip = flange + R @ offset` |
| 手眼 `T_ee_cam` | 相机 → 法兰（非指尖） |

运动规划在生成法兰路点时应用 TCP 偏移；视觉解算链路不变。

---

## 运行时可视化

由 `config/default.yaml` → `vision` 控制：

程序只使用一个窗口：`tubeGrabber Dashboard`。

| 区域 | 内容 |
|------|------|
| 左列 | 最近几次全局 YOLO 扫描结果 |
| 中上 | 最新 pick / place 精定位结果 |
| 中下 | 运动过程实时相机画面 |
| 右列 | 24 槽试管状态表、置信度、坐标与 Z 来源 |

dashboard 不等待按键确认，会随流程自动刷新。无图形界面时设 `vision.display_enabled: false`；如需保存最新标注图，设 `vision.save_latest_views: true`。

---

## 离线验证（无机械臂）

```bash
# 由已保存 scan 图验证 24 槽映射
python scripts/test_slot_mapper_offline.py data/captures/scan_xxx_color.png \
  --depth data/captures/scan_xxx_depth.png

# 规划 / 状态表 / 精定位
python scripts/test_motion_planner.py left.a1 right.b2
python scripts/test_command_validator.py
python scripts/test_refine_offline.py color.png depth.png left.a1
python scripts/test_tube_registry.py color.png --depth depth.png
```

硬件分层测试脚本位于 `scripts/test_*.py`。

---

## 项目结构

```text
tubeGrabber/
├── main.py                 # CLI 入口
├── config/                 # 运行配置、示教位姿、标定文件
├── drivers/                # 机械臂、相机、夹爪驱动
├── perception/             # YOLO、坐标变换、槽位映射、精定位
├── world/                  # TubeRegistry 状态表
├── planning/               # 指令校验、运动规划
├── tasks/                  # 状态机与工厂
├── utils/                  # 配置加载、可视化、架面标定工具
├── scripts/                # 上机与离线测试脚本
├── data/models/            # YOLO 权重
└── docs/                   # 开发与上机文档
```
