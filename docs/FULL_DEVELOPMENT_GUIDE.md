# 24 孔试管架夹取放置 — 保姆级全流程开发指南

> **读者**：Python 初学者，有 Realman RM-65B + 奥比 336L + 自训 YOLO  
> **导师原则**：你亲手写每一行代码；本文只告诉你**做什么、为什么、怎么验收、容易踩什么坑**。  
> **最后更新**：2026-07-01

---

## 目录

1. [先读：三个你必须记住的设计决定](#1-先读三个你必须记住的设计决定)
2. [硬件与已有资产清单](#2-硬件与已有资产清单)
3. [坐标系与数学（写代码前必须懂）](#3-坐标系与数学写代码前必须懂)
4. [推荐工程目录（先搭骨架）](#4-推荐工程目录先搭骨架)
5. [配置文件详解（第一份要写的文件）](#5-配置文件详解第一份要写的文件)
6. [分阶段实施路线（严格按顺序）](#6-分阶段实施路线严格按顺序)
7. [Phase 0：环境与项目骨架](#7-phase-0环境与项目骨架)
8. [Phase 1：驱动层 — 机械臂](#8-phase-1驱动层--机械臂)
9. [Phase 2：驱动层 — 相机](#9-phase-2驱动层--相机)
10. [Phase 3：驱动层 — 夹爪（RS485 Modbus）](#10-phase-3驱动层--夹爪rs485-modbus)
11. [Phase 4：感知层 — YOLO + 坐标变换](#11-phase-4感知层--yolo--坐标变换)
12. [Phase 5：核心 — 槽位映射 SlotMapper（最难）](#12-phase-5核心--槽位映射-slotmapper最难)
13. [Phase 6：世界模型 — 试管状态表](#13-phase-6世界模型--试管状态表)
14. [Phase 7：规划层 — 指令校验与运动路点](#14-phase-7规划层--指令校验与运动路点)
15. [Phase 8：任务层 — 完整状态机](#15-phase-8任务层--完整状态机)
16. [Phase 9：主程序与 CLI](#16-phase-9主程序与-cli)
17. [附录 A：完整状态机逐步说明（对应你的 1~12 步）](#附录-a完整状态机逐步说明对应你的-112-步)
18. [附录 B：夹爪 Modbus 操作备忘](#附录-b夹爪-modbus-操作备忘)
19. [附录 C：验收 Checklist](#附录-c验收-checklist)
20. [附录 D：故障排查表](#附录-d故障排查表)

---

## 1. 先读：三个你必须记住的设计决定

### 决定 1：试管架会移动 → **绝不能**把槽位坐标写死在配置文件里

错误做法：

```text
left.A1 的 base 坐标 = (300, 200, 50)  # 写死在 yaml 里
```

正确做法：

```text
每次 SCAN_GLOBAL：
  1. YOLO 检测所有 tube/empty
  2. SlotMapper 根据「当前这一帧的检测分布」建立 rack 坐标系
  3. 给每个检测赋 slot_id（left.A1 等）
  4. 每个槽的 base_xyz 来自「当前深度 + 当前手眼 + 当前臂姿」
```

**逻辑编号（left.A1）是稳定的；物理坐标（x,y,z）每次扫描都要重新算。**

### 决定 2：精定位永远遵循「停稳 → 拍照 → 算坐标 → 再动」

不要在机械臂运动过程中依赖深度图。  
「全程 YOLO 追踪」只作为 **短距离 Approach 阶段的 2D 辅助**，丢框就必须停、重拍。

### 决定 3：分层解耦，上层不出现 SDK 细节

| 层 | 职责 | 禁止 |
|----|------|------|
| `drivers/` | 连硬件、发指令、取原始数据 | 不知道试管、YOLO |
| `perception/` | 图像 → 检测 → 坐标 | 不知道任务状态机 |
| `world/` | 维护 24 槽状态表 | 不调用机械臂 |
| `planning/` | 校验指令、生成路点 | 不做视觉推理 |
| `tasks/` | 编排状态机 | 不算像素坐标 |

---

## 2. 硬件与已有资产清单

| 设备 | 参数 | SDK / 协议 |
|------|------|------------|
| Realman RM-65B | IP `192.168.1.18`，端口 `8080` | `pip install Robotic_Arm` |
| 奥比中光 336L | serial `CP84B4100090`，640×480 | `pyorbbecsdk` |
| 末端夹爪 | RS485 电动平行夹爪，全行程 256000 步 | Modbus RTU（经臂末端 485，`port=1`） |
| YOLO | 两类：`empty`、`tube` | 已训练权重 |
| 手眼标定 | 4×4 矩阵 `T_ee_cam` | eye-in-hand，已有 |
| 已知位姿 | 全局拍摄位、末端竖直位、左/右工作区大概位姿 | 示教或程序读取 |

### 试管架编号规则（逻辑坐标）

```text
左侧架 left          右侧架 right
列: 3  2  1          列: 1  2  3
行: a (上)            行: D (上)
    b                     C
    c                     B
    d (下)                A (下)
```

共 24 槽：left 12 + right 12。  
命名格式：`{side}.{row}{col}`，例如 `left.a1`、`right.B2`（建议代码里统一小写存储，显示时再格式化）。

---

## 3. 坐标系与数学（写代码前必须懂）

### 3.1 涉及的坐标系

```text
pixel (u, v)          图像像素
camera (Xc,Yc,Zc)     相机光学坐标系，Z 为深度
ee (末端)              机械臂法兰 / TCP
base (基座)            机械臂基坐标系
rack (架子)            每次扫描动态建立，原点在 left 架左上角之类
```

### 3.2 eye-in-hand 下：像素 + 深度 → 基坐标

**每次拍照时必须记录当前臂姿 `T_base_ee`（4×4）。**

公式（概念，你自己实现成函数）：

```text
1. (u, v, d) → 相机坐标系点 P_cam     # 用相机内参 + 深度
2. P_ee = T_ee_cam @ P_cam             # 手眼矩阵（已有 4×4）
3. P_base = T_base_ee @ P_ee           # 当前时刻正运动学
```

**你要在 config 里准备：**

- 相机内参矩阵 `K`（fx, fy, cx, cy）
- 手眼矩阵 `T_ee_cam`（4×4，明确是 eye-in-hand）
- TCP 偏移（若标定参考点是相机光心，抓取点是夹爪中心，需要 `T_ee_gripper`）

> **导师提醒**：很多人标定的是相机光心，但抓取用夹爪指尖。如果标定结果不是 gripper 尖，必须加一个固定的 TCP 偏移，否则 XY 永远差几毫米到几厘米。

### 3.3 空槽的 Z 策略

| 槽状态 | Z 怎么来 |
|--------|----------|
| 有试管 `tube` | 深度图中心点 + 手眼链 → `z_measured` |
| 空槽 `empty` | 使用 **rack 平面高度** `z_rack`（见下） |

**rack 平面高度 `z_rack` 的计算（写 utils）：**

1. 优先：取当前帧所有 `tube` 检测的 base Z，统计 median 作为管口高度参考
2. 减去试管插入深度偏移（配置项 `tube_above_rack_mm`）
3. 或：取所有 `empty` 检测中有有效深度的点，median 作为架面
4. 若都不够：用上一次扫描的 `z_rack` 或配置默认值

**放置空槽时**：XY 用精定位结果，Z 用 `z_rack + place_insert_mm`（插入深度，配置项）。

---

## 4. 推荐工程目录（先搭骨架）

在 `tube_ws/` 下创建：

```text
tube_ws/
├── config/
│   ├── default.yaml          # 主配置（IP、阈值、位姿路径）
│   ├── camera_intrinsics.yaml
│   ├── hand_eye.yaml         # 4x4 矩阵
│   └── rack_layout.yaml      # 行列数、镜像规则（不是固定坐标！）
├── data/
│   ├── calib/                # 标定文件备份
│   ├── captures/             # 调试存图
│   └── models/               # YOLO 权重
├── drivers/
│   ├── __init__.py
│   ├── arm_driver.py
│   ├── camera_driver.py
│   └── gripper_driver.py
├── perception/
│   ├── __init__.py
│   ├── yolo_detector.py
│   ├── coord_transform.py
│   └── slot_mapper.py
├── world/
│   ├── __init__.py
│   └── tube_registry.py
├── planning/
│   ├── __init__.py
│   ├── command_validator.py
│   └── motion_planner.py
├── tasks/
│   ├── __init__.py
│   └── pick_place_fsm.py
├── utils/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── rack_height.py
│   └── logging_utils.py
├── tests/
│   ├── test_slot_mapper_offline.py   # 离线测映射
│   └── test_coord_transform.py
├── docs/
│   └── FULL_DEVELOPMENT_GUIDE.md     # 本文件
├── main.py
├── requirements.txt
└── README.md
```

**第一步**：只创建空文件 + `__init__.py`，不要一次写满。

---

## 5. 配置文件详解（第一份要写的文件）

### 5.1 `config/default.yaml` 模板

```yaml
arm:
  ip: "192.168.1.18"
  port: 8080
  # 运动参数
  default_speed: 20          # 按 SDK 单位填，先慢后快
  approach_speed: 10       # 靠近试管架时更慢

camera:
  serial: "CP84B4100090"
  width: 640
  height: 480
  fps: 30
  depth_min_mm: 100
  depth_max_mm: 800

gripper:
  modbus_port: 1             # 末端 485
  device_id: 1               # 夹爪从站地址，按你实际读到的改
  baudrate: 115200           # 按夹爪手册改
  full_stroke: 256000
  open_position: 256000      # 张开目标步数 — 需实测标定
  close_position: 0          # 闭合目标步数 — 需实测标定
  # 寄存器地址 — 以你夹爪手册为准，下面为占位
  reg_position: 36
  reg_speed: 38
  reg_force: 40
  reg_command: 43

yolo:
  model_path: "data/models/best.pt"
  conf_threshold: 0.5
  iou_threshold: 0.45
  classes:
    0: empty
    1: tube

vision:
  refine_conf_threshold: 0.6   # 精定位时可更严

motion:
  approach_height_mm: 50       # 精定位前停在目标上方 5cm
  pick_insert_mm: 25           # 抓取下降深度（实测调整）
  pick_retreat_mm: 100         # 夹取后上提 1cm
  place_insert_mm: 20          # 放置插入深度
  place_retreat_mm: 100

poses:
  scan_pose: "config/poses/scan_pose.json"       # 6D 位姿或 joint
  vertical_ee_pose: "config/poses/vertical.json"
  left_region_pose: "config/poses/left_region.json"
  right_region_pose: "config/poses/right_region.json"

paths:
  transit_z_offset_mm: 80      # 路点中间抬高量

rack:
  rows: 4
  cols: 3
  sides: ["left", "right"]
  # 镜像：left 列顺序 3,2,1；right 列顺序 1,2,3
  left_col_order: [3, 2, 1]
  right_col_order: [1, 2, 3]
  left_row_order: ["a", "b", "c", "d"]   # 上到下
  right_row_order: ["d", "c", "b", "a"]  # 对应物理 D,C,B,A

registry:
  slot_match_max_dist_mm: 15   # 精定位时检测中心与预期槽的最大偏差
```

### 5.2 `config/hand_eye.yaml`

```yaml
# eye-in-hand: 将相机坐标系下的点变换到末端坐标系
T_ee_cam:
  - [r11, r12, r13, tx]
  - [r21, r22, r23, ty]
  - [r31, r32, r33, tz]
  - [0,    0,    0,    1 ]
```

### 5.3 `config/camera_intrinsics.yaml`

```yaml
fx: 525.0
fy: 525.0
cx: 320.0
cy: 240.0
distortion: [0, 0, 0, 0, 0]   # 若已知
```

---

## 6. 分阶段实施路线（严格按顺序）

```text
Phase 0  环境 + 空目录
Phase 1  ArmDriver      → 能连上、读 pose、动一下
Phase 2  CameraDriver  → 能取 RGB + Depth 对齐帧
Phase 3  GripperDriver → 能张开/闭合/读位置
Phase 4  YOLO + CoordTransform → 单点像素→base 验证
Phase 5  SlotMapper    → 离线照片映射 24 槽（最关键）
Phase 6  TubeRegistry  → 状态表 CRUD
Phase 7  Validator + MotionPlanner
Phase 8  PickPlaceFSM  → 完整流程
Phase 9  main.py CLI
```

**不要跳阶段。** Phase 5 不通，后面全是盲走。

---

## 7. Phase 0：环境与项目骨架

### 你要做的事

1. 创建第 4 节的目录结构（空文件即可）
2. 写 `requirements.txt`：

```text
numpy
opencv-python
pyyaml
ultralytics          # 或你 YOLO 用的框架
pyorbbecsdk
Robotic_Arm
```

3. 创建虚拟环境并安装
4. 把 YOLO 权重、手眼矩阵、内参、示教位姿放进 `data/` 和 `config/`

### 验收标准

- [ ] `python -c "import numpy; ..."` 全部 import 成功
- [ ] 目录结构与上一节一致
- [ ] `config/default.yaml` 填好真实 IP 和 serial

---

## 8. Phase 1：驱动层 — 机械臂

### 文件：`drivers/arm_driver.py`

### 你要实现的类：`ArmDriver`

**职责**：封装 `Robotic_Arm` SDK，对外只暴露干净接口。

**建议接口（你自己写实现）：**

```python
class ArmDriver:
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    def get_pose(self) -> np.ndarray:
        """返回 4x4 T_base_ee，或返回 (x,y,z,rx,ry,rz) 由上层转换"""

    def move_j(self, joint_angles: list, speed: int) -> bool: ...
    def move_p(self, pose_6d: list, speed: int) -> bool: ...
    def move_p_pose_matrix(self, T: np.ndarray, speed: int) -> bool: ...

    def wait_motion_done(self, timeout: float = 30.0) -> bool: ...
    def stop(self) -> None: ...
```

### 实现步骤（按顺序写）

1. **`connect()`**  
   - 实例化 SDK 对象，连 `192.168.1.18:8080`  
   - 失败抛明确异常或返回 False，打日志

2. **`get_pose()`**  
   - 读当前 TCP 位姿  
   - **单元测试**：连上后打印 pose，手动挪臂，看数值是否变化

3. **`move_j()` 到已知安全位**  
   - 先极慢速度，确认方向对

4. **`wait_motion_done()`**  
   - SDK 若有「到位」查询就用；没有就 poll pose 变化小于阈值

5. **封装「到 scan_pose」**（可先写在 main 测试脚本，后迁到 MotionPlanner）

### 验收标准

- [ ] 连接/断开 10 次无泄漏
- [ ] `get_pose()` 与示教器显示误差在可接受范围
- [ ] 能稳定移动到 `scan_pose` 并重复（重复精度你记录一下，后面精定位要用）

### 常见坑

- 单位：mm vs m，deg vs rad — **读 SDK 文档，在 driver 层统一成 mm + rad**
- 没有 wait 到位就拍照 → 坐标全漂

---

## 9. Phase 2：驱动层 — 相机

### 文件：`drivers/camera_driver.py`

### 你要实现的类：`CameraDriver`

```python
class Frame:
    color: np.ndarray      # HxWx3 BGR
    depth: np.ndarray      # HxW uint16，单位 mm
    timestamp: float

class CameraDriver:
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def capture(self) -> Frame: ...
    def get_intrinsics(self) -> dict: ...
```

### 实现步骤

1. 按 serial `CP84B4100090` 打开设备
2. 配置 640×480，开启 depth 流
3. **`capture()` 必须 RGB 与 Depth 对齐**（同一硬件帧或软件对齐）
4. 深度无效值（0）要在 perception 层过滤，driver 原样返回
5. 调试：存一张 `color.png` + `depth.png` 到 `data/captures/`

### 验收标准

- [ ] 连续 100 帧不崩溃
- [ ] 深度在试管位置有合理值（不是大片 0）
- [ ] 内参与 `camera_intrinsics.yaml` 一致（或从 SDK 读取写入 yaml）

---

## 10. Phase 3：驱动层 — 夹爪（RS485 Modbus）

### 文件：`drivers/gripper_driver.py`

### 背景

RM-65B 末端 RS485 通过机械臂 JSON 协议转发 Modbus RTU（`port=1`）。  
参考：[睿尔曼 Modbus JSON 协议](https://develop.realman-robotics.com/robot/json/modbus/)

### 你要实现的类：`GripperDriver`

```python
class GripperDriver:
    def __init__(self, arm: ArmDriver, config: dict): ...
    def setup_modbus(self) -> bool:
        """set_modbus_mode port=1"""

    def open(self, wait: bool = True) -> bool: ...
    def close(self, wait: bool = True) -> bool: ...
    def get_position(self) -> int: ...
    def is_ready(self) -> bool: ...
```

### 实现步骤

1. **首次调试（用官方工具或 Python 调 JSON）**  
   - `set_modbus_mode`：`port=1`, `baudrate=`, `timeout=`  
   - 读夹爪站号 / 状态寄存器，确认 `device_id`  
   - 读当前位置寄存器，确认 `reg_position=36` 等是否与手册一致

2. **标定开合位置**  
   - 全行程 256000  
   - 手动试：写到 `open_position` 能张开且不顶限位  
   - 写到 `close_position` 能夹住试管但不打滑（记录数值到 yaml）

3. **封装 `open()` / `close()`**  
   - 写目标位置寄存器  
   - poll 当前位置直到稳定或超时

4. **与臂联动注意**  
   - Modbus 配置在臂重启后可能丢失 → 在 `connect()` 后总是 `setup_modbus()`

### 验收标准

- [ ] 连续 open/close 20 次成功
- [ ] 夹空试管不滑落（物理测试）
- [ ] 超时/无响应有明确报错

### 附录 B 有更多寄存器备忘

---

## 11. Phase 4：感知层 — YOLO + 坐标变换

### 11.1 `perception/yolo_detector.py`

```python
@dataclass
class Detection:
    class_name: str       # "empty" | "tube"
    confidence: float
    bbox: tuple           # x1,y1,x2,y2
    center_uv: tuple      # u,v

class YoloDetector:
    def load(self, model_path: str) -> None: ...
    def detect(self, bgr_image: np.ndarray) -> list[Detection]: ...
    def draw(self, image, detections) -> np.ndarray: ...  # 调试可视化
```

**你要做：**

1. 加载 `best.pt`
2. 过滤 `conf < threshold`
3. 计算每个框中心 `(u,v)`

### 11.2 `perception/coord_transform.py`

```python
def pixel_depth_to_camera(u, v, depth_mm, K) -> np.ndarray:
    """返回相机系 3D 点 (3,)"""

def camera_to_base(P_cam, T_ee_cam, T_base_ee) -> np.ndarray:
    """返回 base 系 (3,)"""

def pixel_to_base(u, v, depth_mm, K, T_ee_cam, T_base_ee) -> np.ndarray | None:
    """depth 无效返回 None"""
```

### 11.3 本阶段验证脚本（你自己写 `scripts/verify_one_point.py`）

**流程：**

1. 臂到 `scan_pose`，停稳
2. 拍照
3. 跑 YOLO，手动选一个 tube 框
4. 取中心深度（见下「深度采样策略」）
5. 读 `get_pose()`，算 base 坐标
6. 臂移到该点上方 50mm（先不夹）
7. 再拍，看偏差多少

### 深度采样策略（写进 utils 或 coord_transform）

不要只取 1 像素：

```text
以 (u,v) 为中心取 5x5 或 7x7 窗口
去掉 depth=0 和离群值
取 median 作为最终深度
```

### 验收标准

- [ ] 单点重复 10 次，XY 标准差 < 你设定的阈值（建议先 < 3mm，达不到就查标定）
- [ ] 可视化：图上画框 + 标 depth 值

---

## 12. Phase 5：核心 — 槽位映射 SlotMapper（最难）

### 文件：`perception/slot_mapper.py`

### 问题定义

**输入**：一帧全局图的 `list[Detection]`（每个有 center_uv, class, confidence）  
**输出**：`dict[slot_id, SlotObservation]`

```python
@dataclass
class SlotObservation:
    slot_id: str              # "left.a1"
    klass: str                # "empty" | "tube" | "unknown"
    confidence: float
    pixel_uv: tuple | None
    base_xyz: tuple | None    # 若有深度则填
    z_source: str             # "measured" | "rack_plane" | "missing"
```

### 算法思路（保姆级）

#### Step 1：分离左右架

```text
所有检测中心点 { (u,v) }
按 u 坐标排序，找最大间隙 → 分成 left 簇 和 right 簇
（全局图里两架水平分开，中间有空隙）
```

若自动分失败：用配置 `split_u_threshold` 或 left/right 的 u 范围。

#### Step 2：每簇内建 2D 网格

对每个簇：

```text
1. 将 (u,v) 投影到 2D（或用 base_xy 若已算）
2. 对 u 方向做 1D 聚类 → 3 列
3. 对 v 方向做 1D 聚类 → 4 行
4. 得到 3×4 = 12 个网格中心
```

聚类可用：

- 简单 k-means (k=3, k=4)
- 或排序后等分（若透视不大且检测齐）

#### Step 3：检测 ↔ 网格匹配

```text
每个 detection 中心 → 找最近的网格中心
若距离 > max_assign_dist → 标记 orphan（不参与）
每个网格最多保留 1 个 detection（取 confidence 最高）
```

#### Step 4：赋 slot_id

```text
left 簇：行从上到下 → a,b,c,d；列按 left_col_order [3,2,1]
right 簇：行从上到下 → d,c,b,a 对应物理 D,C,B,A；列按 right_col_order [1,2,3]
```

**命名统一小写**：`left.a3`, `right.b2`

#### Step 5：补全缺失槽

24 格中没检测到的：

- 默认 `klass="unknown"`，或根据邻居推断（后期再加）
- 第一次版本：**缺失就标 unknown，校验时不允许作为源或目标**

### 离线测试（必须先做）

`tests/test_slot_mapper_offline.py`：

1. 读 `data/captures/scan_001.png`
2. 跑 YOLO
3. SlotMapper 输出 24 项
4. 可视化：每个槽画编号

### 验收标准

- [ ] 24 槽全部有 slot_id（允许 unknown）
- [ ] 与人工标注对比，正确率 > 95%（10 张图统计）
- [ ] 左右不互换、列顺序不镜像错

### 常见坑

- 某行 YOLO 漏检 → 网格歪 → **用期望 3×4 几何约束**（RANSAC 拟合网格）
- 透视导致行间距不等 → 用聚类比等分更稳
- **架子移动不影响 SlotMapper**，因为它每帧重新聚类

---

## 13. Phase 6：世界模型 — 试管状态表

### 文件：`world/tube_registry.py`

### 你要实现的类：`TubeRegistry`

```python
@dataclass
class SlotState:
    slot_id: str
    klass: str
    confidence: float
    pixel_uv: tuple | None
    base_xyz: tuple | None
    z_source: str
    updated_at: float

class TubeRegistry:
    def __init__(self, all_slot_ids: list[str]): ...
    def update_from_scan(self, observations: dict[str, SlotObservation], z_rack: float): ...
    def update_slot(self, slot_id: str, **kwargs): ...
    def get(self, slot_id: str) -> SlotState: ...
    def find_empty_slots(self) -> list[str]: ...
    def find_tube_slots(self) -> list[str]: ...
    def to_table_str(self) -> str: ...   # 打印给用户看
```

### `update_from_scan` 逻辑

```text
对每个 slot_id:
  若有观测 → 更新 klass, conf, uv, xyz
  若 klass==empty → base_z = z_rack, z_source=rack_plane
  若 klass==tube  → base_z 来自 measured
  若无观测 → klass=unknown
```

### 验收标准

- [ ] 扫描后能打印完整 24 行表格
- [ ] `find_empty_slots()` / `find_tube_slots()` 正确

---

## 14. Phase 7：规划层 — 指令校验与运动路点

### 14.1 `planning/command_validator.py`

```python
@dataclass
class MoveCommand:
    src: str   # "left.a1"
    dst: str   # "right.b2"

class CommandValidator:
    def parse(self, text: str) -> MoveCommand:
        """解析 'left.a1 right.b2'"""

    def validate(self, cmd: MoveCommand, registry: TubeRegistry) -> tuple[bool, str]:
        """
        规则：
        1. src/dst 格式合法且在 24 槽内
        2. src 必须是 tube（非 unknown/empty）
        3. dst 必须是 empty
        4. 若没有任何 empty → 拒绝
        5. src != dst
        返回 (ok, reason)
        """
```

### 14.2 `planning/motion_planner.py`

**职责**：生成路点，不包含 SDK 调用。

```python
class MotionPlanner:
    def __init__(self, config, arm_poses): ...

    def plan_to_scan(self) -> list[Waypoint]: ...
    def plan_transit_to_slot(self, slot: SlotState, side: str) -> list[Waypoint]:
        """scan → left/right_region → slot 上方 approach_height"""

    def build_approach_pose(self, base_xyz, approach_height_mm) -> np.ndarray:
        """竖直末端姿态 + 目标上方高度"""

    def build_retreat_pose(self, current_pose, retreat_mm) -> np.ndarray: ...
```

### 路点策略（保姆级）

```text
任意移动建议分三段：
  1. 抬高到 safe Z（当前 Z + transit_z_offset 或 fixed safe_z）
  2. 平移到目标上方
  3. 下降到 approach_height（50mm）

左架 / 右架：
  先到 left_region_pose 或 right_region_pose 附近，再进近
  避免从右架扫直接插到左架低处
```

### 验收标准

- [ ] 非法指令全部被拒，reason 可读
- [ ] 路点序列在仿真/示教器上看无碰架风险（你肉眼检查）

---

## 15. Phase 8：任务层 — 完整状态机

### 文件：`tasks/pick_place_fsm.py`

### 状态枚举

```python
class State(Enum):
    INIT = "init"
    CHECK_HW = "check_hw"
    SCAN_GLOBAL = "scan_global"
    WAIT_CMD = "wait_cmd"
    VALIDATE_CMD = "validate_cmd"
    PICK_TRANSIT = "pick_transit"
    PICK_REFINE = "pick_refine"
    PICK_GRASP = "pick_grasp"
    VERIFY_PICK = "verify_pick"
    PLACE_TRANSIT = "place_transit"
    PLACE_REFINE = "place_refine"
    PLACE_RELEASE = "place_release"
    VERIFY_PLACE = "verify_place"
    DONE = "done"
    FAILED = "failed"
```

### 类：`PickPlaceFSM`

```python
class PickPlaceFSM:
    def __init__(self, arm, camera, gripper, detector, mapper, registry, planner, validator, config): ...

    def run_once(self) -> bool:
        """执行一次完整 pick-place，或 run_interactive 循环等命令"""

    def step(self): ...  # 单步推进，便于调试
```

### 每个状态做什么（实现 checklist）

| 状态 | 动作 |
|------|------|
| CHECK_HW | arm.connect, camera.connect, gripper.setup_modbus, gripper.open |
| SCAN_GLOBAL | move scan_pose → capture → yolo → mapper → 算 z_rack → registry.update |
| WAIT_CMD | input() 或 CLI |
| VALIDATE_CMD | validator.validate |
| PICK_TRANSIT | 按 planner 移动到 src 上方 approach_height |
| PICK_REFINE | 停稳 → capture → yolo 找 src 附近框 → depth → 更新 registry → 微调 XY |
| PICK_GRASP | open → 降 pick_insert → close → 升 pick_retreat |
| VERIFY_PICK | 回 scan → 重扫 → 检查 src 是否变 empty |
| PLACE_TRANSIT | 到 dst 上方 |
| PLACE_REFINE | 停稳 → capture → yolo（empty 框）→ XY；Z 用 z_rack |
| PLACE_RELEASE | 降 place_insert → open → 升 retreat |
| VERIFY_PLACE | 回 scan → 检查 dst 是否变 tube |

### 「实时 YOLO 追踪」怎么加（可选，Phase 8 完成后再加）

在 `PICK_TRANSIT` 中：

```text
loop:
  capture (非阻塞或短曝光)
  yolo 找 class=tube 且距离预期 UV 最近的框
  若找到 → 微调臂 XY（小步）
  若连续 N 帧丢失 → 停止，进入 PICK_REFINE 或 FAILED
```

**第一版可以不做追踪**，先靠 PICK_REFINE 单次精定位。

### 验收标准

- [ ] 完整流程空跑（不夹，只移动）走通
- [ ] 夹取成功 3/3
- [ ] 故意夹空，VERIFY_PICK 能报错

---

## 16. Phase 9：主程序与 CLI

### 文件：`main.py`

```text
启动
  → 加载 config
  → 初始化各模块
  → FSM.run_interactive():
       SCAN_GLOBAL
       打印 registry.to_table_str()
       while True:
         用户输入: left.a1 right.b2
         validate → 执行 pick_place → 打印结果
         重新 SCAN 或询问是否继续
```

### 建议 CLI 命令

```text
scan              重新扫描
move left.a1 right.b2   执行搬运
table             打印当前表
quit              退出
```

### 验收标准

- [ ] 非程序员能按提示操作
- [ ] 任何失败有清晰中文 reason

---

## 附录 A：完整状态机逐步说明（对应你的 1~12 步）

| 你的步骤 | 对应状态 | 补充说明 |
|----------|----------|----------|
| 1 检查连接 | CHECK_HW | 臂+相机+夹爪 modbus 三者 |
| 2 全局拍摄+YOLO | SCAN_GLOBAL | 必须 wait_motion_done 后再 capture |
| 3 映射编号 | SCAN_GLOBAL 内 SlotMapper | 见 Phase 5 |
| 4 深度+手眼 | SCAN_GLOBAL / REFINE | 同步 get_pose |
| 5 维护表 | TubeRegistry | |
| 6 等用户指令 | WAIT_CMD | |
| 7 校验 | VALIDATE_CMD | tube→empty |
| 8 实时追踪 | PICK_TRANSIT（可选） | 第一版可跳过 |
| 9 上方 5cm 精定位 | PICK_REFINE | 停稳拍照 |
| 10 夹取 | PICK_GRASP | open→down→close→up |
| 11 验证源位 empty | VERIFY_PICK | 失败则 ABORT，整程序重启或 rescan |
| 12 放置+验证 | PLACE_* + VERIFY_PLACE | 空槽 Z 用 rack 平面 |

---

## 附录 B：夹爪 Modbus 操作备忘

> **重要**：以下寄存器地址需与你手头夹爪手册核对。你提到全行程 256000、寄存器 36/38/40/43，与官方 DRV42 示例（0x002B=43 写位置）可能编号方式不同。

### 初始化（每次程序启动）

通过臂 JSON API（`Robotic_Arm` 或 socket 8080）：

```json
{"command":"set_modbus_mode","port":1,"baudrate":115200,"timeout":2}
```

### 读位置（示例）

```json
{"command":"read_multiple_holding_registers","port":1,"address":36,"num":2,"device":1}
```

### 写目标位置（示例）

```json
{"command":"write_registers","port":1,"address":43,"num":1,"data":[高字节, 低字节],"device":1}
```

### 你要自己记录的标定表

| 动作 | 寄存器值 | 备注 |
|------|----------|------|
| 完全张开 | | |
| 夹紧试管 | | |
| 读取就绪状态 | | |

---

## 附录 C：验收 Checklist

### 里程碑 M1：硬件互通

- [ ] Arm 连接、读 pose、动
- [ ] Camera RGB+Depth
- [ ] Gripper open/close

### 里程碑 M2：视觉几何

- [ ] 单点 pixel→base 重复精度 OK
- [ ] 手眼 + TCP 偏移验证

### 里程碑 M3：槽位映射

- [ ] 10 张图 SlotMapper 正确率 > 95%
- [ ] 可视化 24 槽编号

### 里程碑 M4：空跑流程

- [ ] FSM 全流程移动无碰撞
- [ ] 校验逻辑正确

### 里程碑 M5：实机夹取

- [ ] 3 次连续 pick-place 成功
- [ ] 失败路径 VERIFY 能检出

---

## 附录 D：故障排查表

| 现象 | 可能原因 | 查哪里 |
|------|----------|--------|
| base 坐标整体偏移 | 手eye错 / 没读 pose / 单位错 | coord_transform, get_pose 时机 |
| 左右槽反了 | 镜像规则错 | rack_layout.yaml, SlotMapper |
| 深度为 0 | 反光、超量程 | depth_min/max, median 滤波 |
| 夹取后仍显示 tube | 没夹到 / YOLO 漏检 | VERIFY 阈值、物理夹爪力度 |
| 放置偏 | 空槽 Z 错 / 仅 XY 精定位 | rack_height utils |
| 架子挪了后全偏 | 用了旧固定坐标 | 必须每次 SCAN 重算 |
| 夹爪无响应 | modbus 未配置 / device_id 错 | gripper setup_modbus |
| 移动中坐标跳 | 未停稳拍照 | wait_motion_done |

---

## 你接下来立刻要做的第一件事

1. 按 **第 4 节** 创建空目录  
2. 按 **第 5 节** 写好 `config/default.yaml`（填真实 IP、serial、模型路径）  
3. 完成 **Phase 1** 的 `ArmDriver.connect()` + `get_pose()`  

写完后把 `arm_driver.py` 贴给我（或说明卡在哪一步），我帮你 review 接口和边界，**不会替你写完整实现**。

---

*文档版本：v1.0 | 项目：tube_ws*
