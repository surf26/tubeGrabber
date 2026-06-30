# 上机调试完整流程

RM-65B + Orbbec 336L 腕部相机 + RS485 夹爪，24 孔试管架视觉抓放。

**原则：先慢、先不夹、先短路径、每步可回退。不要跳阶段。**

---

## 目录

1. [上机前准备](#1-上机前准备)
2. [安全注意事项](#2-安全注意事项)
3. [环境与配置检查](#3-环境与配置检查)
4. [Phase A：硬件分层验收](#4-phase-a硬件分层验收)
5. [Phase B：Scan 位与视觉](#5-phase-bscan-位与视觉)
6. [Phase C：架面 Z 标定](#6-phase-c架面-z-标定)
7. [Phase D：主程序 Scan + 可视化](#7-phase-d主程序-scan--可视化)
8. [Phase E：空跑 dry-run](#8-phase-e空跑-dry-run)
9. [Phase F：夹爪试抓](#9-phase-f夹爪试抓)
10. [Phase G：完整抓放 move](#10-phase-g完整抓放-move)
11. [参数调优速查](#11-参数调优速查)
12. [常见故障排查](#12-常见故障排查)
13. [命令速查表](#13-命令速查表)
14. [建议时间线](#14-建议时间线)

---

## 1. 上机前准备

### 1.1 硬件

| 项目 | 要求 |
|------|------|
| 机械臂 | RM-65B，IP `192.168.1.18:8080`，与工控机同网段 |
| 相机 | Orbbec 336L，serial `CP84B4100090`，USB 直连稳定 |
| 夹爪 | RS485 经臂末端 port=1，Modbus RTU |
| 试管架 | 摆稳，尽量 **部分有管 + 部分空孔**（便于 z_rack 估计） |
| 显示器 | OpenCV 弹窗需要 GUI（SSH 无 DISPLAY 时关 `vision.display_enabled`） |

### 1.2 软件

```bash
conda activate surf
cd ~/tube_ws
pip install -r requirements.txt   # 依赖有变更时执行
```

确认权重存在：`data/models/best.pt`

### 1.3 建议准备的测试槽位

提前在纸上记两个槽（示例）：

- **源槽 SRC**：有试管，如 `left.a1`
- **目标槽 DST**：空孔，如 `right.b2`

后续 dry-run / move 都用这一对，直到稳定再换槽。

### 1.4 记录表（建议打印或记在笔记里）

| 步骤 | 时间 | 通过/失败 | 现象 | 改动参数 |
|------|------|-----------|------|----------|
| 臂连接 | | | | |
| 相机 | | | | |
| 夹爪 | | | | |
| scan 映射 | | | | |
| dry-run | | | | |
| move | | | | |

---

## 2. 安全注意事项

### 必须遵守

1. **急停随手可及**：示教器急停 + 软件 Ctrl+C。
2. **首次动臂低速**：`config/default.yaml` 保持 `default_speed: 20`、`approach_speed: 10`，验证通过再加快。
3. **夹爪周围无人无物**：`test_gripper.py` 和首次 insert 时人在侧方观察。
4. **不要跳阶段**：夹爪不通 → 用 `--no-gripper` 只验视觉和路径，不要硬跑 `move`。
5. **示教器冲突**：SDK 控制时确认示教器未占用/未报警。
6. **相机与臂抢资源**：scan/refine 前程序会自动 `stop_live()`；若黑屏，降低 `live_preview_fps`。

### 异常时怎么做

| 情况 | 操作 |
|------|------|
| 臂异常运动 | 急停 → Ctrl+C → 查终端最后一行 state |
| 程序卡住 | 看是否在等 `scan_view` / `refine_view` 按键（Enter/Space） |
| 相机黑屏 | 降 `vision.live_preview_fps: 10`，或暂时 `display_enabled: false` |
| refine 反复失败 | 不要继续 insert，先调 scan 位 / 光照 / YOLO |

---

## 3. 环境与配置检查

### 3.1 关键配置（`config/default.yaml`）

```yaml
arm:
  ip: "192.168.1.18"
  default_speed: 20        # 调试阶段勿加大
  approach_speed: 10

camera:
  serial: "CP84B4100090"
  depth_min_mm: 100
  depth_max_mm: 800

gripper:
  tcp_offset_mm: [0, 0, -220]   # 法兰→指尖，ee 系，需实测
  open_position / close_position  # 夹爪行程，需实测

motion:
  approach_height_mm: 50
  pick_insert_mm: 25           # 首次试抓建议从 10 起
  pick_retreat_mm: 100

registry:
  slot_match_max_dist_mm: 15   # refine XY 匹配阈值
```

### 3.2 坐标语义（调试时必须理解）

| 名称 | 含义 |
|------|------|
| 臂 `get_pose_6d()` | **法兰**位姿（mm + rad） |
| 视觉 `base_xyz` | **夹爪 TCP（抓取点）**在基坐标下的目标位置 |
| `tcp_offset_mm` | ee 系下 TCP 相对法兰的平移，`tip = flange + R @ offset` |
| 手眼 `T_ee_cam` | 相机 → **法兰**（不是指尖） |

运动规划会自动把 TCP 目标换算法兰应到的点；**视觉链不改**。

### 3.3 示教位姿文件

| 文件 | 用途 |
|------|------|
| `config/poses/scan_pose.json` | 全局 24 孔拍摄位 |
| `config/poses/left_region.json` | 左架 transit 中间点 |
| `config/poses/right_region.json` | 右架 transit 中间点 |
| `config/poses/vertical.json` | 抓取/放置末端姿态（宜竖直向下） |
| `config/poses/lift_pose.json` | 预留，当前 retreat 用相对抬高 |

**注意**：region 位姿不对会导致 transit 撞架或绕远路；scan 位不对会导致 mapping 乱。

### 3.4 架面高度

`config/rack_layout.yaml`：

```yaml
tube_above_rack_mm: 30
default_rack_plane_z_mm: null   # 需 calibrate_rack_height.py 写入
```

若全架无 tube、无有效 depth，`estimate_z_rack` 会失败。

---

## 4. Phase A：硬件分层验收

**目标：臂、相机、夹爪各自独立可用。约 15–20 分钟。**

### A.1 机械臂

```bash
python scripts/test_arm_connect.py
```

**通过标准：**

- [ ] 连接成功
- [ ] 打印 pose，xyz 为 mm 量级（与示教器对比）
- [ ] 手动挪臂后 pose 变化

**失败排查：** IP/端口、防火墙、SDK、`Robotic_Arm` 包、示教器报警。

### A.2 相机

```bash
python scripts/test_camera_capture.py
```

**通过标准：**

- [ ] 640×480 彩色图正常
- [ ] 深度图非全零，试管/架面有有效值

**失败排查：** USB、serial 不匹配、多相机冲突。

### A.3 夹爪

```bash
python scripts/test_gripper.py
```

**通过标准：**

- [ ] `setup_modbus()` 成功
- [ ] 能读 position
- [ ] open → close 循环 3 次无报错

**失败排查：**

- `gripper.device_id` 错误
- 寄存器地址与手册不符（`reg_position` 等）
- 485 未接好 / 波特率不对

**夹爪不通时：** 后续可用 `--no-gripper` 继续 Phase B–E。

---

## 5. Phase B：Scan 位与视觉

**目标：scan 位稳定、24 槽映射正确。约 20–30 分钟。**

### B.1 到 scan 位拍图

```bash
python scripts/test_scan_and_capture.py
```

**通过标准：**

- [ ] 臂稳定到达 `scan_pose.json`
- [ ] `data/captures/` 生成 `scan_*_color.png` + `scan_*_depth.png`
- [ ] 画面内左右两架、24 孔尽量完整
- [ ] 深度在架面区域有效

### B.2 坐标链验证（推荐）

```bash
python scripts/verify_coord_transform.py
```

确认像素→base 数值在合理范围（几百 mm，与现场尺寸一致）。

### B.3 离线 24 槽映射

```bash
python scripts/test_slot_mapper_offline.py \
  data/captures/scan_xxx_color.png \
  --depth data/captures/scan_xxx_depth.png
```

打开 `data/captures/slot_map_*.png`：

**通过标准：**

- [ ] 24 个槽编号与物理位置一致（左架列 3-2-1，右架列 1-2-3）
- [ ] tube / empty 与实物一致
- [ ] 编号没有整体错位或左右颠倒

**mapping 不对时先修这些，不要进 FSM：**

- 重新示教 `scan_pose.json`（高度、俯角、居中）
- 检查光照、曝光
- 调 `yolo.conf_threshold`
- 确认 `config/rack_layout.yaml` 行列顺序

### B.4 连续两次 scan 拍图（稳定性）

再跑一次 `test_scan_and_capture.py`，对比两次 slot_map：

- [ ] 同一物理槽的 uv / base_xyz 抖动 < 几 mm（记录典型值，供 refine 参考）

---

## 6. Phase C：架面 Z 标定

**当 `default_rack_plane_z_mm` 为 null，或 empty 槽 Z 明显离谱时执行。约 10 分钟。**

### 真机

```bash
python scripts/calibrate_rack_height.py
```

### 离线（用已有 scan 图）

```bash
python scripts/calibrate_rack_height.py --offline \
  data/captures/scan_xxx_color.png \
  data/captures/scan_xxx_depth.png
```

**操作：** 在窗口里点击 **空孔** 中心，多点几次，脚本写 `default_rack_plane_z_mm` 到 `rack_layout.yaml`。

**通过标准：**

- [ ] 写入的 Z 与 tube 底高度逻辑一致
- [ ] 再跑 offline slot_map，empty 槽 base_z 合理

预览不写文件：加 `--no-save`

---

## 7. Phase D：主程序 Scan + 可视化

**目标：FSM 扫描链路 + 弹窗标注。约 15 分钟。**

```bash
python main.py scan
```

或交互模式：

```bash
python main.py
# 启动后自动 connect_and_scan
```

### 可视化窗口

| 窗口名 | 何时出现 | 内容 |
|--------|----------|------|
| `camera_live` | 臂移动中 | 纯相机，**无 YOLO** |
| `scan_view` | 到 scan 位拍照后 | YOLO 框 + 24 槽编号 + 图例 |
| `refine_view` | pick/place 精定位后 | 检测 + 目标槽十字 |

**操作：** scan/refine 窗口按 **Enter / Space** 继续。

无 GUI 时：`config/default.yaml` → `vision.display_enabled: false`

### 检查项

- [ ] 终端打印 `[SCAN_GLOBAL] z_rack=... mm, tubes=N`
- [ ] `table` 或打印的状态表里 tube/empty 与实物一致
- [ ] 连扫 2 次（交互里输入 `scan`）结果稳定

---

## 8. Phase E：空跑 dry-run

**目标：全路径移动 + refine，不夹取。核心调试阶段。约 30–45 分钟。**

```bash
python main.py dry-run left.a1 right.b2 --no-gripper
```

将 `left.a1` / `right.b2` 换成你的 SRC / DST。

### 8.1 FSM 空跑顺序

```
PICK_TRANSIT → PICK_REFINE → PICK_GRASP(跳过) →
PLACE_TRANSIT → PLACE_REFINE → PLACE_RELEASE(跳过) →
SCAN_GLOBAL → DONE
```

### 8.2 逐步观察

| 阶段 | 看什么 | 合格 |
|------|--------|------|
| PICK_TRANSIT | 路点打印、live 预览、无碰撞 | 到源槽 approach 上方 |
| PICK_REFINE | `refine_view`、终端 dist_xy | 十字在 tube 上，dist_xy ≤ 15mm |
| pick approach | 法兰下降 | TCP 应在管上方约 `approach_height_mm` |
| PLACE_TRANSIT | 经 region 到目标侧 | 无撞架 |
| PLACE_REFINE | 十字对准 empty 孔 | dist_xy 合格 |
| 回 scan | scan_view | 24 槽仍正常 |

### 8.3 refine 失败

终端类似：`dist_xy=xx mm` 超限或 `[FAILED] ...`

**不要继续下降 insert。** 先修：

1. scan 位 / mapping 是否准
2. 光照是否变化
3. `registry.slot_match_max_dist_mm` 临时调到 20 试（不是长久方案）
4. `vision.refine_conf_threshold` 略降

### 8.4 TCP 偏移验证

```bash
python scripts/test_motion_planner.py left.a1 right.b2
```

看输出：

```
tcp_offset_mm (ee): (0, 0, -220)
approach 法兰 z=... mm  TCP z=... mm
```

**TCP z** 应 ≈ 试管 base_z + `approach_height_mm`。

**实测 tcp_offset 方法：**

1. 示教 `vertical.json` 姿态
2. 移到试管正上方，记录法兰 Z
3. 手动/慢速下到指尖贴管口，量高度差
4. 写入 `gripper.tcp_offset_mm` 第三项（ee -Z 方向为负）
5. 重复 dry-run 验证

---

## 9. Phase F：夹爪试抓

**dry-run 通过后再做。约 20 分钟。**

### 9.1 不要直接 full move

建议流程：

1. 再跑一遍 dry-run **不带** `--no-gripper`（仍会跳过 grasp，但会连夹爪 Modbus）
2. 确认 CHECK_HW 里夹爪 open 正常
3. 单独在 **一个已知 tube 槽** 试 insert + close

### 9.2 首次 insert 参数

```yaml
motion:
  pick_insert_mm: 10    # 从小开始，每次 +5
gripper:
  close_position: ...   # 按手册与实测
```

人在侧方，**approach_speed: 10**，随时准备急停。

### 9.3 通过标准

- [ ] 夹爪能张开到不碰邻管
- [ ] 下降后 close 能夹住试管
- [ ] retreat 上提不拖架、不甩管

### 9.4 放置试单步

对 empty 槽：insert → open → retreat，看试管是否稳定落入孔中。

---

## 10. Phase G：完整抓放 move

```bash
python main.py move left.a1 right.b2
```

### 10.1 真机完整顺序

```
PICK_TRANSIT → PICK_REFINE → PICK_GRASP → VERIFY_PICK →
PLACE_TRANSIT → PLACE_REFINE → PLACE_RELEASE → VERIFY_PLACE →
SCAN_GLOBAL → DONE
```

### 10.2 成功标准

- [ ] 实物试管从 SRC 移到 DST
- [ ] VERIFY_PICK：回 scan 后 SRC 为 `empty`
- [ ] VERIFY_PLACE：回 scan 后 DST 为 `tube`
- [ ] 无碰撞、无急停

### 10.3 VERIFY 失败含义

| 失败 | 可能原因 |
|------|----------|
| VERIFY_PICK 仍为 tube | 没夹到、夹爪行程、insert 不够、YOLO 仍看到管 |
| VERIFY_PLACE 仍为 empty | 没放下、放太深弹飞、YOLO 漏检 |
| 中途 refine 失败 | 回到 Phase E |

### 10.4 连续测试

```bash
python main.py
```

```
scan
table
move left.a1 right.b2
scan
quit
```

---

## 11. 参数调优速查

| 现象 | 优先调整 |
|------|----------|
| 24 槽错位 | `scan_pose.json`、光照、YOLO conf |
| empty 槽 Z 不对 | `calibrate_rack_height.py`、`tube_above_rack_mm` |
| refine dist_xy 大 | 光照、refine conf、`slot_match_max_dist_mm` |
| approach 太高/太低 | `approach_height_mm` |
| 指尖与管口 Z 偏差 | `gripper.tcp_offset_mm` |
| insert 顶底/夹空 | `pick_insert_mm` / `place_insert_mm` |
| 夹不紧 | `close_position`、夹爪 force 寄存器 |
| transit 撞架 | `left/right_region.json`、`transit_z_offset_mm` |
| 抓取姿态歪 | `vertical.json` 重新示教 |
| 移动太快 | `default_speed`、`approach_speed` |

---

## 12. 常见故障排查

### 12.1 连接类

```
机械臂连接失败 → IP、网线、示教器、8080 端口
相机连接失败   → serial、USB、权限
夹爪 Modbus 失败 → device_id、波特率、port=1、接线
```

### 12.2 视觉类

```
z_rack 无法估计 → 标定 rack height 或确保 scan 里有 tube/empty+depth
mapping 左右反   → rack_layout.yaml 列顺序
YOLO 漏检        → 光照、conf_threshold、重新采集训练图
深度无效         → depth_min/max_mm、距离、镜头污迹
```

### 12.3 运动类

```
move_p 报错      → 奇异点、超限位、姿态不可达 → 查 region/vertical
refine 后 approach 偏 → tcp_offset、vertical 姿态
insert 后撞架    → insert 过大、tcp_offset Z 不对
```

### 12.4 软件类

```
ImportError cv2/numpy → conda activate surf
imshow 无窗口        → DISPLAY、或关 display_enabled
程序等待不动         → 是否在等 scan_view 按键
相机 capture 卡死    → 降 live_preview_fps，确保 stop_live 后再拍
```

---

## 13. 命令速查表

```bash
conda activate surf && cd ~/tube_ws

# ── 分层硬件测试 ──
python scripts/test_arm_connect.py
python scripts/test_camera_capture.py
python scripts/test_gripper.py
python scripts/test_scan_and_capture.py
python scripts/test_move_scan.py

# ── 视觉 / 规划离线 ──
python scripts/verify_coord_transform.py
python scripts/test_slot_mapper_offline.py data/captures/xxx_color.png --depth data/captures/xxx_depth.png
python scripts/test_tube_registry.py data/captures/xxx_color.png --depth data/captures/xxx_depth.png
python scripts/test_refine_offline.py color.png depth.png left.a1
python scripts/test_motion_planner.py left.a1 right.b2
python scripts/test_command_validator.py

# ── 标定 ──
python scripts/calibrate_rack_height.py
python scripts/calibrate_rack_height.py --offline color.png depth.png

# ── 主程序 ──
python main.py scan
python main.py dry-run left.a1 right.b2 --no-gripper
python main.py dry-run left.a1 right.b2          # 连夹爪但不夹
python main.py move left.a1 right.b2
python main.py                                   # 交互模式

# ── 交互命令 ──
# scan | table | move SRC DST | dry-run SRC DST | quit
```

---

## 14. 建议时间线

| 时段 | 内容 | 最低完成标准 |
|------|------|--------------|
| 0:00–0:20 | Phase A 硬件 | 臂+相机必通；夹爪可次日 |
| 0:20–0:50 | Phase B 视觉 | slot_map 24 槽正确 |
| 0:50–1:00 | Phase C rack Z | default_rack_plane_z_mm 有值 |
| 1:00–1:20 | Phase D main scan | scan_view 与 table 正确 |
| 1:20–2:00 | Phase E dry-run | 全路径+refine 通过 |
| 2:00–2:20 | Phase F 试夹 | 单点能夹起 |
| 2:20–2:40 | Phase G move | 完整搬运 1 次成功 |
| 余量 | 换槽、重复 scan 稳定性 | — |

**一天最低目标：Phase E dry-run 通过。** 夹爪与 move 可放在第二天。

---

## 附录：架构与数据流（调试时心里有数）

```
相机 RGB-D + 法兰位姿
    ↓ YOLO + SlotMapper / Refine
base_xyz（TCP 抓取目标）
    ↓ TubeRegistry
MotionPlanner（tcp_offset → 法兰路点）
    ↓ ArmDriver.move_p
真机运动 + GripperDriver
```

- **Scan / VERIFY**：全表 24 槽刷新
- **Refine**：只更新当前 src/dst 槽
- **移动中**：live 预览，无 YOLO 追踪
- **Scan / Refine 拍照前**：自动停 live，避免抢 pipeline

---

## 附录：相关文档

- 开发阶段总览：`docs/FULL_DEVELOPMENT_GUIDE.md`
- 快速入口：`README.md`
- 主配置：`config/default.yaml`
- 手眼：`config/hand_eye.yaml`

---

*文档版本：与 tube_ws Phase 9 + 可视化 + TCP 补偿一致。*
