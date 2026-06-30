# tube_ws 代码审查报告

> 审查时间：2026-07-01  
> 审查范围：全部源代码 vs `FULL_DEVELOPMENT_GUIDE.md` + `README.md` + `ON_SITE_DEBUG_GUIDE.md`  
> 评级说明：🔴 严重（上机必出问题）/ 🟠 重要（大概率出问题）/ 🟡 值得关注 / ✅ 正确

---

## 总体结论

设计分层（drivers → perception → world → planning → tasks）完整，接口协议、配置结构、FSM 状态枚举均与设计文档一致，框架质量较好。**但有 2 个严重 Bug 会导致抓取/放置时机械臂运动到错误位置，以及 1 个脚本接口错误，上机前必须修复。**

---

## 🔴 严重问题（必须修复）

### Bug 1：`_build_insert_pose` 下降方向完全反了

**文件**：`planning/motion_planner.py`，第 207–208 行

```python
# 当前代码（错误）
R = rotation_matrix_rpy(rx, ry, rz)
tip -= R @ np.array([0.0, 0.0, float(insert_mm)])
```

**问题分析**：

当末端竖直向下时（`rx ≈ π`，来自 `vertical.json`）：

```
R @ [0,0,1] ≈ [0,0,-1]（基坐标系，即 EE Z 轴指向基坐标 -Z = 向下）
tip -= R @ [0,0,insert_mm]
    ≡ tip -= [0,0,-insert_mm]（基坐标系）
    ≡ tip.z += insert_mm         ← TCP 往上走！
```

执行结果：抓取时 TCP 不是下降到试管，而是从 approach 位继续**上抬** `insert_mm`，根本不会接触试管，夹爪在空中闭合。放置同理。

**正确写法**：

```python
# 修正后
tip += R @ np.array([0.0, 0.0, float(insert_mm)])
# 等价于 tip.z -= insert_mm（TCP 向下移动）
```

**同时存在的逻辑问题**：`insert_mm` 仅为 25mm，而 approach_pose 中 TCP 已经在目标点上方 `approach_height_mm=50mm`。从 approach 到 insert 只移动 25mm，TCP 仍悬停在管口上方 25mm（50-25=25mm），无法有效抓取。

建议修正为下降 `approach_height_mm + insert_mm`，或将 approach_height 和 insert 分开处理：

```python
# 方案：从 approach_pose 完整下降到试管+insert深度
total_descent = self._approach_height_mm + float(insert_mm)
tip += R @ np.array([0.0, 0.0, total_descent])
```

---

### Bug 2：`tcp_offset_mm` 符号需要上机实测验证（高危配置项）

**文件**：`config/default.yaml`，第 26 行

```yaml
tcp_offset_mm: [0, 0, -220]   # 注释：法兰→指尖，ee -Z 方向负值
```

**问题分析**：

代码中 `tip = flange + R @ tcp_offset_mm`（`coord_transform.py` `tip_xyz_to_flange_xyz` 逻辑）。

当末端竖直向下（`rx≈π`），EE Z 轴指向基坐标 -Z 方向：

```
R @ [0, 0, -220]
= -220 × (R @ [0,0,1])
= -220 × [0,0,-1]（基坐标系）
= [0, 0, +220]（基坐标系）

→ tip = flange + [0,0,+220]（基坐标）
→ TCP 在法兰上方 220mm，而非下方
```

**但夹爪物理上应在法兰下方 220mm**。这会导致：

1. `build_approach_pose` 计算的法兰位置完全错误（偏移 440mm）
2. 所有视觉→运动的坐标换算全部偏移

**如果此值已通过实测验证可以忽略此条**（说明 Realman EE 坐标系定义与常见约定不同）。

**上机必做验证步骤**（`ON_SITE_DEBUG_GUIDE.md` Phase E 8.4 节已有说明）：

```bash
python scripts/test_motion_planner.py left.a1 right.b2
```

输出应满足：`TCP z ≈ 试管 base_z + approach_height_mm`。若不满足，修改 `tcp_offset_mm` 的符号。

---

### Bug 3：`calibrate_rack_height.py` 真机模式调用 `cam.connect()` 传参错误

**文件**：`scripts/calibrate_rack_height.py`，第 114 行

```python
# 当前代码（错误）
cam = CameraDriver(serial=cam_cfg["serial"])   # 只传了 serial
cam.connect(width=cam_cfg["width"], height=cam_cfg["height"], fps=cam_cfg["fps"])  # connect 无这些参数
```

**`CameraDriver.connect()` 签名**（`drivers/camera_driver.py` 第 60 行）：

```python
def connect(self) -> bool:    # 无参数！
```

`width/height/fps` 应在 `__init__` 时传入，不是 `connect()`。运行真机标定时会立即崩溃：

```
TypeError: connect() got unexpected keyword arguments 'width'
```

**修正**：

```python
cam = CameraDriver(
    serial=cam_cfg["serial"],
    width=cam_cfg["width"],
    height=cam_cfg["height"],
    fps=cam_cfg["fps"],
)
cam.connect()
```

---

## 🟠 重要问题（可能影响功能）

### 问题 4：`_do_pick_refine` / `_do_place_refine` 重复调用 YOLO 推理

**文件**：`tasks/pick_place_fsm.py`，第 386–393 行 & 488–494 行

```python
# 精定位（内部调用一次 detect）
result = refine_pick_slot(...)

# 绘制可视化（又调用一次 detect）
refine_vis = draw_refine_annotation(
    frame.color,
    self._refine_detector.detect(frame.color),   # ← 第二次推理
    ...
)
```

YOLO 推理被执行了**两次**，在嵌入式设备（Jetson 等）上每次约 30–100ms，会产生明显延迟。且两次结果不保证一致（若第二次检测结果与第一次不同，可视化标注与实际精定位结果不对应）。

**建议修改**：让 `refine_slot()` 返回中间的 `detections`，或在外层复用：

```python
# 修改 refine.py 返回 detections，或在 fsm 里：
detections_for_viz = result_detections  # 复用精定位时的检测
```

---

### 问题 5：`_parse_pose_6d_mm_rad` 的单位判断阈值脆弱

**文件**：`drivers/arm_driver.py`，第 188–191 行

```python
if max(abs(x), abs(y), abs(z)) < 10:
    x, y, z = x * 1000.0, y * 1000.0, z * 1000.0
```

**逻辑**：用绝对值是否 < 10 来判断"单位是米"。

**隐患**：机械臂在工作空间靠近基座时（如 x=5mm, y=3mm, z=8mm），全部绝对值 < 10，会被误判为"单位是米"→ 乘以 1000 → 坐标跳变到几千毫米。scan_pose 在 z=388mm 不受影响，但 approach 位 z 可能更低，需确认不会触发此判断。

**建议**：改为读取 dict 格式时固定乘 1000，list 格式时记录 SDK 文档确认单位，不依赖动态判断。

---

### 问题 6：`fsm_factory.py` 在安全确认前就连接机械臂

**文件**：`tasks/fsm_factory.py`，第 36–39 行；`main.py`，第 128 行

```python
# main.py：先 build（含臂连接），后 _confirm_start
fsm = build_pick_place_fsm(dry_run=False, skip_gripper=skip_gripper)  # ← 此时已连臂
# ...
if not _confirm_start():   # ← 安全确认在后
```

当 `skip_gripper=False` 时，`build_pick_place_fsm` 内部会调用 `arm.connect()`，在用户按 Enter 确认安全前机械臂已建立连接。连接本身不会运动，但不符合"先确认安全再连硬件"的原则。

**建议**：将 `arm.connect()` 移至 `_do_check_hw`，工厂只做对象初始化不做连接。

---

## 🟡 设计建议与小问题

### 建议 7：`refine.py` 歧义阈值 2mm 可能过严

**文件**：`perception/refine.py`，第 187–191 行

```python
if second.dist_xy_mm - best.dist_xy_mm < 2.0:
    raise RefineError("精定位歧义: ...")
```

24 槽密排，相邻槽间距通常 20–30mm。若光照不均导致两个相邻 tube 都在 `max_dist_xy_mm=15mm` 内，且两者到预期中心距离差 < 2mm，就会歧义报错。建议：

- 若现场频繁触发，将阈值调高（如 5mm）
- 或在精定位前将 `max_dist_xy_mm` 收窄（如改为 10mm）

---

### 建议 8：`config/default.yaml` 中 `lift_pose` 有配置但从未使用

**文件**：`config/default.yaml`，第 64 行

```yaml
lift_pose: "config/poses/lift_pose.json"   # 预留，未使用
```

`MotionPlanner` 从未加载此文件，`ON_SITE_DEBUG_GUIDE.md` 也写明"当前 retreat 用相对抬高"。文件 `config/poses/lift_pose.json` 存在但无效。建议删除或添加注释标注"预留，暂不使用"，避免混淆。

---

### 建议 9：`rack_layout.yaml` 中 `default_rack_plane_z_mm: null` 首次运行必须预先标定

**文件**：`config/rack_layout.yaml`，第 18 行

当前值为 `null`。若上机时试管架**全部空槽**且无有效深度（光照、量程问题），`estimate_z_rack` 会抛出：

```
TubeRegistryError: 无法估计 z_rack，请提供 depth 图或 --z-rack
```

**上机前必须先运行**：

```bash
python scripts/calibrate_rack_height.py
```

文档已有说明，但作为首次上机最容易忘记的步骤，建议在 `ON_SITE_DEBUG_GUIDE.md` Phase A 之前加一个 `Phase 0: 预检清单`，把这一条列入。

---

### 建议 10：`VERIFY_PICK` 和 `VERIFY_PLACE` 会触发 `scan_view` 弹窗等待按键

**文件**：`tasks/pick_place_fsm.py`，第 427–438 行 & 528–540 行

`_do_verify_pick` → `_do_scan_global()` → 弹出 `scan_view` → 等待 Enter/Space。

在自动完整 `move` 流程中，VERIFY 会**暂停等待人工确认**，不是全自动的。这是设计意图，但容易让测试者误以为程序卡死。建议在 `ON_SITE_DEBUG_GUIDE.md` 明确说明 VERIFY 阶段需要按键。

---

## ✅ 设计与实现一致性确认

| 模块 | 设计要求 | 代码状态 |
|------|---------|---------|
| 目录结构 | `docs/FULL_DEVELOPMENT_GUIDE.md` §4 | ✅ 完全一致 |
| 分层约束 | drivers→perception→world→planning→tasks | ✅ 无跨层直接调用 |
| FSM 状态枚举 | 14 个状态 | ✅ 完整实现 |
| 槽位命名 | `{side}.{row}{col}` 统一小写 | ✅ `CommandValidator.normalize_slot_id` 做了归一化 |
| 扫描驱动世界模型 | 不硬编码坐标 | ✅ `rack_layout.yaml` 无固定坐标 |
| 停稳—拍照—精定位 | wait_motion_done 后 capture | ✅ 所有 REFINE/SCAN 前先 `wait_motion_done()` |
| TCP 感知规划 | base_xyz → TCP → 法兰 | ✅ `tip_xyz_to_flange_xyz` 在路点生成时统一换算 |
| eye-in-hand 坐标链 | pixel→cam→ee→base | ✅ `pixel_to_base_mm` 三步链路正确 |
| 深度采样 | 5×5 window + median | ✅ `sample_depth_mm` 实现 |
| 空槽 Z 策略 | tube_z median - offset 或 rack_plane | ✅ `estimate_z_rack` + `_resolve_base_xyz` |
| SlotMapper 左右分离 | 最大间隙法 | ✅ `_find_split_u` |
| 1D k-means 网格映射 | k=3列, k=4行 | ✅ `_cluster_1d` 实现 |
| 夹爪 Modbus 32位步数 | high_word << 16 | ✅ `_position_to_regs` / `_regs_to_position` |
| 硬件单位统一 | mm + rad | ✅ `_pose_6d_mm_rad_to_sdk` / `_parse_pose_6d_mm_rad` |
| 相机 RGB-D 对齐 | 硬件D2C优先，软件AlignFilter备用 | ✅ `_try_enable_hw_d2c` + fallback |
| 可视化总开关 | `vision.display_enabled` | ✅ `VisionDisplay` 统一管控 |
| live 预览无 YOLO | camera_live 纯相机 | ✅ `CameraPreview._loop` 无推理 |
| dry-run 跳过夹取 | PICK_GRASP / PLACE_RELEASE 跳过 | ✅ `if self._dry_run: return True` |
| 指令格式校验 | 正则 `(left|right).[a-d][1-3]` | ✅ `SLOT_ID_RE` |
| 有效指令规则 | src=tube, dst=empty, src≠dst | ✅ `CommandValidator.validate` |
| requirements.txt | 6个依赖 | ✅ 与文档一致 |

---

## 上机前修复优先级

| 优先级 | 问题 | 影响 |
|--------|------|------|
| 🔴 P0 | Bug 1：`_build_insert_pose` 方向反了 | 抓取必失败，TCP上抬而非下降 |
| 🔴 P0 | Bug 3：`calibrate_rack_height.py` connect 传参错误 | 真机标定脚本崩溃 |
| 🔴 P0 | Bug 2：`tcp_offset_mm` 符号验证 | 所有坐标偏 440mm（若符号错） |
| 🟠 P1 | 问题 4：YOLO 推理两次 | 精定位额外延迟 |
| 🟡 P2 | 建议 9：必须先运行 calibrate | 首次扫描可能崩溃 |

---

## 修复代码片段

### 修复 Bug 1（motion_planner.py）

```python
def _build_insert_pose(
    self,
    approach_pose: tuple[float, float, float, float, float, float],
    insert_mm: float,
) -> tuple[float, float, float, float, float, float]:
    """沿末端 +Z 下降（当末端向下时即基坐标 -Z），合计移动 approach_height + insert_mm。"""
    x, y, z, rx, ry, rz = approach_pose
    tip = np.array(
        flange_xyz_to_tip_xyz((x, y, z), (rx, ry, rz), self._tcp_offset_mm),
        dtype=np.float64,
    )
    R = rotation_matrix_rpy(rx, ry, rz)
    # 修正：+= 而非 -=；总下降量 = approach_height + insert
    total_descent = self._approach_height_mm + float(insert_mm)
    tip += R @ np.array([0.0, 0.0, total_descent])   # ← 改这里
    flange_xyz = tip_xyz_to_flange_xyz(tip, (rx, ry, rz), self._tcp_offset_mm)
    return (*flange_xyz, rx, ry, rz)
```

> ⚠️ **验证方法**：`test_motion_planner.py` 打印 approach 和 insert 两个 pose，insert 的法兰 z 应该比 approach **低**（基坐标 z 更小）。

### 修复 Bug 3（calibrate_rack_height.py）

```python
# 第 111–117 行改为：
arm = ArmDriver(ip=arm_cfg["ip"], port=arm_cfg["port"])
cam = CameraDriver(
    serial=cam_cfg["serial"],
    width=cam_cfg["width"],
    height=cam_cfg["height"],
    fps=cam_cfg["fps"],
)
try:
    arm.connect()
    cam.connect()     # ← 无参数
    arm.move_p(pose_6d, speed=speed)
    ...
```

---

## 上机检测当天核查清单

以下是根据代码审查补充的上机前 checklist（在 `ON_SITE_DEBUG_GUIDE.md` 基础上）：

- [ ] `python scripts/test_arm_connect.py` — 确认位姿单位为 mm 量级（如 z=300~500mm）
- [ ] `python scripts/test_motion_planner.py left.a1 right.b2` — **打印确认**：insert z < approach z（TCP 下降而非上升）
- [ ] `python scripts/calibrate_rack_height.py` — **首次必须执行**，写入 `default_rack_plane_z_mm`
- [ ] 目测 `config/default.yaml` 中 `tcp_offset_mm` 的第三项符号，dry-run 后与示教器 TCP Z 比对
- [ ] `python main.py dry-run left.a1 right.b2 --no-gripper` — approach 位 TCP 高度合理（管口上方约 50mm）
- [ ] SCAN 弹窗出现后须按 Enter/Space 继续（不是卡死）
- [ ] VERIFY_PICK 扫描弹窗同上

---

*代码审查完成。核心框架设计优秀，主要风险集中在坐标运算方向（Bug 1）和上机前配置参数的物理验证（tcp_offset_mm 符号）。修复 Bug 1 和 Bug 3 是明天上机前的最低要求。*
