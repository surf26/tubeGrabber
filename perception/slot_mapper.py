"""YOLO 检测 → 24 孔试管架槽位映射。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # 匈牙利指派可选依赖；缺失时回退贪心
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - 环境相关
    _HAS_SCIPY = False

from perception.coord_transform import (
    CoordTransformError,
    pixel_to_base_mm,
    pixel_uv_to_rack_plane_mm,
)
from perception.yolo_detector import Detection
from utils.config_loader import load_yaml


class SlotMapperError(RuntimeError):
    """槽位映射失败。"""


@dataclass
class SlotObservation:
    slot_id: str
    klass: str = "unknown"  # tube | empty | unknown
    confidence: float = 0.0
    pixel_uv: tuple[float, float] | None = None
    base_xyz: tuple[float, float, float] | None = None  # 抓取目标（夹爪 TCP），基坐标 mm
    z_source: str = "missing"  # measured | rack_plane | missing | unknown


@dataclass
class _DetPoint:
    detection: Detection
    u: float
    v: float
    side: str = ""
    x_mm: float | None = None  # 反投影到架面的度量 X（mm）
    y_mm: float | None = None  # 反投影到架面的度量 Y（mm）


class SlotMapper:
    def __init__(
        self,
        rack_config: dict[str, Any] | None = None,
        *,
        max_assign_dist_px: float = 55.0,
        image_width: int = 640,
        use_grid_assignment: bool = True,
    ) -> None:
        self._rack = rack_config or load_yaml("config/rack_layout.yaml")
        self._max_assign_dist_px = max_assign_dist_px
        self._image_width = image_width

        self._left_rows = [str(r) for r in self._rack["left_row_order"]]
        self._left_cols = [int(c) for c in self._rack["left_col_order"]]
        self._right_rows = [str(r) for r in self._rack["right_row_order"]]
        self._right_cols = [int(c) for c in self._rack["right_col_order"]]
        self._tube_above_rack_mm = float(self._rack.get("tube_above_rack_mm", 30.0))
        default_z = self._rack.get("default_rack_plane_z_mm")
        self._default_rack_z_mm = float(default_z) if default_z is not None else None

        # --- 度量栅格指派参数 ---
        self._use_grid_assignment = bool(use_grid_assignment)
        pitch_cfg = self._rack.get("slot_pitch_mm") or {}
        self._pitch_col = _opt_float(pitch_cfg.get("col"))
        self._pitch_row = _opt_float(pitch_cfg.get("row"))
        self._min_points_for_grid = 4
        self._grid_fit_max_resid_frac = 0.35
        self._assign_gate_frac = 0.5
        self._last_theta: dict[str, float] = {"left": 0.0, "right": 0.0}

    @property
    def last_rack_theta(self) -> dict[str, float]:
        """上次 map() 每侧估计的架面旋转角 θ（rad）。"""
        return dict(self._last_theta)

    def all_slot_ids(self) -> list[str]:
        ids: list[str] = []
        for row in self._left_rows:
            for col in self._left_cols:
                ids.append(f"left.{row}{col}")
        for row in self._right_rows:
            for col in self._right_cols:
                ids.append(f"right.{row}{col}")
        return ids

    def map(
        self,
        detections: list[Detection],
        depth: np.ndarray | None = None,
        *,
        K: np.ndarray | None = None,
        dist: np.ndarray | None = None,
        T_ee_cam: np.ndarray | None = None,
        T_base_ee: np.ndarray | None = None,
        depth_min_mm: float = 100,
        depth_max_mm: float = 800,
        z_rack_override: float | None = None,
    ) -> dict[str, SlotObservation]:
        """
        将 YOLO 检测映射到 24 个逻辑槽位。

        优先在架面度量系下做旋转感知的二维栅格一对一指派（对整体旋转鲁棒）；
        缺标定或点太少时回退到旧的逐轴一维聚类。
        tube：深度图测 base_xyz；empty：pixel_uv + z_rack 平面反投影 XY。
        """
        self._last_theta = {"left": 0.0, "right": 0.0}
        result = {slot_id: SlotObservation(slot_id=slot_id) for slot_id in self.all_slot_ids()}
        if not detections:
            return result

        have_calib = (
            K is not None
            and dist is not None
            and T_ee_cam is not None
            and T_base_ee is not None
        )
        z_assign = self._resolve_assign_z(
            detections, depth, K, dist, T_ee_cam, T_base_ee,
            z_rack_override, depth_min_mm, depth_max_mm,
        )

        points = [_DetPoint(det, det.center_uv[0], det.center_uv[1]) for det in detections]
        if have_calib and z_assign is not None:
            for p in points:
                p.x_mm, p.y_mm = self._metric_xy(
                    p.u, p.v, z_assign, K, dist, T_ee_cam, T_base_ee
                )

        split_u = _find_split_u([p.u for p in points], self._image_width)
        left_pts = [p for p in points if p.u < split_u]
        right_pts = [p for p in points if p.u >= split_u]

        self._assign_one_side(
            left_pts, "left", result, depth, K, dist, T_ee_cam, T_base_ee,
            z_assign, depth_min_mm, depth_max_mm,
        )
        self._assign_one_side(
            right_pts, "right", result, depth, K, dist, T_ee_cam, T_base_ee,
            z_assign, depth_min_mm, depth_max_mm,
        )

        if have_calib:
            fill_z = z_rack_override if z_rack_override is not None else z_assign
            self._fill_empty_on_rack_plane(
                result, K, dist, T_ee_cam, T_base_ee, z_rack=fill_z
            )

        return result

    def _resolve_assign_z(
        self,
        detections: list[Detection],
        depth: np.ndarray | None,
        K: np.ndarray | None,
        dist: np.ndarray | None,
        T_ee_cam: np.ndarray | None,
        T_base_ee: np.ndarray | None,
        z_rack_override: float | None,
        depth_min_mm: float,
        depth_max_mm: float,
    ) -> float | None:
        """确定用于度量反投影的架面 Z：override 优先 → 本帧 tube 深度估计 → 默认标定。"""
        if z_rack_override is not None:
            return float(z_rack_override)
        if (
            depth is None
            or K is None
            or dist is None
            or T_ee_cam is None
            or T_base_ee is None
        ):
            return self._default_rack_z_mm
        zs: list[float] = []
        for det in detections:
            if det.class_name == "empty":
                continue
            try:
                p_base, _ = pixel_to_base_mm(
                    det.center_uv[0], det.center_uv[1], depth, K, dist,
                    T_ee_cam, T_base_ee,
                    depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm,
                )
            except CoordTransformError:
                continue
            zs.append(float(p_base[2]))
        if zs:
            return float(np.median(zs)) - self._tube_above_rack_mm
        return self._default_rack_z_mm

    def _metric_xy(
        self,
        u: float,
        v: float,
        z_assign: float,
        K: np.ndarray,
        dist: np.ndarray,
        T_ee_cam: np.ndarray,
        T_base_ee: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """像素反投影到 z_assign 架面，得到度量 XY（mm）。"""
        try:
            x, y, _ = pixel_uv_to_rack_plane_mm(
                u, v, float(z_assign), K, dist, T_ee_cam, T_base_ee
            )
            return float(x), float(y)
        except CoordTransformError:
            return None, None

    def _assign_one_side(
        self,
        points: list[_DetPoint],
        side: str,
        result: dict[str, SlotObservation],
        depth: np.ndarray | None,
        K: np.ndarray | None,
        dist: np.ndarray | None,
        T_ee_cam: np.ndarray | None,
        T_base_ee: np.ndarray | None,
        z_assign: float | None,
        depth_min_mm: float,
        depth_max_mm: float,
    ) -> None:
        if not points:
            self._last_theta[side] = 0.0
            return

        metric_pts = [p for p in points if p.x_mm is not None and p.y_mm is not None]
        if (
            self._use_grid_assignment
            and z_assign is not None
            and len(metric_pts) >= self._min_points_for_grid
        ):
            if self._grid_assign(
                metric_pts, side, result, depth, K, dist, T_ee_cam, T_base_ee,
                depth_min_mm, depth_max_mm,
            ):
                return

        # 回退：旧逐轴一维聚类
        self._assign_side(
            points, side, result, depth, K, dist, T_ee_cam, T_base_ee,
            depth_min_mm, depth_max_mm,
        )
        self._last_theta[side] = 0.0

    def _grid_assign(
        self,
        points: list[_DetPoint],
        side: str,
        result: dict[str, SlotObservation],
        depth: np.ndarray | None,
        K: np.ndarray | None,
        dist: np.ndarray | None,
        T_ee_cam: np.ndarray | None,
        T_base_ee: np.ndarray | None,
        depth_min_mm: float,
        depth_max_mm: float,
    ) -> bool:
        """在度量系里估计旋转+步距，一对一指派到 n_cols×n_rows 栅格。成功返回 True。"""
        row_order = self._left_rows if side == "left" else self._right_rows
        col_order = self._left_cols if side == "left" else self._right_cols
        n_rows = len(row_order)
        n_cols = len(col_order)

        xy = np.array([[p.x_mm, p.y_mm] for p in points], dtype=np.float64)

        # ① 估计架面旋转角
        theta = _estimate_grid_rotation(xy)

        # ② 绕 -theta 旋到对齐系：x' → 列方向，y' → 行方向
        c, s = math.cos(theta), math.sin(theta)
        xr = c * xy[:, 0] + s * xy[:, 1]
        yr = -s * xy[:, 0] + c * xy[:, 1]

        # ③ 步距：配置优先，否则在线估计；任一失败则回退
        px = self._pitch_col if self._pitch_col is not None else _estimate_axis_pitch(
            xr.tolist(), n_cols
        )
        py = self._pitch_row if self._pitch_row is not None else _estimate_axis_pitch(
            yr.tolist(), n_rows
        )
        if px is None or py is None or px <= 0 or py <= 0:
            return False

        # ④ 原点拟合 + 行列索引
        ox = _fit_origin(xr.tolist(), px)
        oy = _fit_origin(yr.tolist(), py)
        col_idx = np.clip(np.round((xr - ox) / px).astype(int), 0, n_cols - 1)
        row_idx = np.clip(np.round((yr - oy) / py).astype(int), 0, n_rows - 1)

        # ⑤ 拟合残差健康检查
        resid = np.hypot(xr - (ox + col_idx * px), yr - (oy + row_idx * py))
        if float(np.median(resid)) > self._grid_fit_max_resid_frac * min(px, py):
            return False

        # ⑥ 理想格点代价矩阵 → 一对一指派（门限内才落槽）
        cells = [(ci, ri) for ci in range(n_cols) for ri in range(n_rows)]
        cost = np.zeros((len(points), len(cells)), dtype=np.float64)
        for i in range(len(points)):
            for j, (ci, ri) in enumerate(cells):
                cost[i, j] = math.hypot(
                    xr[i] - (ox + ci * px), yr[i] - (oy + ri * py)
                )

        gate = self._assign_gate_frac * min(px, py)
        for i, j in _solve_assignment(cost):
            if cost[i, j] > gate:
                continue
            ci, ri = cells[j]
            slot_id = f"{side}.{row_order[ri]}{col_order[ci]}"
            result[slot_id] = _detection_to_observation(
                slot_id, points[i].detection, depth, K, dist, T_ee_cam, T_base_ee,
                depth_min_mm, depth_max_mm,
            )

        self._last_theta[side] = float(theta)
        return True

    def _fill_empty_on_rack_plane(
        self,
        result: dict[str, SlotObservation],
        K: np.ndarray,
        dist: np.ndarray,
        T_ee_cam: np.ndarray,
        T_base_ee: np.ndarray,
        *,
        z_rack: float | None = None,
    ) -> None:
        """empty：用 pixel_uv 与 z_rack 平面求交得到 base_xyz。"""
        from world.tube_registry import TubeRegistryError, estimate_z_rack

        if z_rack is None:
            try:
                z_rack = estimate_z_rack(
                    result,
                    tube_above_rack_mm=self._tube_above_rack_mm,
                    default_z_mm=self._default_rack_z_mm,
                )
            except TubeRegistryError:
                return

        for slot_id, obs in result.items():
            if obs.klass != "empty" or obs.pixel_uv is None:
                continue
            u, v = obs.pixel_uv
            try:
                base_xyz = pixel_uv_to_rack_plane_mm(
                    u, v, z_rack, K, dist, T_ee_cam, T_base_ee,
                )
            except CoordTransformError:
                continue
            result[slot_id] = SlotObservation(
                slot_id=slot_id,
                klass=obs.klass,
                confidence=obs.confidence,
                pixel_uv=obs.pixel_uv,
                base_xyz=base_xyz,
                z_source="rack_plane",
            )

    def to_table_str(self, observations: dict[str, SlotObservation]) -> str:
        lines = [f"{'slot_id':<12} {'class':<8} {'conf':>6}  {'uv':<18} {'base_xyz'}", "-" * 70]
        for slot_id in self.all_slot_ids():
            obs = observations[slot_id]
            uv = f"({obs.pixel_uv[0]:.0f},{obs.pixel_uv[1]:.0f})" if obs.pixel_uv else "-"
            if obs.base_xyz:
                xyz = f"({obs.base_xyz[0]:.1f},{obs.base_xyz[1]:.1f},{obs.base_xyz[2]:.1f})"
            else:
                xyz = "-"
            lines.append(
                f"{slot_id:<12} {obs.klass:<8} {obs.confidence:>6.3f}  {uv:<18} {xyz}"
            )
        return "\n".join(lines)

    def _assign_side(
        self,
        points: list[_DetPoint],
        side: str,
        result: dict[str, SlotObservation],
        depth: np.ndarray | None,
        K: np.ndarray | None,
        dist: np.ndarray | None,
        T_ee_cam: np.ndarray | None,
        T_base_ee: np.ndarray | None,
        depth_min_mm: float,
        depth_max_mm: float,
    ) -> None:
        if not points:
            return

        row_order = self._left_rows if side == "left" else self._right_rows
        col_order = self._left_cols if side == "left" else self._right_cols

        col_labels = _cluster_1d([p.u for p in points], k=len(col_order))
        row_labels = _cluster_1d([p.v for p in points], k=len(row_order))

        grid_centers: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for p, col_idx, row_idx in zip(points, col_labels, row_labels):
            grid_centers.setdefault((col_idx, row_idx), []).append((p.u, p.v))

        grid_mean: dict[tuple[int, int], tuple[float, float]] = {}
        for key, uv_list in grid_centers.items():
            us = [uv[0] for uv in uv_list]
            vs = [uv[1] for uv in uv_list]
            grid_mean[key] = (float(np.mean(us)), float(np.mean(vs)))

        assigned: dict[str, _DetPoint] = {}
        for p, col_idx, row_idx in zip(points, col_labels, row_labels):
            if col_idx >= len(col_order) or row_idx >= len(row_order):
                continue

            row = row_order[row_idx]
            col = col_order[col_idx]
            slot_id = f"{side}.{row}{col}"

            mean_uv = grid_mean.get((col_idx, row_idx))
            if mean_uv is not None:
                dist_px = np.hypot(p.u - mean_uv[0], p.v - mean_uv[1])
                if dist_px > self._max_assign_dist_px:
                    continue

            prev = assigned.get(slot_id)
            if prev is None or p.detection.confidence > prev.detection.confidence:
                assigned[slot_id] = p

        for slot_id, p in assigned.items():
            obs = _detection_to_observation(
                slot_id,
                p.detection,
                depth,
                K,
                dist,
                T_ee_cam,
                T_base_ee,
                depth_min_mm,
                depth_max_mm,
            )
            result[slot_id] = obs


def _detection_to_observation(
    slot_id: str,
    det: Detection,
    depth: np.ndarray | None,
    K: np.ndarray | None,
    dist: np.ndarray | None,
    T_ee_cam: np.ndarray | None,
    T_base_ee: np.ndarray | None,
    depth_min_mm: float,
    depth_max_mm: float,
) -> SlotObservation:
    u, v = det.center_uv
    base_xyz = None
    z_source = "missing"

    if det.class_name == "empty":
        return SlotObservation(
            slot_id=slot_id,
            klass=det.class_name,
            confidence=det.confidence,
            pixel_uv=(float(u), float(v)),
            base_xyz=None,
            z_source="missing",
        )

    if depth is not None and K is not None and dist is not None and T_ee_cam is not None and T_base_ee is not None:
        try:
            p_base, _ = pixel_to_base_mm(
                u,
                v,
                depth,
                K,
                dist,
                T_ee_cam,
                T_base_ee,
                depth_min_mm=depth_min_mm,
                depth_max_mm=depth_max_mm,
            )
            base_xyz = (float(p_base[0]), float(p_base[1]), float(p_base[2]))
            z_source = "measured"
        except CoordTransformError:
            pass

    return SlotObservation(
        slot_id=slot_id,
        klass=det.class_name,
        confidence=det.confidence,
        pixel_uv=(float(u), float(v)),
        base_xyz=base_xyz,
        z_source=z_source,
    )


def _find_split_u(values: list[float], image_width: int) -> float:
    """按 u 方向最大间隙分成左架 / 右架。"""
    if len(values) < 2:
        return image_width / 2.0

    sorted_u = sorted(values)
    max_gap = -1.0
    split = image_width / 2.0
    for i in range(len(sorted_u) - 1):
        gap = sorted_u[i + 1] - sorted_u[i]
        if gap > max_gap:
            max_gap = gap
            split = (sorted_u[i] + sorted_u[i + 1]) / 2.0
    return split


def _cluster_1d(values: list[float], k: int) -> list[int]:
    """
    一维 k-means，返回每个值的簇索引 0..k-1。
    簇编号按质心从小到大排序（u/v 小的簇号小）。
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return []
    if n == 1:
        return [0]
    k = max(1, min(k, n))

    centroids = np.linspace(float(arr.min()), float(arr.max()), k)
    labels = np.zeros(n, dtype=int)

    for _ in range(40):
        dists = np.abs(arr[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        new_centroids = centroids.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = float(arr[mask].mean())
        if np.allclose(centroids, new_centroids, atol=1e-3):
            break
        centroids = new_centroids

    order = np.argsort(centroids)
    remap = {int(old): new for new, old in enumerate(order)}
    return [remap[int(label)] for label in labels]


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_grid_rotation(points_xy: np.ndarray) -> float:
    """
    由最近邻方向估计栅格旋转角 θ（rad），返回 [-π/4, π/4)。

    行/列方向正交且无向，最近邻方向具有 π/2 周期，故用 4 倍角圆均值折叠
    （θ 与 θ±π/2、θ+π 都折叠到同一 4θ），对缺列/缺行、方向偏置都鲁棒。
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return 0.0

    angles: list[float] = []
    for i in range(n):
        d = pts - pts[i]
        dist2 = d[:, 0] ** 2 + d[:, 1] ** 2
        dist2[i] = np.inf
        j = int(np.argmin(dist2))
        dx = float(pts[j, 0] - pts[i, 0])
        dy = float(pts[j, 1] - pts[i, 1])
        angles.append(math.atan2(dy, dx))

    ang = np.asarray(angles, dtype=np.float64)
    mean4 = math.atan2(
        float(np.mean(np.sin(4.0 * ang))),
        float(np.mean(np.cos(4.0 * ang))),
    )
    return mean4 / 4.0


def _estimate_axis_pitch(values: list[float], k: int) -> float | None:
    """
    在线估计一维栅格步距：试 m∈[2,k] 层，取残差最小；残差近平局时偏向更多层
    （更小步距），以便缺整列/整行时仍恢复真实间距。
    """
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n < 2 or k < 2:
        return None
    span = vals[-1] - vals[0]
    if span <= 1e-6:
        return None

    cands: list[tuple[float, int, float]] = []  # (resid, m, pitch)
    for m in range(2, k + 1):
        pitch = span / (m - 1)
        if pitch <= 1e-6:
            continue
        acc = 0.0
        for v in vals:
            idx = round((v - vals[0]) / pitch)
            acc += (v - (vals[0] + idx * pitch)) ** 2
        cands.append((math.sqrt(acc / n), m, pitch))
    if not cands:
        return None

    min_resid = min(c[0] for c in cands)
    tol = max(1e-6, 0.15 * (span / (k - 1)))
    near = [c for c in cands if c[0] <= min_resid + tol]
    near.sort(key=lambda c: -c[1])  # 近平局偏多层
    return near[0][2]


def _fit_origin(values: list[float], pitch: float) -> float:
    """给定步距，拟合栅格原点（最小层的坐标）。"""
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    v0 = float(vals.min())
    phase = (vals - v0) / pitch
    resid = phase - np.round(phase)
    offset = float(np.mean(resid)) * pitch
    return v0 + offset


def _solve_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """一对一最小代价指派：有 scipy 用匈牙利，否则贪心。返回 (行, 列) 对。"""
    cost = np.asarray(cost, dtype=np.float64)
    if _HAS_SCIPY:
        rows, cols = _linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))

    n, m = cost.shape
    order = sorted(
        (
            (float(cost[i, j]), i, j)
            for i in range(n)
            for j in range(m)
        ),
        key=lambda t: t[0],
    )
    used_r: set[int] = set()
    used_c: set[int] = set()
    pairs: list[tuple[int, int]] = []
    limit = min(n, m)
    for _, i, j in order:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        pairs.append((i, j))
        if len(pairs) >= limit:
            break
    return pairs
