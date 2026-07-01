"""YOLO 检测 → 24 孔试管架槽位映射。

流水线（map）：
  1. 盖子框(tube) → 真实深度 → base XYZ；由此估 z_rack（盖顶 - 盖高）
  2. 空槽框(empty) 及深度失效的盖子 → 射线打 z_rack 平面 → base XY
  3. 左右分架
  4. 每侧在 base XY 上做点阵吸附(grid_fit) → (行,列) → slot_id
方向与透视：在真实 XY 平面上拟合，透视自动消失、盖子/空槽高度差不再造成偏移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.coord_transform import (
    CoordTransformError,
    pixel_plane_to_base_mm,
    pixel_to_base_mm,
)
from perception.detection import Detection
from perception.grid_fit import fit_grid
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
    base: tuple[float, float, float] | None = None  # base XYZ（盖子=深度，空槽=射线打平面）


def estimate_rack_plane_z(
    tube_zs: list[float],
    *,
    tube_above_rack_mm: float = 30.0,
    default_z_mm: float | None = None,
) -> float | None:
    """
    架面高度 z_rack（mm）。纯函数，不依赖 registry。
    主力：盖顶 Z 中位数 - 盖高；兜底：预标定 default_z_mm；都无 → None。
    """
    if tube_zs:
        return float(np.median(tube_zs)) - float(tube_above_rack_mm)
    if default_z_mm is not None:
        return float(default_z_mm)
    return None


class SlotMapper:
    def __init__(
        self,
        rack_config: dict[str, Any] | None = None,
        *,
        max_assign_dist_px: float = 55.0,
        image_width: int = 640,
        method: str = "lattice",
        residual_max_ratio: float = 0.35,
        min_points_for_fit: int = 6,
        side_max_count: int = 12,
    ) -> None:
        self._rack = rack_config or load_yaml("config/rack_layout.yaml")
        self._max_assign_dist_px = max_assign_dist_px
        self._image_width = image_width
        self._method = method
        self._residual_max_ratio = residual_max_ratio
        self._min_points_for_fit = min_points_for_fit
        self._side_max_count = side_max_count

        self._left_rows = [str(r) for r in self._rack["left_row_order"]]
        self._left_cols = [int(c) for c in self._rack["left_col_order"]]
        self._right_rows = [str(r) for r in self._rack["right_row_order"]]
        self._right_cols = [int(c) for c in self._rack["right_col_order"]]

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
        tube_above_rack_mm: float = 30.0,
        default_z_rack_mm: float | None = None,
    ) -> tuple[dict[str, SlotObservation], float | None]:
        """
        映射 24 槽并返回 (observations, z_rack)。
        z_rack 由盖子深度估计（空槽射线打平面需要它），因此在 map 内部计算并返回。
        """
        result = {slot_id: SlotObservation(slot_id=slot_id) for slot_id in self.all_slot_ids()}
        if not detections:
            z = float(default_z_rack_mm) if default_z_rack_mm is not None else None
            return result, z

        points = [_DetPoint(det, det.center_uv[0], det.center_uv[1]) for det in detections]
        has_geom = all(x is not None for x in (depth, K, dist, T_ee_cam, T_base_ee))

        z_rack: float | None = (
            float(default_z_rack_mm) if default_z_rack_mm is not None else None
        )
        if has_geom:
            z_rack = self._project_points(
                points, depth, K, dist, T_ee_cam, T_base_ee,
                depth_min_mm, depth_max_mm, tube_above_rack_mm, default_z_rack_mm,
            )

        split_u = _find_split_u([p.u for p in points], self._image_width)
        left_pts = [p for p in points if p.u < split_u]
        right_pts = [p for p in points if p.u >= split_u]
        for pts, side in ((left_pts, "left"), (right_pts, "right")):
            if len(pts) > self._side_max_count:
                print(f"[SlotMapper] 警告: {side} 侧检测 {len(pts)} 个 > {self._side_max_count}")
            self._assign_side(pts, side, result)
        return result, z_rack

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

    # --- 内部 ---

    def _project_points(
        self,
        points: list[_DetPoint],
        depth: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        T_ee_cam: np.ndarray,
        T_base_ee: np.ndarray,
        depth_min_mm: float,
        depth_max_mm: float,
        tube_above_rack_mm: float,
        default_z_rack_mm: float | None,
    ) -> float | None:
        """盖子走真实深度、空槽走射线打平面；返回 z_rack。"""
        tube_zs: list[float] = []
        for p in points:
            if p.detection.class_name != "tube":
                continue
            try:
                pb, _ = pixel_to_base_mm(
                    p.u, p.v, depth, K, dist, T_ee_cam, T_base_ee,
                    depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm,
                )
                p.base = (float(pb[0]), float(pb[1]), float(pb[2]))
                tube_zs.append(float(pb[2]))
            except CoordTransformError:
                p.base = None  # 盖子深度失效，稍后射线打平面兜底

        z_rack = estimate_rack_plane_z(
            tube_zs, tube_above_rack_mm=tube_above_rack_mm, default_z_mm=default_z_rack_mm
        )
        if z_rack is not None:
            for p in points:
                if p.base is not None:
                    continue  # 盖子已用真实深度
                try:
                    pb = pixel_plane_to_base_mm(p.u, p.v, z_rack, K, dist, T_ee_cam, T_base_ee)
                    p.base = (float(pb[0]), float(pb[1]), float(pb[2]))
                except CoordTransformError:
                    p.base = None
        return z_rack

    def _assign_side(
        self,
        points: list[_DetPoint],
        side: str,
        result: dict[str, SlotObservation],
    ) -> None:
        if not points:
            return
        row_order = self._left_rows if side == "left" else self._right_rows
        col_order = self._left_cols if side == "left" else self._right_cols

        if self._method == "lattice" and self._try_lattice(points, side, row_order, col_order, result):
            return
        self._assign_side_legacy(points, side, row_order, col_order, result)

    def _try_lattice(
        self,
        points: list[_DetPoint],
        side: str,
        row_order: list[str],
        col_order: list[int],
        result: dict[str, SlotObservation],
    ) -> bool:
        """在 base XY（缺失则退回像素）上做点阵吸附。成功返回 True。"""
        if len(points) < self._min_points_for_fit:
            return False
        use_base = all(p.base is not None for p in points)
        if use_base:
            coords = np.array([[p.base[0], p.base[1]] for p in points], dtype=np.float64)
        else:
            coords = np.array([[p.u, p.v] for p in points], dtype=np.float64)

        fit = fit_grid(
            coords, len(col_order), len(row_order),
            residual_max_ratio=self._residual_max_ratio,
            min_points=self._min_points_for_fit,
        )
        if not fit.ok:
            return False

        valid = [(p, ij) for p, ij in zip(points, fit.indices) if ij is not None]
        if not valid:
            return False

        cols = np.array([ij[0] for _, ij in valid])
        rows = np.array([ij[1] for _, ij in valid])
        us = np.array([p.u for p, _ in valid])
        vs = np.array([p.v for p, _ in valid])
        n_cols, n_rows = len(col_order), len(row_order)
        # 方向校正：col_idx 随图像 u 增大、row_idx 随 v 增大（与 config 的 col/row_order 约定一致）
        if _corr_sign(cols, us) < 0:
            cols = (n_cols - 1) - cols
        if _corr_sign(rows, vs) < 0:
            rows = (n_rows - 1) - rows

        assigned: dict[str, _DetPoint] = {}
        for (p, _), c, r in zip(valid, cols, rows):
            if not (0 <= c < n_cols and 0 <= r < n_rows):
                continue
            slot_id = f"{side}.{row_order[r]}{col_order[c]}"
            prev = assigned.get(slot_id)
            if prev is None or p.detection.confidence > prev.detection.confidence:
                assigned[slot_id] = p
        for slot_id, p in assigned.items():
            result[slot_id] = _point_to_observation(slot_id, p)
        return True

    def _assign_side_legacy(
        self,
        points: list[_DetPoint],
        side: str,
        row_order: list[str],
        col_order: list[int],
        result: dict[str, SlotObservation],
    ) -> None:
        """旧算法：u/v 各自 1D 分堆（兜底，架子须近似正对）。坐标沿用已投影的 p.base。"""
        col_labels = _cluster_1d([p.u for p in points], k=len(col_order))
        row_labels = _cluster_1d([p.v for p in points], k=len(row_order))

        grid_uv: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for p, ci, ri in zip(points, col_labels, row_labels):
            grid_uv.setdefault((ci, ri), []).append((p.u, p.v))
        grid_mean = {
            key: (float(np.mean([u for u, _ in lst])), float(np.mean([v for _, v in lst])))
            for key, lst in grid_uv.items()
        }

        assigned: dict[str, _DetPoint] = {}
        for p, ci, ri in zip(points, col_labels, row_labels):
            if ci >= len(col_order) or ri >= len(row_order):
                continue
            mean_uv = grid_mean.get((ci, ri))
            if mean_uv is not None and np.hypot(p.u - mean_uv[0], p.v - mean_uv[1]) > self._max_assign_dist_px:
                continue
            slot_id = f"{side}.{row_order[ri]}{col_order[ci]}"
            prev = assigned.get(slot_id)
            if prev is None or p.detection.confidence > prev.detection.confidence:
                assigned[slot_id] = p
        for slot_id, p in assigned.items():
            result[slot_id] = _point_to_observation(slot_id, p)


def _point_to_observation(slot_id: str, p: _DetPoint) -> SlotObservation:
    det = p.detection
    if p.base is None:
        z_source = "missing"
    elif det.class_name == "tube":
        z_source = "measured"
    else:
        z_source = "rack_plane"
    return SlotObservation(
        slot_id=slot_id,
        klass=det.class_name,
        confidence=det.confidence,
        pixel_uv=(float(p.u), float(p.v)),
        base_xyz=p.base,
        z_source=z_source,
    )


def _corr_sign(a: np.ndarray, b: np.ndarray) -> float:
    """a 与 b 的相关符号；方差退化时返回 +1（不翻转）。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 1.0
    c = float(np.corrcoef(a, b)[0, 1])
    return -1.0 if c < 0 else 1.0


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
    """一维 k-means，返回每个值的簇索引 0..k-1（质心从小到大编号）。"""
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
