"""YOLO 检测 → 24 孔试管架槽位映射。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.coord_transform import CoordTransformError, pixel_to_base_mm
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


class SlotMapper:
    def __init__(
        self,
        rack_config: dict[str, Any] | None = None,
        *,
        max_assign_dist_px: float = 55.0,
        image_width: int = 640,
    ) -> None:
        self._rack = rack_config or load_yaml("config/rack_layout.yaml")
        self._max_assign_dist_px = max_assign_dist_px
        self._image_width = image_width

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
    ) -> dict[str, SlotObservation]:
        """
        将 YOLO 检测映射到 24 个逻辑槽位。
        若提供 depth + 标定 + 臂姿，则填充 base_xyz。
        """
        result = {slot_id: SlotObservation(slot_id=slot_id) for slot_id in self.all_slot_ids()}
        if not detections:
            return result

        points = [_DetPoint(det, det.center_uv[0], det.center_uv[1]) for det in detections]
        split_u = _find_split_u([p.u for p in points], self._image_width)

        left_pts = [p for p in points if p.u < split_u]
        right_pts = [p for p in points if p.u >= split_u]

        self._assign_side(left_pts, "left", result, depth, K, dist, T_ee_cam, T_base_ee, depth_min_mm, depth_max_mm)
        self._assign_side(right_pts, "right", result, depth, K, dist, T_ee_cam, T_base_ee, depth_min_mm, depth_max_mm)
        return result

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
