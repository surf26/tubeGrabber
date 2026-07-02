"""精定位：approach 位 YOLO + 基坐标 XY 匹配预期槽。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perception.coord_transform import (
    CoordTransformError,
    pixel_to_base_mm,
    pixel_uv_to_rack_plane_mm,
)
from perception.yolo_detector import Detection, YoloDetector
from world.tube_registry import TubeRegistry


class RefineError(RuntimeError):
    """精定位失败。"""


@dataclass
class RefineResult:
    slot_id: str
    klass: str
    base_xyz: tuple[float, float, float]  # 抓取目标（夹爪 TCP），基坐标 mm
    dist_xy_mm: float
    confidence: float
    pixel_uv: tuple[float, float]
    z_source: str


def refine_pick_slot(
    slot_id: str,
    registry: TubeRegistry,
    color_bgr: np.ndarray,
    depth: np.ndarray,
    detector: YoloDetector,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    max_dist_xy_mm: float = 15.0,
    ambiguity_min_delta_mm: float = 2.0,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
    detections: list[Detection] | None = None,
) -> RefineResult:
    """抓取精定位：匹配 tube，Z 用 measured。"""
    return refine_slot(
        slot_id,
        registry,
        color_bgr,
        depth,
        detector,
        K,
        dist,
        T_ee_cam,
        T_base_ee,
        expected_klass="tube",
        z_override=None,
        max_dist_xy_mm=max_dist_xy_mm,
        ambiguity_min_delta_mm=ambiguity_min_delta_mm,
        depth_min_mm=depth_min_mm,
        depth_max_mm=depth_max_mm,
        detections=detections,
    )


def refine_place_slot(
    slot_id: str,
    registry: TubeRegistry,
    color_bgr: np.ndarray,
    depth: np.ndarray,
    detector: YoloDetector,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    max_dist_xy_mm: float = 15.0,
    ambiguity_min_delta_mm: float = 2.0,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
    detections: list[Detection] | None = None,
) -> RefineResult:
    """放置精定位：匹配 empty，Z 用 z_rack。"""
    z_rack = registry.z_rack
    if z_rack is None:
        raise RefineError("registry.z_rack 未设置，请先 SCAN_GLOBAL")
    return refine_slot(
        slot_id,
        registry,
        color_bgr,
        depth,
        detector,
        K,
        dist,
        T_ee_cam,
        T_base_ee,
        expected_klass="empty",
        z_override=float(z_rack),
        max_dist_xy_mm=max_dist_xy_mm,
        ambiguity_min_delta_mm=ambiguity_min_delta_mm,
        depth_min_mm=depth_min_mm,
        depth_max_mm=depth_max_mm,
        detections=detections,
    )


def refine_slot(
    slot_id: str,
    registry: TubeRegistry,
    color_bgr: np.ndarray,
    depth: np.ndarray,
    detector: YoloDetector,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    expected_klass: str,
    z_override: float | None,
    max_dist_xy_mm: float = 15.0,
    ambiguity_min_delta_mm: float = 2.0,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
    detections: list[Detection] | None = None,
) -> RefineResult:
    """
    在基坐标系下匹配预期槽：refined base_xy 与 registry 预期 base_xy
    平面距离 <= max_dist_xy_mm 视为同一目标。
    若多个候选落入该范围，最近与次近 dist_xy 差距 < ambiguity_min_delta_mm 则报歧义。
    """
    state = registry.get(slot_id)
    if state.base_xyz is None:
        raise RefineError(f"{slot_id} 缺少预期 base_xyz")
    if state.klass != expected_klass:
        raise RefineError(
            f"{slot_id} 预期 klass={expected_klass}，当前 registry 为 {state.klass}"
        )

    exp_x, exp_y, _exp_z = state.base_xyz
    if detections is None:
        detections = detector.detect(color_bgr)

    candidates: list[RefineResult] = []
    errors: list[str] = []
    for det in detections:
        if det.class_name != expected_klass:
            continue
        try:
            if z_override is not None:
                meas_x, meas_y, meas_z = pixel_uv_to_rack_plane_mm(
                    det.center_uv[0],
                    det.center_uv[1],
                    float(z_override),
                    K,
                    dist,
                    T_ee_cam,
                    T_base_ee,
                )
            else:
                p_base, _ = pixel_to_base_mm(
                    det.center_uv[0],
                    det.center_uv[1],
                    depth,
                    K,
                    dist,
                    T_ee_cam,
                    T_base_ee,
                    depth_min_mm=depth_min_mm,
                    depth_max_mm=depth_max_mm,
                )
                meas_x, meas_y, meas_z = float(p_base[0]), float(p_base[1]), float(p_base[2])
        except CoordTransformError as exc:
            errors.append(f"uv={det.center_uv}: {exc}")
            continue

        dist_xy = float(np.hypot(meas_x - exp_x, meas_y - exp_y))
        if dist_xy > max_dist_xy_mm:
            continue

        if z_override is not None:
            base_xyz = (meas_x, meas_y, float(z_override))
            z_source = "rack_plane"
        else:
            base_xyz = (meas_x, meas_y, meas_z)
            z_source = "measured"

        candidates.append(
            RefineResult(
                slot_id=slot_id,
                klass=expected_klass,
                base_xyz=base_xyz,
                dist_xy_mm=dist_xy,
                confidence=det.confidence,
                pixel_uv=(float(det.center_uv[0]), float(det.center_uv[1])),
                z_source=z_source,
            )
        )

    if not candidates:
        detail = "; ".join(errors[:3]) if errors else "无匹配检测"
        raise RefineError(
            f"{slot_id} 精定位失败: {expected_klass} 在 base_xy ±{max_dist_xy_mm}mm "
            f"内无有效检测 (预期 {exp_x:.1f},{exp_y:.1f}) — {detail}"
        )

    best = min(candidates, key=lambda c: c.dist_xy_mm)
    if len(candidates) > 1:
        second = sorted(candidates, key=lambda c: c.dist_xy_mm)[1]
        delta = second.dist_xy_mm - best.dist_xy_mm
        if delta < ambiguity_min_delta_mm:
            raise RefineError(
                f"{slot_id} 精定位歧义: 最近 {best.dist_xy_mm:.1f}mm vs "
                f"次近 {second.dist_xy_mm:.1f}mm (差距 {delta:.1f}mm < "
                f"{ambiguity_min_delta_mm:.1f}mm)"
            )
    return best
