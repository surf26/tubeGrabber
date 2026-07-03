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
    projected_match_max_px: float = 80.0,
    projected_bbox_margin_px: float = 20.0,
    allow_projected_fallback: bool = True,
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
        projected_match_max_px=projected_match_max_px,
        projected_bbox_margin_px=projected_bbox_margin_px,
        allow_projected_fallback=allow_projected_fallback,
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
    projected_match_max_px: float = 80.0,
    projected_bbox_margin_px: float = 20.0,
    allow_projected_fallback: bool = True,
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

    exp_x, exp_y, exp_z = state.base_xyz
    exp_uv: tuple[float, float] | None = None
    try:
        exp_uv = base_xyz_to_pixel_uv(state.base_xyz, K, dist, T_ee_cam, T_base_ee)
    except CoordTransformError as exc:
        errors_project = f"预期点投影失败: {exc}"
    else:
        errors_project = ""
    if detections is None:
        detections = detector.detect(color_bgr)

    candidates: list[RefineResult] = []
    errors: list[str] = []
    for det in detections:
        if det.class_name != expected_klass:
            continue
        matched_by_projection = False
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
            if (
                not allow_projected_fallback
                or z_override is not None
                or exp_uv is None
                or not _detection_matches_projected_target(
                    det,
                    exp_uv,
                    max_center_dist_px=projected_match_max_px,
                    bbox_margin_px=projected_bbox_margin_px,
                )
            ):
                continue
            meas_x, meas_y, meas_z = float(exp_x), float(exp_y), float(exp_z)
            matched_by_projection = True

        dist_xy = float(np.hypot(meas_x - exp_x, meas_y - exp_y))
        if dist_xy > max_dist_xy_mm:
            continue

        if z_override is not None:
            base_xyz = (meas_x, meas_y, float(z_override))
            z_source = "rack_plane"
        else:
            base_xyz = (meas_x, meas_y, meas_z)
            z_source = "projected_global" if matched_by_projection else "measured"

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
        detail_items = errors[:3]
        if errors_project:
            detail_items.append(errors_project)
        detail = "; ".join(detail_items) if detail_items else "无匹配检测"
        raise RefineError(
            f"{slot_id} 精定位失败: {expected_klass} 在 base_xy ±{max_dist_xy_mm}mm "
            f"内无有效检测 (预期 {exp_x:.1f},{exp_y:.1f}) — {detail}"
        )

    ordered = sorted(candidates, key=lambda c: _candidate_sort_key(c, exp_uv))
    best = ordered[0]
    if len(candidates) > 1 and best.z_source != "projected_global":
        second = ordered[1]
        delta = second.dist_xy_mm - best.dist_xy_mm
        if second.z_source != "projected_global" and delta < ambiguity_min_delta_mm:
            raise RefineError(
                f"{slot_id} 精定位歧义: 最近 {best.dist_xy_mm:.1f}mm vs "
                f"次近 {second.dist_xy_mm:.1f}mm (差距 {delta:.1f}mm < "
                f"{ambiguity_min_delta_mm:.1f}mm)"
            )
    return best


def base_xyz_to_pixel_uv(
    base_xyz: tuple[float, float, float],
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
) -> tuple[float, float]:
    """把第一次扫描得到的基坐标点投影到当前相机图像。"""
    T_base_cam = T_base_ee @ T_ee_cam
    T_cam_base = np.linalg.inv(T_base_cam)
    p_base = np.array([base_xyz[0], base_xyz[1], base_xyz[2], 1.0], dtype=np.float64)
    p_cam = T_cam_base @ p_base
    if p_cam[2] <= 1e-6:
        raise CoordTransformError("预期点在相机后方")

    import cv2

    pts, _ = cv2.projectPoints(
        p_cam[:3].reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        dist,
    )
    u, v = pts.reshape(-1, 2)[0]
    return float(u), float(v)


def _detection_matches_projected_target(
    det: Detection,
    projected_uv: tuple[float, float],
    *,
    max_center_dist_px: float,
    bbox_margin_px: float,
) -> bool:
    u, v = projected_uv
    x1, y1, x2, y2 = det.bbox
    in_bbox = (
        x1 - bbox_margin_px <= u <= x2 + bbox_margin_px
        and y1 - bbox_margin_px <= v <= y2 + bbox_margin_px
    )
    center_dist = float(np.hypot(det.center_uv[0] - u, det.center_uv[1] - v))
    return in_bbox or center_dist <= max_center_dist_px


def _candidate_sort_key(
    result: RefineResult,
    projected_uv: tuple[float, float] | None,
) -> tuple[bool, float, float]:
    if result.z_source != "projected_global":
        return (False, result.dist_xy_mm, -result.confidence)
    if projected_uv is None:
        pixel_dist = 1e9
    else:
        pixel_dist = float(
            np.hypot(result.pixel_uv[0] - projected_uv[0], result.pixel_uv[1] - projected_uv[1])
        )
    return (True, pixel_dist, -result.confidence)
