"""精定位：approach 位 YOLO + 基坐标 XY 匹配预期槽。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
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
    use_ellipse_center: bool = False,
    ellipse_bbox_pad_px: float = 8.0,
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
        use_ellipse_center=use_ellipse_center,
        ellipse_bbox_pad_px=ellipse_bbox_pad_px,
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
    use_ellipse_center: bool = False,
    ellipse_bbox_pad_px: float = 8.0,
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
        measure_uv, uv_source = _measurement_uv(
            color_bgr,
            det,
            use_ellipse_center=use_ellipse_center,
            ellipse_bbox_pad_px=ellipse_bbox_pad_px,
        )
        matched_by_projection = False
        try:
            if z_override is not None:
                meas_x, meas_y, meas_z = pixel_uv_to_rack_plane_mm(
                    measure_uv[0],
                    measure_uv[1],
                    float(z_override),
                    K,
                    dist,
                    T_ee_cam,
                    T_base_ee,
                )
            else:
                p_base, _ = pixel_to_base_mm(
                    measure_uv[0],
                    measure_uv[1],
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
            errors.append(f"uv={measure_uv}({uv_source}): {exc}")
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
                pixel_uv=(float(measure_uv[0]), float(measure_uv[1])),
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


def _measurement_uv(
    color_bgr: np.ndarray,
    det: Detection,
    *,
    use_ellipse_center: bool,
    ellipse_bbox_pad_px: float,
) -> tuple[tuple[float, float], str]:
    """返回用于 3D 反投影的像素点：优先框内椭圆中心，失败回退 YOLO 框中心。"""
    if not use_ellipse_center or det.class_name != "tube":
        return det.center_uv, "bbox"
    fitted = _fit_tube_ellipse_center(color_bgr, det.bbox, ellipse_bbox_pad_px)
    if fitted is None:
        return det.center_uv, "bbox"
    return fitted, "ellipse"


def _fit_tube_ellipse_center(
    color_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_px: float,
) -> tuple[float, float] | None:
    h, w = color_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(0, int(round(pad_px)))
    ix1 = max(0, int(np.floor(x1)) - pad)
    iy1 = max(0, int(np.floor(y1)) - pad)
    ix2 = min(w, int(np.ceil(x2)) + pad)
    iy2 = min(h, int(np.ceil(y2)) + pad)
    if ix2 - ix1 < 8 or iy2 - iy1 < 8:
        return None

    roi = color_bgr[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median + 30))
    edges = cv2.Canny(gray, lower, upper)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_h, roi_w = gray.shape[:2]
    bbox_center = np.array([(x1 + x2) / 2.0 - ix1, (y1 + y2) / 2.0 - iy1])
    best: tuple[float, tuple[float, float]] | None = None
    for contour in contours:
        if len(contour) < 5:
            continue
        area = float(cv2.contourArea(contour))
        if area < 20 or area > roi_w * roi_h * 0.8:
            continue
        (cx, cy), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major = max(float(axis_a), float(axis_b))
        minor = min(float(axis_a), float(axis_b))
        if minor < 4 or major < 6:
            continue
        ratio = major / max(minor, 1e-6)
        if ratio > 3.0:
            continue
        if not (-pad <= cx <= roi_w + pad and -pad <= cy <= roi_h + pad):
            continue

        center_dist = float(np.linalg.norm(np.array([cx, cy]) - bbox_center))
        score = area - center_dist * 8.0 - (ratio - 1.0) * 20.0
        if best is None or score > best[0]:
            best = (score, (float(cx + ix1), float(cy + iy1)))

    return None if best is None else best[1]


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
