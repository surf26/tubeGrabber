"""扫描 / refine 标注与单窗口仪表盘显示。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from utils.opencv_gui import cv2
import numpy as np

from perception.refine import RefineResult
from perception.slot_mapper import SlotMapper, SlotObservation
from perception.yolo_detector import Detection


_COLORS = {
    "tube": (60, 220, 60),
    "empty": (50, 180, 255),
    "unknown": (140, 140, 140),
}


def draw_scan_annotation(
    bgr: np.ndarray,
    detections: list[Detection],
    observations: dict[str, SlotObservation],
    mapper: SlotMapper,
    *,
    title: str = "SCAN",
    z_rack: float | None = None,
    font_scale: float = 0.28,
) -> np.ndarray:
    """YOLO 框 + 24 槽编号 + 图例。"""
    canvas = _draw_yolo_boxes(bgr, detections, font_scale)

    for slot_id in mapper.all_slot_ids():
        obs = observations.get(slot_id)
        if obs is None or obs.pixel_uv is None:
            continue
        u, v = int(obs.pixel_uv[0]), int(obs.pixel_uv[1])
        klass = obs.klass
        color = _COLORS.get(klass, _COLORS["unknown"])
        cv2.circle(canvas, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
        _text(canvas, slot_id, (u + 4, v - 2), color, font_scale * 0.85)

    tube_n = sum(1 for o in observations.values() if o.klass == "tube")
    empty_n = sum(1 for o in observations.values() if o.klass == "empty")
    header = f"{title}  tube={tube_n} empty={empty_n} det={len(detections)}"
    if z_rack is not None:
        header += f"  z_rack={z_rack:.0f}mm"
    _caption(canvas, header, (6, 14))

    y0 = 28
    for label, color in (("tube", _COLORS["tube"]), ("empty", _COLORS["empty"]), ("unk", _COLORS["unknown"])):
        cv2.circle(canvas, (10, y0), 2, color, -1, lineType=cv2.LINE_AA)
        _text(canvas, label, (18, y0 + 3), color, font_scale * 0.75)
        y0 += 12

    return canvas


def draw_refine_annotation(
    bgr: np.ndarray,
    detections: list[Detection],
    slot_id: str,
    result: RefineResult | None = None,
    *,
    title: str = "REFINE",
    font_scale: float = 0.28,
) -> np.ndarray:
    """精定位帧：全部检测 + 高亮目标槽。"""
    canvas = _draw_yolo_boxes(bgr, detections, font_scale)

    if result is not None:
        u, v = int(result.pixel_uv[0]), int(result.pixel_uv[1])
        color = _COLORS.get(result.klass, (0, 255, 255))
        cv2.drawMarker(canvas, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)
        _text(canvas, slot_id, (u + 4, v - 2), color, font_scale * 0.85)
        sub = f"dist_xy={result.dist_xy_mm:.1f}mm conf={result.confidence:.2f}"
        _caption(canvas, f"{title} {slot_id}  {sub}", (6, 14))
    else:
        _caption(canvas, f"{title} {slot_id}", (6, 14))

    return canvas


def _draw_yolo_boxes(
    bgr: np.ndarray,
    detections: list[Detection],
    font_scale: float,
) -> np.ndarray:
    canvas = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = _COLORS.get(det.class_name, (180, 180, 180))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        u, v = int(det.center_uv[0]), int(det.center_uv[1])
        cv2.circle(canvas, (u, v), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        label = f"{det.class_name[0]}:{det.confidence:.2f}"
        _text(canvas, label, (x1, max(y1 - 1, 8)), color, font_scale * 0.7)
    return canvas


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        lineType=cv2.LINE_AA,
    )


def _caption(img: np.ndarray, text: str, org: tuple[int, int]) -> None:
    """顶部/底部说明文字"""
    _text(img, text, org, (230, 230, 230), 0.28)


class VisionDisplay:
    """FSM 用：单窗口 dashboard，展示 scan/refine/live/状态表。"""

    def __init__(self, config: dict[str, Any]) -> None:
        vis = config.get("vision", {})
        self._enabled = bool(vis.get("display_enabled", True))
        self._save_views = bool(vis.get("save_latest_views", False))
        self._capture_dir = Path(vis.get("capture_dir", "data/captures"))
        self._window = str(vis.get("dashboard_window", "tubeGrabber Dashboard"))
        self._width = int(vis.get("dashboard_width", 1720))
        self._height = int(vis.get("dashboard_height", 960))
        self._scan_history_limit = max(1, int(vis.get("scan_history_count", 3)))
        self._live_enabled = bool(vis.get("live_preview_enabled", True))
        self._live_interval_s = 1.0 / max(1.0, float(vis.get("live_preview_fps", 5)))

        self._scan_history: list[np.ndarray] = []
        self._refine_img: np.ndarray | None = None
        self._live_img: np.ndarray | None = None
        self._registry = None
        self._camera = None
        self._status = "waiting"
        self._last_live_t = 0.0
        self._window_ready = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind_camera(self, camera) -> None:
        self._camera = camera
        self.render()

    def set_status(self, message: str) -> None:
        self._status = message
        self.render()

    def update_registry(self, registry) -> None:
        self._registry = registry
        self.render()

    def show_scan(self, bgr: np.ndarray, *, registry=None, status: str = "scan updated") -> None:
        if not self._enabled:
            return
        img = np.ascontiguousarray(bgr)
        self._scan_history.insert(0, img)
        self._scan_history = self._scan_history[: self._scan_history_limit]
        if registry is not None:
            self._registry = registry
        self._status = status
        self._save_view("latest_scan_view.png", img)
        self.render()

    def show_refine(self, bgr: np.ndarray, *, registry=None, status: str = "refine updated") -> None:
        if not self._enabled:
            return
        img = np.ascontiguousarray(bgr)
        self._refine_img = img
        if registry is not None:
            self._registry = registry
        self._status = status
        self._save_view("latest_refine_view.png", img)

        self.render()

    def tick_live(self, *, status: str | None = None) -> None:
        if not self._enabled:
            return
        if status is not None:
            self._status = status
        now = time.monotonic()
        if (
            self._live_enabled
            and self._camera is not None
            and now - self._last_live_t >= self._live_interval_s
        ):
            self._last_live_t = now
            try:
                self._live_img = self._camera.capture().color
            except Exception as exc:
                self._status = f"live camera unavailable: {exc}"
        self.render()

    def render(self) -> None:
        if not self._enabled:
            return
        canvas = self._compose_dashboard()
        self._ensure_window()
        cv2.imshow(self._window, canvas)
        cv2.waitKey(1)

    def close_all(self) -> None:
        cv2.destroyAllWindows()
        self._window_ready = False

    def wait_for_quit(self, message: str = "press q/Esc to close dashboard") -> None:
        if not self._enabled:
            return
        self._status = message
        self.render()
        print(f"[viewer] {message}")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                return

    def _save_view(self, filename: str, img: np.ndarray) -> None:
        if not self._save_views:
            return
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            path = self._capture_dir / filename
            cv2.imwrite(str(path), img)
            print(f"[viewer] 已保存 {path}")
        except Exception as exc:
            print(f"[viewer] 保存图像失败: {exc}")

    def _ensure_window(self) -> None:
        if self._window_ready:
            return
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window, self._width, self._height)
        self._window_ready = True

    def _compose_dashboard(self) -> np.ndarray:
        canvas = np.full((self._height, self._width, 3), (18, 20, 24), dtype=np.uint8)
        margin = 12
        header_h = 42
        left_w = 430
        table_w = 520
        center_w = self._width - left_w - table_w - margin * 4
        body_y = header_h + margin
        body_h = self._height - body_y - margin
        left_x = margin
        center_x = left_x + left_w + margin
        table_x = center_x + center_w + margin

        _draw_header(canvas, "tubeGrabber", self._status)

        scan_rect = (left_x, body_y, left_w, body_h)
        refine_rect = (center_x, body_y, center_w, int(body_h * 0.62))
        live_rect = (
            center_x,
            refine_rect[1] + refine_rect[3] + margin,
            center_w,
            body_h - refine_rect[3] - margin,
        )
        table_rect = (table_x, body_y, table_w, body_h)

        _draw_panel(canvas, scan_rect, "Global YOLO scans")
        self._draw_scan_history(canvas, scan_rect)

        _draw_panel(canvas, refine_rect, "Refine scan")
        _paste_or_placeholder(canvas, self._refine_img, _inner_rect(refine_rect), "waiting for refine")

        _draw_panel(canvas, live_rect, "Live camera")
        _paste_or_placeholder(canvas, self._live_img, _inner_rect(live_rect), "waiting for camera")

        _draw_panel(canvas, table_rect, "Tube registry")
        self._draw_registry_table(canvas, table_rect)
        return canvas

    def _draw_scan_history(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        inner = _inner_rect(rect)
        x, y, w, h = inner
        gap = 8
        slot_h = max(1, (h - gap * (self._scan_history_limit - 1)) // self._scan_history_limit)
        for i in range(self._scan_history_limit):
            item_y = y + i * (slot_h + gap)
            item_rect = (x, item_y, w, slot_h)
            image = self._scan_history[i] if i < len(self._scan_history) else None
            label = f"scan #{i + 1}" if image is not None else "waiting for scan"
            _paste_or_placeholder(canvas, image, item_rect, label)
            cv2.putText(
                canvas,
                label,
                (item_rect[0] + 8, item_rect[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 238, 240),
                1,
                cv2.LINE_AA,
            )

    def _draw_registry_table(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = _inner_rect(rect)
        row_h = 29
        header_y = y + 22
        _put_text(canvas, "slot", (x + 8, header_y), 0.45, (180, 190, 200), 1)
        _put_text(canvas, "state", (x + 110, header_y), 0.45, (180, 190, 200), 1)
        _put_text(canvas, "conf", (x + 210, header_y), 0.45, (180, 190, 200), 1)
        _put_text(canvas, "xyz / z_src", (x + 292, header_y), 0.45, (180, 190, 200), 1)
        cv2.line(canvas, (x, y + 34), (x + w, y + 34), (65, 72, 80), 1)

        if self._registry is None:
            _put_text(canvas, "waiting for first scan", (x + 8, y + 70), 0.55, (170, 176, 184), 1)
            return

        z_rack = getattr(self._registry, "z_rack", None)
        if z_rack is not None:
            _put_text(canvas, f"z_rack={z_rack:.1f} mm", (x + w - 150, y - 8), 0.42, (155, 205, 255), 1)

        yy = y + 60
        for slot_id in self._registry.slot_ids():
            if yy + row_h > y + h:
                break
            state = self._registry.get(slot_id)
            color = _COLORS.get(state.klass, _COLORS["unknown"])
            if state.klass == "tube":
                bg = (26, 48, 32)
            elif state.klass == "empty":
                bg = (48, 39, 24)
            else:
                bg = (34, 36, 40)
            cv2.rectangle(canvas, (x, yy - 17), (x + w, yy + 8), bg, -1)
            cv2.circle(canvas, (x + 9, yy - 4), 4, color, -1, lineType=cv2.LINE_AA)
            _put_text(canvas, slot_id, (x + 20, yy), 0.45, (235, 238, 240), 1)
            _put_text(canvas, state.klass, (x + 110, yy), 0.45, color, 1)
            _put_text(canvas, f"{state.confidence:.2f}", (x + 210, yy), 0.43, (210, 216, 222), 1)
            if state.base_xyz:
                xyz = f"{state.base_xyz[0]:.0f},{state.base_xyz[1]:.0f},{state.base_xyz[2]:.0f}"
            else:
                xyz = "-"
            _put_text(canvas, f"{xyz}  {state.z_source}", (x + 292, yy), 0.39, (205, 211, 218), 1)
            yy += row_h


def _draw_header(canvas: np.ndarray, title: str, status: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (28, 32, 38), -1)
    _put_text(canvas, title, (16, 27), 0.72, (245, 248, 250), 2)
    _put_text(canvas, status, (190, 27), 0.48, (150, 210, 255), 1)
    timestamp = time.strftime("%H:%M:%S")
    _put_text(canvas, timestamp, (canvas.shape[1] - 90, 27), 0.48, (190, 198, 205), 1)


def _draw_panel(canvas: np.ndarray, rect: tuple[int, int, int, int], title: str) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 34, 40), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 68, 76), 1)
    _put_text(canvas, title, (x + 10, y + 24), 0.52, (238, 241, 244), 1)
    cv2.line(canvas, (x, y + 34), (x + w, y + 34), (58, 66, 74), 1)


def _inner_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    pad = 10
    title_h = 34
    return (x + pad, y + title_h + pad, w - pad * 2, h - title_h - pad * 2)


def _paste_or_placeholder(
    canvas: np.ndarray,
    image: np.ndarray | None,
    rect: tuple[int, int, int, int],
    message: str,
) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 22, 26), -1)
    if image is None:
        _put_text(canvas, message, (x + 16, y + h // 2), 0.55, (135, 142, 150), 1)
        return
    fitted = _fit_image(image, w, h)
    canvas[y : y + h, x : x + w] = fitted


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.full((height, width, 3), (20, 22, 26), dtype=np.uint8)
    if image.size == 0:
        return out
    ih, iw = image.shape[:2]
    scale = min(width / max(1, iw), height / max(1, ih))
    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
