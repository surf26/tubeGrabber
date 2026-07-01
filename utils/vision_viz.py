"""扫描 / refine 标注与移动过程相机预览。"""

from __future__ import annotations

import sys
import threading
import time
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

    _caption(canvas, "Enter/Space 或终端回车继续", (6, canvas.shape[0] - 6))
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

    _caption(canvas, "Enter/Space 或终端回车继续", (6, canvas.shape[0] - 6))
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


class CameraPreview:
    """移动过程中取流；仅在主线程 tick() 里 imshow（OpenCV Qt 不支持多线程弹窗）。"""

    def __init__(
        self,
        camera,
        *,
        window_name: str = "camera_live",
        fps: float = 15.0,
        font_scale: float = 0.28,
    ) -> None:
        self._camera = camera
        self._window = window_name
        self._interval = 1.0 / max(1.0, fps)
        self._font_scale = font_scale
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._latest = None
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        self._thread = threading.Thread(target=self._capture_loop, name="camera_preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.5)
            self._thread = None
        with self._lock:
            self._latest = None

    def tick(self) -> None:
        """主线程刷新预览（须在 move 等待循环中调用）。"""
        if not self.is_running:
            return
        with self._lock:
            img = None if self._latest is None else self._latest.copy()
        if img is not None:
            _caption(img, "LIVE preview (no YOLO tracking)", (6, 14))
            cv2.imshow(self._window, np.ascontiguousarray(img))
        cv2.waitKey(1)

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                frame = self._camera.capture()
                with self._lock:
                    self._latest = frame.color.copy()
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._interval - elapsed))


class VisionDisplay:
    """FSM 用：scan/refine 弹窗 + 移动预览。"""

    def __init__(self, config: dict[str, Any]) -> None:
        vis = config.get("vision", {})
        self._enabled = bool(vis.get("display_enabled", True))
        self._live_preview = bool(vis.get("live_preview_enabled", True))
        self._scan_wait_ms = int(vis.get("scan_imshow_wait_ms", 0))
        self._refine_wait_ms = int(vis.get("refine_imshow_wait_ms", 0))
        self._font_scale = float(vis.get("font_scale", 0.28))
        self._preview_fps = float(vis.get("live_preview_fps", 15))
        self._preview: CameraPreview | None = None
        self._camera = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind_camera(self, camera) -> None:
        self._camera = camera
        self._preview = CameraPreview(
            camera,
            fps=self._preview_fps,
            font_scale=self._font_scale,
        )

    def show_scan(self, bgr: np.ndarray) -> None:
        if not self._enabled:
            return
        self.stop_live()
        self._hide_live_window()
        img = np.ascontiguousarray(bgr)
        cv2.namedWindow("scan_view", cv2.WINDOW_NORMAL)
        cv2.imshow("scan_view", img)
        cv2.resizeWindow("scan_view", img.shape[1], img.shape[0])
        cv2.waitKey(1)
        self._wait("scan_view", self._scan_wait_ms)

    def show_refine(self, bgr: np.ndarray) -> None:
        if not self._enabled:
            return
        self.stop_live()
        self._hide_live_window()
        img = np.ascontiguousarray(bgr)
        cv2.namedWindow("refine_view", cv2.WINDOW_NORMAL)
        cv2.imshow("refine_view", img)
        cv2.resizeWindow("refine_view", img.shape[1], img.shape[0])
        cv2.waitKey(1)
        self._wait("refine_view", self._refine_wait_ms)

    def start_live(self) -> None:
        if not self._enabled or not self._live_preview or self._preview is None:
            return
        self._preview.start()

    def stop_live(self) -> None:
        if self._preview is not None:
            self._preview.stop()

    def tick_live(self) -> None:
        if self._preview is not None:
            self._preview.tick()

    def _hide_live_window(self) -> None:
        try:
            cv2.destroyWindow("camera_live")
        except cv2.error:
            pass

    def close_all(self) -> None:
        self.stop_live()
        cv2.destroyAllWindows()

    @staticmethod
    def _wait(window: str, wait_ms: int) -> None:
        """等待用户确认继续。"""
        if wait_ms > 0:
            cv2.waitKey(wait_ms)
            return

        # 有终端时直接用 input()，比 OpenCV waitKey 可靠（Qt 窗口常抢不到 Enter）
        if sys.stdin.isatty():
            print(f"[{window}] 查看弹窗图像后，在本终端按 Enter 继续...")
            try:
                input()
            except EOFError:
                pass
            cv2.waitKey(1)
            return

        print(f"[{window}] 继续：点选图像窗口后 Enter/Space")
        qt_continue = {16777220, 16777221, 16777232}
        ascii_continue = {13, 10, 32, ord("c"), ord("C")}
        while True:
            key = cv2.waitKeyEx(30)
            if key != -1 and (
                key in qt_continue or (key & 0xFF) in ascii_continue
            ):
                break
