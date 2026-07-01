"""YOLO 检测器封装。"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from perception.detection import Detection

__all__ = ["Detection", "YoloDetector"]


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        class_id_to_name: dict[int, str] | None = None,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._class_id_to_name = class_id_to_name or {0: "empty", 1: "tube"}
        self._model: YOLO | None = None

    def load(self) -> None:
        self._model = YOLO(self._model_path)

    def detect(self, bgr_image: np.ndarray) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        results = self._model.predict(
            source=bgr_image,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[Detection] = []
        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            class_name = self._class_id_to_name.get(cls_id, str(cls_id))
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center_uv=(u, v),
                )
            )
        return detections

    def draw(self, bgr_image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """在图像上绘制检测框（调试用）。"""
        canvas = bgr_image.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            color = (0, 255, 0) if det.class_name == "tube" else (0, 165, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            u, v = (int(det.center_uv[0]), int(det.center_uv[1]))
            cv2.circle(canvas, (u, v), 4, (0, 0, 255), -1)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas
