"""检测结果数据类型（轻量，无第三方依赖）。

单独成文件，避免下游(slot_mapper/refine/vision_viz)为了一个 dataclass 而被迫
导入 ultralytics/torch。YoloDetector 从这里复用同一类型。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    center_uv: tuple[float, float]  # u, v
