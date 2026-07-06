"""测试公共夹具：路径注入 + perception.yolo_detector 桩（无 torch/ultralytics 时）。"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 优先真实模块（含 Detection / YoloDetector）
    import perception.yolo_detector  # noqa: F401
except Exception:  # torch/ultralytics 不可用时注入轻量桩
    _stub = types.ModuleType("perception.yolo_detector")

    @dataclass
    class Detection:  # 字段与真实模块保持一致
        class_name: str
        confidence: float
        bbox: tuple
        center_uv: tuple

    class YoloDetector:  # 占位：纯算法单测不会实际推理
        def __init__(self, *args, **kwargs) -> None:
            self._model = None

        def load(self) -> None:  # pragma: no cover
            self._model = object()

        def detect(self, *args, **kwargs):  # pragma: no cover
            return []

    _stub.Detection = Detection
    _stub.YoloDetector = YoloDetector
    sys.modules["perception.yolo_detector"] = _stub
