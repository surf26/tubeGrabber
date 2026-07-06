"""按配置创建感知和槽位状态模块。"""

from __future__ import annotations

from typing import Any

from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from utils.config_loader import load_yaml
from world.tube_registry import TubeRegistry


def build_detector(config: dict[str, Any], *, refine: bool = False) -> YoloDetector:
    """创建并加载 YOLO detector。"""
    yolo_cfg = config["yolo"]
    class_map = {int(k): v for k, v in yolo_cfg["classes"].items()}
    conf = yolo_cfg["conf_threshold"]
    if refine:
        conf = config.get("vision", {}).get("refine_conf_threshold", conf)

    detector = YoloDetector(
        model_path=yolo_cfg["model_path"],
        conf_threshold=conf,
        iou_threshold=yolo_cfg["iou_threshold"],
        class_id_to_name=class_map,
    )
    detector.load()
    return detector


def build_slot_mapper(config: dict[str, Any]) -> SlotMapper:
    """创建 24 槽映射器。"""
    rack = load_yaml(config["calib"]["rack_layout"])
    return SlotMapper(rack_config=rack, image_width=config["camera"]["width"])


def build_registry(mapper: SlotMapper) -> TubeRegistry:
    """按 mapper 槽位顺序创建空状态表。"""
    return TubeRegistry(mapper.all_slot_ids())
