"""组装 PickPlaceFSM 及依赖模块。"""

from __future__ import annotations

from typing import Any

from drivers.arm_driver import ArmDriver
from drivers.camera_driver import CameraDriver
from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from planning.command_validator import CommandValidator
from planning.motion_planner import MotionPlanner
from tasks.pick_place_fsm import PickPlaceFSM
from utils.config_loader import load_config, load_yaml
from world.tube_registry import TubeRegistry


def build_pick_place_fsm(
    config: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    skip_gripper: bool = False,
) -> PickPlaceFSM:
    cfg = config or load_config()

    arm = ArmDriver(ip=cfg["arm"]["ip"], port=cfg["arm"]["port"])
    cam = CameraDriver(
        serial=cfg["camera"]["serial"],
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
        fps=cfg["camera"]["fps"],
    )

    gripper = None  # CHECK_HW 连接臂后按需创建 GripperDriver

    yolo_cfg = cfg["yolo"]
    class_map = {int(k): v for k, v in yolo_cfg["classes"].items()}
    detector = YoloDetector(
        model_path=yolo_cfg["model_path"],
        conf_threshold=yolo_cfg["conf_threshold"],
        iou_threshold=yolo_cfg["iou_threshold"],
        class_id_to_name=class_map,
    )
    detector.load()

    refine_conf = cfg.get("vision", {}).get(
        "refine_conf_threshold",
        yolo_cfg["conf_threshold"],
    )
    refine_detector = YoloDetector(
        model_path=yolo_cfg["model_path"],
        conf_threshold=refine_conf,
        iou_threshold=yolo_cfg["iou_threshold"],
        class_id_to_name=class_map,
    )
    refine_detector.load()

    rack = load_yaml(cfg["calib"]["rack_layout"])
    mapper_cfg = cfg.get("mapper", {})
    mapper = SlotMapper(
        rack_config=rack,
        image_width=cfg["camera"]["width"],
        method=mapper_cfg.get("method", "lattice"),
        residual_max_ratio=float(mapper_cfg.get("residual_max_ratio", 0.35)),
        min_points_for_fit=int(mapper_cfg.get("min_points_for_fit", 6)),
        side_max_count=int(mapper_cfg.get("side_max_count", 12)),
    )
    registry = TubeRegistry(mapper.all_slot_ids())
    planner = MotionPlanner.from_config(cfg)
    validator = CommandValidator(registry.slot_ids())

    return PickPlaceFSM(
        arm=arm,
        camera=cam,
        gripper=gripper,
        detector=detector,
        refine_detector=refine_detector,
        mapper=mapper,
        registry=registry,
        planner=planner,
        validator=validator,
        config=cfg,
        dry_run=dry_run,
        skip_gripper=skip_gripper,
    )
