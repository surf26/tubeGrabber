"""组装 PickPlaceFSM 及依赖模块。"""

from __future__ import annotations

from typing import Any

from drivers.arm_driver import ArmDriver
from drivers.camera_driver import CameraDriver
from planning.command_validator import CommandValidator
from planning.motion_planner import MotionPlanner
from tasks.pick_place_fsm import PickPlaceFSM
from utils.config_loader import load_config
from utils.perception_factory import build_detector, build_registry, build_slot_mapper


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

    detector = build_detector(cfg)
    refine_detector = build_detector(cfg, refine=True)
    mapper = build_slot_mapper(cfg)
    registry = build_registry(mapper)
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
