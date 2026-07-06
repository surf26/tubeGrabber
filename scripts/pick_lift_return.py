"""真机单槽抓取闭环：scan -> 到位 -> 下探 -> 夹住 -> 上提 -> 回观察位。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver
from drivers.camera_driver import CameraDriver
from drivers.gripper_driver import GripperDriver
from perception.coord_transform import load_T_ee_cam, load_intrinsics, pose_6d_to_matrix
from perception.refine import refine_pick_slot
from planning.motion_planner import MotionPlanner, Waypoint, format_waypoints
from utils.config_loader import load_config, load_yaml
from utils.perception_factory import build_detector, build_registry, build_slot_mapper
from utils.vision_viz import VisionDisplay, draw_refine_annotation, draw_scan_annotation
from world.tube_registry import TubeRegistry, estimate_z_rack


def main() -> int:
    parser = argparse.ArgumentParser(description="单槽抓取后上提并回 scan 观察位")
    parser.add_argument("--slot", default=None, help="指定抓取槽位；默认取 scan 中第一个 tube")
    parser.add_argument("--open", type=int, default=135, help="抓取前夹爪张开位置")
    parser.add_argument("--grip", type=int, default=110, help="夹住位置")
    parser.add_argument("--descend-mm", type=float, default=30.0, help="从 approach 下探距离")
    parser.add_argument("--retreat-mm", type=float, default=100.0, help="夹住后上提距离")
    parser.add_argument("--no-refine", action="store_true", help="不做近距离 pick refine")
    args = parser.parse_args()

    cfg = load_config()
    arm = ArmDriver(cfg["arm"]["ip"], cfg["arm"]["port"])
    camera = CameraDriver(
        serial=cfg["camera"]["serial"],
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
        fps=cfg["camera"].get("fps", 30),
    )
    gripper = None
    viz = VisionDisplay(cfg)

    try:
        print("确认工作空间安全，只执行单槽抓取上提并回 scan，按 Enter 开始（Ctrl+C 取消）...")
        input()

        print("[CHECK_HW] 连接机械臂/相机/夹爪...")
        arm.connect()
        camera.connect()
        gripper = GripperDriver(arm, cfg["gripper"])
        gripper.setup_modbus()
        print("[CHECK_HW] OK")

        detector, refine_detector, mapper, registry, planner = _build_modules(cfg)
        K, dist = load_intrinsics(cfg["calib"]["camera_intrinsics"])
        T_ee_cam = load_T_ee_cam(cfg["calib"]["hand_eye"])

        print("[SCAN] 移动到观察位并识别...")
        _execute_waypoints(arm, planner.plan_to_scan(None), cfg, "scan")
        arm.wait_motion_done()
        time.sleep(0.2)
        frame = camera.capture()
        scan_pose = arm.get_pose_6d()
        observations = mapper.map(
            detector.detect(frame.color),
            frame.depth,
            K=K,
            dist=dist,
            T_ee_cam=T_ee_cam,
            T_base_ee=pose_6d_to_matrix(scan_pose),
            depth_min_mm=cfg["camera"].get("depth_min_mm", 100),
            depth_max_mm=cfg["camera"].get("depth_max_mm", 800),
        )
        rack_layout = load_yaml(cfg["calib"]["rack_layout"])
        z_rack = estimate_z_rack(
            observations,
            tube_above_rack_mm=float(rack_layout.get("tube_above_rack_mm", 30)),
            default_z_mm=rack_layout.get("default_rack_plane_z_mm"),
        )
        registry.update_from_scan(observations, z_rack)
        detections = detector.detect(frame.color)
        viz.show_scan(
            draw_scan_annotation(
                frame.color,
                detections,
                observations,
                mapper,
                title="PICK_LIFT_SCAN",
                z_rack=z_rack,
                font_scale=float(cfg.get("vision", {}).get("font_scale", 0.28)),
            )
        )

        slot_id = args.slot or _first_tube_slot(registry)
        state = registry.get(slot_id)
        if state.klass != "tube" or state.base_xyz is None:
            print(f"[ABORT] {slot_id} 当前不是可抓取 tube: klass={state.klass}, base={state.base_xyz}")
            return 1
        print(
            f"[TARGET] {slot_id} tube base=({state.base_xyz[0]:.1f},"
            f"{state.base_xyz[1]:.1f},{state.base_xyz[2]:.1f})"
        )

        waypoints = planner.plan_pick_transit(state, scan_pose)
        print(format_waypoints(waypoints))
        print("[MOVE] 到源槽 approach/refine 高位...")
        _execute_waypoints(arm, waypoints, cfg, "pick_transit")
        approach = waypoints[-1].pose_6d

        if not args.no_refine:
            print("[PICK_REFINE] 停稳拍照并修正抓取点...")
            arm.wait_motion_done()
            time.sleep(0.2)
            frame = camera.capture()
            pose_6d = arm.get_pose_6d()
            detections = refine_detector.detect(frame.color)
            max_dist = float(cfg.get("registry", {}).get("slot_match_max_dist_mm", 15))
            ambiguity = float(cfg.get("registry", {}).get("refine_ambiguity_min_delta_mm", 2))
            result = refine_pick_slot(
                slot_id,
                registry,
                frame.color,
                frame.depth,
                refine_detector,
                K,
                dist,
                T_ee_cam,
                pose_6d_to_matrix(pose_6d),
                max_dist_xy_mm=max_dist,
                ambiguity_min_delta_mm=ambiguity,
                depth_min_mm=cfg["camera"].get("depth_min_mm", 100),
                depth_max_mm=cfg["camera"].get("depth_max_mm", 800),
                detections=detections,
            )
            registry.update_slot(
                slot_id,
                base_xyz=result.base_xyz,
                pixel_uv=result.pixel_uv,
                confidence=result.confidence,
                z_source=result.z_source,
            )
            viz.show_refine(
                draw_refine_annotation(
                    frame.color,
                    detections,
                    slot_id,
                    result,
                    title="PICK_REFINE",
                    font_scale=float(cfg.get("vision", {}).get("font_scale", 0.28)),
                )
            )
            approach = planner.build_approach_pose(result.base_xyz)
            _move_pose(arm, approach, cfg["arm"]["approach_speed"], "pick_refined_approach")

        print(
            f"[CONFIRM] 将夹爪开到 {args.open}，下探 {args.descend_mm:.1f}mm，"
            f"夹到 {args.grip}，上提 {args.retreat_mm:.1f}mm。确认安全按 Enter..."
        )
        input()

        print(f"[GRIPPER] open -> {args.open}")
        gripper.move_to(args.open)

        insert = planner.build_approach_pose(
            registry.get(slot_id).base_xyz,
            max(0.0, float(cfg["motion"].get("approach_height_mm", 50)) - args.descend_mm),
        )
        _move_pose(arm, insert, cfg["arm"]["approach_speed"], "descend_30mm")

        print(f"[GRIPPER] grip -> {args.grip}")
        gripper.move_to(args.grip)

        retreat = planner.build_retreat_pose(insert, args.retreat_mm)
        _move_pose(arm, retreat, cfg["arm"]["default_speed"], "retreat")

        print("[RETURN] 回 scan 观察位...")
        _execute_waypoints(arm, planner.plan_to_scan(retreat), cfg, "return_scan")
        arm.wait_motion_done()
        frame = camera.capture()
        out = Path("data/captures/latest_pick_lift_return.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        cv2.imwrite(str(out), frame.color)
        print(f"[DONE] 已回观察位，保存照片 {out}")
        return 0
    except KeyboardInterrupt:
        print("\n[ABORT] 用户取消")
        return 130
    finally:
        viz.close_all()
        try:
            camera.disconnect()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass


def _build_modules(cfg):
    detector = build_detector(cfg)
    refine_detector = build_detector(cfg, refine=True)
    mapper = build_slot_mapper(cfg)
    registry = build_registry(mapper)
    planner = MotionPlanner.from_config(cfg)
    return detector, refine_detector, mapper, registry, planner


def _first_tube_slot(registry: TubeRegistry) -> str:
    slots = registry.find_tube_slots()
    if not slots:
        raise RuntimeError("scan 未发现 tube，无法抓取")
    return slots[0]


def _execute_waypoints(
    arm: ArmDriver,
    waypoints: list[Waypoint],
    cfg: dict,
    phase: str,
) -> None:
    for wp in waypoints:
        _move_pose(
            arm,
            wp.pose_6d,
            wp.speed or cfg["arm"]["default_speed"],
            f"{phase}/{wp.label}",
        )


def _move_pose(
    arm: ArmDriver,
    pose: tuple[float, float, float, float, float, float],
    speed: int,
    label: str,
) -> None:
    cur = arm.get_pose_6d()
    print(
        f"  move_p [{label}] xyz=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
        f"rpy=({pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f}) | "
        f"当前xyz=({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f})"
    )
    arm.move_p(pose, speed=speed, block=True)


if __name__ == "__main__":
    raise SystemExit(main())
