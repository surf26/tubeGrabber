"""真机放置当前夹持试管：scan -> 目标孔 -> 下探 -> 松开 -> 上提 -> 回观察位。"""

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
from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from planning.motion_planner import Waypoint, MotionPlanner, format_waypoints
from utils.config_loader import load_config, load_yaml
from utils.vision_viz import VisionDisplay, draw_scan_annotation
from world.tube_registry import TubeRegistry, estimate_z_rack


def main() -> int:
    parser = argparse.ArgumentParser(description="将当前夹持的试管放入指定空槽")
    parser.add_argument("slot", help="目标槽位，如 left.a1 / right.a1")
    parser.add_argument("--release", type=int, default=120, help="放置松开位置")
    parser.add_argument("--insert-mm", type=float, default=None, help="覆盖配置中的放置插入深度")
    parser.add_argument("--retreat-mm", type=float, default=100.0, help="松开后上提距离")
    parser.add_argument("--z-rack", type=float, default=None, help="手动指定架面 Z，避免被夹持试管干扰估计")
    parser.add_argument("--min-approach-flange-z", type=float, default=220.0, help="目标 approach 法兰 Z 下限")
    parser.add_argument("--min-insert-flange-z", type=float, default=150.0, help="目标 insert 法兰 Z 下限")
    args = parser.parse_args()

    cfg = load_config()
    arm = ArmDriver(cfg["arm"]["ip"], cfg["arm"]["port"])
    camera = CameraDriver(
        serial=cfg["camera"]["serial"],
        width=cfg["camera"]["width"],
        height=cfg["camera"]["height"],
        fps=cfg["camera"].get("fps", 30),
    )
    viz = VisionDisplay(cfg)

    try:
        print(f"确认目标 {args.slot} 周围安全，按 Enter 开始放置（Ctrl+C 取消）...")
        input()

        print("[CHECK_HW] 连接机械臂/相机/夹爪...")
        arm.connect()
        camera.connect()
        viz.bind_camera(camera)
        gripper = GripperDriver(arm, cfg["gripper"])
        gripper.setup_modbus(skip_initial_close=True)
        print("[CHECK_HW] OK")

        detector, mapper, registry, planner = _build_modules(cfg)
        K, dist = load_intrinsics(cfg["calib"]["camera_intrinsics"])
        T_ee_cam = load_T_ee_cam(cfg["calib"]["hand_eye"])

        print("[SCAN] 回观察位并识别目标孔...")
        current = arm.get_pose_6d()
        _execute_waypoints(arm, planner.plan_to_scan(current), cfg, viz, "scan")
        arm.wait_motion_done()
        time.sleep(0.2)
        frame = camera.capture()
        scan_pose = arm.get_pose_6d()
        detections = detector.detect(frame.color)
        observations = mapper.map(
            detections,
            frame.depth,
            K=K,
            dist=dist,
            T_ee_cam=T_ee_cam,
            T_base_ee=pose_6d_to_matrix(scan_pose),
            depth_min_mm=cfg["camera"].get("depth_min_mm", 100),
            depth_max_mm=cfg["camera"].get("depth_max_mm", 800),
        )
        rack_layout = load_yaml(cfg["calib"]["rack_layout"])
        if args.z_rack is None:
            z_rack = estimate_z_rack(
                observations,
                tube_above_rack_mm=float(rack_layout.get("tube_above_rack_mm", 30)),
                default_z_mm=rack_layout.get("default_rack_plane_z_mm"),
            )
        else:
            z_rack = float(args.z_rack)
            print(f"[SCAN] 使用手动 z_rack={z_rack:.1f}mm")
        registry.update_from_scan(observations, z_rack)
        viz.show_scan(
            draw_scan_annotation(
                frame.color,
                detections,
                observations,
                mapper,
                title="PLACE_HELD_SCAN",
                z_rack=z_rack,
                font_scale=float(cfg.get("vision", {}).get("font_scale", 0.28)),
            )
        )

        dst = registry.get(args.slot)
        if dst.klass != "empty" or dst.base_xyz is None:
            print(f"[ABORT] {args.slot} 不是可放置空槽: klass={dst.klass}, base={dst.base_xyz}")
            return 1
        print(
            f"[TARGET] {args.slot} empty base=({dst.base_xyz[0]:.1f},"
            f"{dst.base_xyz[1]:.1f},{dst.base_xyz[2]:.1f}) z_src={dst.z_source}"
        )

        waypoints = planner.plan_place_transit(dst, scan_pose)
        approach = waypoints[-1].pose_6d
        if approach[2] < args.min_approach_flange_z:
            print(
                f"[ABORT] approach 法兰 Z={approach[2]:.1f}mm < "
                f"{args.min_approach_flange_z:.1f}mm，拒绝低位放置"
            )
            return 1
        print(format_waypoints(waypoints))
        print("[MOVE] 到目标槽 place/refine 高位...")
        _execute_waypoints(arm, waypoints, cfg, viz, "place_transit")

        print(
            f"[CONFIRM] 将下探放置，松开到 {args.release}，上提 {args.retreat_mm:.1f}mm。"
            "确认安全按 Enter..."
        )
        input()

        if args.insert_mm is None:
            insert = planner.build_place_insert_pose(approach)
        else:
            old_insert = cfg.setdefault("motion", {}).get("place_insert_mm")
            cfg["motion"]["place_insert_mm"] = float(args.insert_mm)
            planner = MotionPlanner.from_config(cfg)
            insert = planner.build_place_insert_pose(approach)
            cfg["motion"]["place_insert_mm"] = old_insert
        if insert[2] < args.min_insert_flange_z:
            print(
                f"[ABORT] insert 法兰 Z={insert[2]:.1f}mm < "
                f"{args.min_insert_flange_z:.1f}mm，拒绝下探"
            )
            return 1
        _move_pose(arm, insert, cfg["arm"]["approach_speed"], "place_insert", viz)

        print(f"[GRIPPER] release -> {args.release}")
        gripper.move_to(args.release)

        retreat = planner.build_retreat_pose(insert, args.retreat_mm)
        _move_pose(arm, retreat, cfg["arm"]["default_speed"], "place_retreat", viz)

        print("[RETURN] 回 scan 观察位...")
        _execute_waypoints(arm, planner.plan_to_scan(retreat), cfg, viz, "return_scan")
        arm.wait_motion_done()
        frame = camera.capture()
        out = Path("data/captures/latest_place_held_return.png")
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
    class_map = {int(k): v for k, v in cfg["yolo"]["classes"].items()}
    detector = YoloDetector(
        cfg["yolo"]["model_path"],
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"],
        class_map,
    )
    detector.load()
    rack = load_yaml(cfg["calib"]["rack_layout"])
    mapper = SlotMapper(rack_config=rack, image_width=cfg["camera"]["width"])
    registry = TubeRegistry(mapper.all_slot_ids())
    planner = MotionPlanner.from_config(cfg)
    return detector, mapper, registry, planner


def _execute_waypoints(
    arm: ArmDriver,
    waypoints: list[Waypoint],
    cfg: dict,
    viz: VisionDisplay,
    phase: str,
) -> None:
    for wp in waypoints:
        _move_pose(
            arm,
            wp.pose_6d,
            wp.speed or cfg["arm"]["default_speed"],
            f"{phase}/{wp.label}",
            viz,
        )


def _move_pose(
    arm: ArmDriver,
    pose: tuple[float, float, float, float, float, float],
    speed: int,
    label: str,
    viz: VisionDisplay,
) -> None:
    cur = arm.get_pose_6d()
    print(
        f"  move_p [{label}] xyz=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
        f"rpy=({pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f}) | "
        f"当前xyz=({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f})"
    )
    viz.start_live()
    try:
        arm.move_p(pose, speed=speed, block=True)
    finally:
        viz.stop_live()


if __name__ == "__main__":
    raise SystemExit(main())
