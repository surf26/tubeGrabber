"""
Phase 8 离线验收：scan 建 registry → 精定位 base_xy 匹配。

用法:
  python scripts/test_refine_offline.py color.png depth.png left.a1
  python scripts/test_refine_offline.py color.png depth.png right.b2 --mode place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.coord_transform import load_T_ee_cam, load_intrinsics, pose_6d_to_matrix
from perception.refine import RefineError, refine_pick_slot, refine_place_slot
from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from utils.config_loader import load_config, load_yaml
from world.tube_registry import TubeRegistry, estimate_z_rack


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 refine 离线测试")
    parser.add_argument("color", type=str, help="彩色图")
    parser.add_argument("depth", type=str, help="16 位深度图")
    parser.add_argument("slot_id", type=str, help="槽位，如 left.a1")
    parser.add_argument(
        "--mode",
        choices=("pick", "place", "auto"),
        default="auto",
        help="pick=tube / place=empty / auto 按 registry klass",
    )
    parser.add_argument(
        "--pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="臂姿 mm+rad，默认 scan_pose",
    )
    args = parser.parse_args()

    color = cv2.imread(args.color)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        print("无法读取 color/depth")
        return 1

    cfg = load_config()
    yolo_cfg = cfg["yolo"]
    cam_cfg = cfg["camera"]
    rack_layout = load_yaml(cfg["calib"]["rack_layout"])
    registry_cfg = cfg.get("registry", {})

    if args.pose:
        pose_6d = list(args.pose)
    else:
        scan = load_yaml(cfg["poses"]["scan_pose"])["pose"]
        pose_6d = [
            float(scan["x"]),
            float(scan["y"]),
            float(scan["z"]),
            float(scan["rx"]),
            float(scan["ry"]),
            float(scan["rz"]),
        ]
        print("使用 scan_pose.json")

    K, dist = load_intrinsics(cfg["calib"]["camera_intrinsics"])
    T_ee_cam = load_T_ee_cam(cfg["calib"]["hand_eye"])
    T_base_ee = pose_6d_to_matrix(pose_6d)

    class_map = {int(k): v for k, v in yolo_cfg["classes"].items()}
    scan_detector = YoloDetector(
        model_path=yolo_cfg["model_path"],
        conf_threshold=yolo_cfg["conf_threshold"],
        iou_threshold=yolo_cfg["iou_threshold"],
        class_id_to_name=class_map,
    )
    scan_detector.load()

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

    mapper = SlotMapper(rack_config=rack_layout, image_width=cam_cfg["width"])
    detections = scan_detector.detect(color)
    observations = mapper.map(
        detections,
        depth,
        K=K,
        dist=dist,
        T_ee_cam=T_ee_cam,
        T_base_ee=T_base_ee,
        depth_min_mm=cam_cfg.get("depth_min_mm", 100),
        depth_max_mm=cam_cfg.get("depth_max_mm", 800),
    )

    try:
        z_rack = estimate_z_rack(
            observations,
            tube_above_rack_mm=float(rack_layout.get("tube_above_rack_mm", 30)),
            default_z_mm=rack_layout.get("default_rack_plane_z_mm"),
        )
    except Exception as exc:
        print(f"无法估计 z_rack: {exc}")
        return 1

    registry = TubeRegistry(mapper.all_slot_ids())
    registry.update_from_scan(observations, z_rack)

    slot_id = args.slot_id.strip().lower()
    state = registry.get(slot_id)
    print(f"Registry: {slot_id} klass={state.klass} base_xyz={state.base_xyz}")
    print()

    max_dist = float(registry_cfg.get("slot_match_max_dist_mm", 15))
    ambiguity_delta = float(registry_cfg.get("refine_ambiguity_min_delta_mm", 2))
    depth_min = cam_cfg.get("depth_min_mm", 100)
    depth_max = cam_cfg.get("depth_max_mm", 800)

    mode = args.mode
    if mode == "auto":
        if state.klass == "tube":
            mode = "pick"
        elif state.klass == "empty":
            mode = "place"
        else:
            print(f"{slot_id} 为 {state.klass}，请指定 --mode pick 或 place")
            return 1

    try:
        if mode == "pick":
            result = refine_pick_slot(
                slot_id,
                registry,
                color,
                depth,
                refine_detector,
                K,
                dist,
                T_ee_cam,
                T_base_ee,
                max_dist_xy_mm=max_dist,
                ambiguity_min_delta_mm=ambiguity_delta,
                depth_min_mm=depth_min,
                depth_max_mm=depth_max,
            )
        else:
            result = refine_place_slot(
                slot_id,
                registry,
                color,
                depth,
                refine_detector,
                K,
                dist,
                T_ee_cam,
                T_base_ee,
                max_dist_xy_mm=max_dist,
                ambiguity_min_delta_mm=ambiguity_delta,
                depth_min_mm=depth_min,
                depth_max_mm=depth_max,
            )
    except RefineError as exc:
        print(f"精定位失败: {exc}")
        return 1

    exp = state.base_xyz
    print(f"模式: {mode}")
    print(f"预期 base_xyz: ({exp[0]:.1f}, {exp[1]:.1f}, {exp[2]:.1f})")
    print(
        f"refine base_xyz: ({result.base_xyz[0]:.1f}, {result.base_xyz[1]:.1f}, "
        f"{result.base_xyz[2]:.1f})"
    )
    print(f"dist_xy: {result.dist_xy_mm:.2f} mm (max_dist {max_dist} mm, "
          f"ambiguity_delta {ambiguity_delta} mm)")
    print(f"conf: {result.confidence:.3f}, z_source: {result.z_source}")
    print(f"pixel_uv: ({result.pixel_uv[0]:.0f}, {result.pixel_uv[1]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
