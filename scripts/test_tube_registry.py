"""
Phase 6 离线验收：color 图 → YOLO → SlotMapper → TubeRegistry。

用法:
  python scripts/test_tube_registry.py survey_xxx.jpg
  python scripts/test_tube_registry.py color.png --depth depth.png
  python scripts/test_tube_registry.py color.png --z-rack 120.0
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
from utils.config_loader import load_config, load_yaml
from utils.perception_factory import build_detector, build_registry, build_slot_mapper
from utils.pose_io import load_pose_list
from world.tube_registry import TubeRegistryError, estimate_z_rack


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 TubeRegistry 离线测试")
    parser.add_argument("color", type=str, help="彩色图路径")
    parser.add_argument("--depth", type=str, default=None, help="可选 16 位深度图")
    parser.add_argument(
        "--pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="可选臂姿 mm+rad，用于 base_xyz",
    )
    parser.add_argument(
        "--z-rack",
        type=float,
        default=None,
        help="手动指定 rack 平面 Z（mm）；无 depth 时可用",
    )
    args = parser.parse_args()

    color_path = Path(args.color)
    if not color_path.is_file():
        print(f"找不到图像: {color_path}")
        return 1

    color = cv2.imread(str(color_path))
    if color is None:
        print("无法读取彩色图")
        return 1

    depth = None
    if args.depth:
        depth_path = Path(args.depth)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print("无法读取深度图")
            return 1

    cfg = load_config()
    cam_cfg = cfg["camera"]
    rack_layout = load_yaml(cfg["calib"]["rack_layout"])

    detector = build_detector(cfg)
    detections = detector.detect(color)
    print(f"YOLO 检测数: {len(detections)}")

    mapper = build_slot_mapper(cfg)

    K = dist = T_ee_cam = T_base_ee = None
    if depth is not None:
        K, dist = load_intrinsics(cfg["calib"]["camera_intrinsics"])
        T_ee_cam = load_T_ee_cam(cfg["calib"]["hand_eye"])
        if args.pose:
            pose_6d = list(args.pose)
        else:
            pose_6d = load_pose_list(cfg["poses"]["scan_pose"])
            print("使用 scan_pose.json 作为臂姿（离线近似）")
        T_base_ee = pose_6d_to_matrix(pose_6d)

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

    if args.z_rack is not None:
        z_rack = args.z_rack
        print(f"使用手动 z_rack = {z_rack:.1f} mm")
    else:
        try:
            z_rack = estimate_z_rack(
                observations,
                tube_above_rack_mm=float(rack_layout.get("tube_above_rack_mm", 30)),
                default_z_mm=rack_layout.get("default_rack_plane_z_mm"),
            )
            print(f"估计 z_rack = {z_rack:.1f} mm")
        except TubeRegistryError as exc:
            print(exc)
            print("提示: 无 depth 时可加 --z-rack 120.0 做离线测试")
            return 1

    registry = build_registry(mapper)
    registry.update_from_scan(observations, z_rack)

    print()
    print(registry.to_table_str())

    empty_slots = registry.find_empty_slots()
    tube_slots = registry.find_tube_slots()
    unknown_n = sum(1 for sid in registry.slot_ids() if registry.get(sid).klass == "unknown")

    print()
    print(f"empty ({len(empty_slots)}): {', '.join(empty_slots)}")
    print(f"tube  ({len(tube_slots)}): {', '.join(tube_slots)}")
    print(f"unknown: {unknown_n}, total: {len(registry.slot_ids())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
