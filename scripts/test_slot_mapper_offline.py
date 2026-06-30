"""
Phase 5 离线验收：color 图 → YOLO → SlotMapper → 打印 24 槽 + 可视化。

用法:
  python scripts/test_slot_mapper_offline.py data/captures/scan_xxx_color.png
  python scripts/test_slot_mapper_offline.py color.png --depth depth.png
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
from perception.slot_mapper import SlotMapper
from perception.yolo_detector import YoloDetector
from utils.config_loader import load_config, load_yaml, PROJECT_ROOT
from utils.vision_viz import draw_scan_annotation


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 SlotMapper 离线测试")
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
        "--out",
        type=str,
        default=None,
        help="输出标注图路径，默认 data/captures/slot_map_*.png",
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
    yolo_cfg = cfg["yolo"]
    cam_cfg = cfg["camera"]

    class_map = {int(k): v for k, v in yolo_cfg["classes"].items()}
    detector = YoloDetector(
        model_path=yolo_cfg["model_path"],
        conf_threshold=yolo_cfg["conf_threshold"],
        iou_threshold=yolo_cfg["iou_threshold"],
        class_id_to_name=class_map,
    )
    detector.load()
    detections = detector.detect(color)
    print(f"YOLO 检测数: {len(detections)}")

    mapper = SlotMapper(
        rack_config=load_yaml(cfg["calib"]["rack_layout"]),
        image_width=cam_cfg["width"],
    )

    K = dist = T_ee_cam = T_base_ee = None
    if depth is not None:
        K, dist = load_intrinsics(cfg["calib"]["camera_intrinsics"])
        T_ee_cam = load_T_ee_cam(cfg["calib"]["hand_eye"])
        if args.pose:
            pose_6d = list(args.pose)
        else:
            scan = load_yaml(cfg["poses"]["scan_pose"])["pose"]
            pose_6d = [
                scan["x"],
                scan["y"],
                scan["z"],
                scan["rx"],
                scan["ry"],
                scan["rz"],
            ]
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

    print()
    print(mapper.to_table_str(observations))

    tube_n = sum(1 for o in observations.values() if o.klass == "tube")
    empty_n = sum(1 for o in observations.values() if o.klass == "empty")
    unknown_n = sum(1 for o in observations.values() if o.klass == "unknown")
    print()
    print(f"统计: tube={tube_n}, empty={empty_n}, unknown={unknown_n}, total=24")

    vis_det = draw_scan_annotation(
        color,
        detections,
        observations,
        mapper,
        title="OFFLINE",
        font_scale=float(cfg.get("vision", {}).get("font_scale", 0.28)),
    )

    out_dir = PROJECT_ROOT / "data" / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = out_dir / f"slot_map_{color_path.stem}.png"

    cv2.imwrite(str(out_path), vis_det)
    print(f"标注图: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
