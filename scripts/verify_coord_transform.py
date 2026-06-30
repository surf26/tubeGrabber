"""
Phase 4 验收：scan 位 → YOLO 检测 → 像素+深度+手眼 → 基坐标。

用法:
  python scripts/verify_coord_transform.py           # 真机
  python scripts/verify_coord_transform.py --offline data/captures/scan_xxx_color.png data/captures/scan_xxx_depth.png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver, ArmDriverError
from drivers.camera_driver import CameraDriver, CameraDriverError
from perception.coord_transform import (
    CoordTransformError,
    load_T_ee_cam,
    load_intrinsics,
    pixel_to_base_mm,
    pose_6d_to_matrix,
)
from perception.yolo_detector import YoloDetector
from utils.config_loader import load_config, load_yaml, PROJECT_ROOT


def _pose_dict_to_list(pose: dict) -> list[float]:
    return [
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
        float(pose["rx"]),
        float(pose["ry"]),
        float(pose["rz"]),
    ]


def _pick_detection(detections, prefer_class: str = "tube"):
    if not detections:
        return None
    preferred = [d for d in detections if d.class_name == prefer_class]
    pool = preferred if preferred else detections
    return max(pool, key=lambda d: d.confidence)


def run_offline(color_path: Path, depth_path: Path, pose_6d: list[float]) -> int:
    cfg = load_config()
    yolo_cfg = cfg["yolo"]
    cam_cfg = cfg["camera"]

    color = cv2.imread(str(color_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        print("无法读取 color/depth 图像")
        return 1

    return _run_pipeline(
        color,
        depth,
        pose_6d,
        yolo_cfg,
        cam_cfg,
        save_prefix="offline",
    )


def run_live() -> int:
    cfg = load_config()
    arm_cfg = cfg["arm"]
    cam_cfg = cfg["camera"]
    yolo_cfg = cfg["yolo"]

    scan_data = load_yaml(cfg["poses"]["scan_pose"])
    scan_pose = _pose_dict_to_list(scan_data["pose"])

    print("将移动到 scan_pose 并拍照，确认安全后按 Enter ...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        return 130

    arm = ArmDriver(arm_cfg["ip"], arm_cfg["port"])
    cam = CameraDriver(
        serial=cam_cfg["serial"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg.get("fps", 30),
    )

    try:
        if not arm.connect():
            print("机械臂连接失败")
            return 1
        if not cam.connect():
            print("相机连接失败")
            return 1

        print("移动到 scan_pose ...")
        arm.move_p(scan_pose, speed=arm_cfg["approach_speed"], block=True)
        arm.wait_motion_done()

        pose_6d = list(arm.get_pose_6d())
        print(f"臂姿 mm+rad: {pose_6d}")

        frame = cam.capture()
        return _run_pipeline(
            frame.color,
            frame.depth,
            pose_6d,
            yolo_cfg,
            cam_cfg,
            save_prefix="live",
        )
    except (ArmDriverError, CameraDriverError) as exc:
        print(f"硬件错误: {exc}")
        return 1
    finally:
        cam.disconnect()
        arm.disconnect()


def _run_pipeline(
    color: np.ndarray,
    depth: np.ndarray,
    pose_6d: list[float],
    yolo_cfg: dict,
    cam_cfg: dict,
    save_prefix: str,
) -> int:
    K, dist = load_intrinsics()
    T_ee_cam = load_T_ee_cam()
    T_base_ee = pose_6d_to_matrix(pose_6d)

    class_map = {int(k): v for k, v in yolo_cfg["classes"].items()}
    detector = YoloDetector(
        model_path=yolo_cfg["model_path"],
        conf_threshold=yolo_cfg["conf_threshold"],
        iou_threshold=yolo_cfg["iou_threshold"],
        class_id_to_name=class_map,
    )
    detector.load()

    detections = detector.detect(color)
    print(f"YOLO 检测到 {len(detections)} 个目标")
    for i, det in enumerate(detections):
        print(
            f"  [{i}] {det.class_name} conf={det.confidence:.3f} "
            f"center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f})"
        )

    target = _pick_detection(detections, prefer_class="tube")
    if target is None:
        print("没有可用检测框")
        return 1

    u, v = target.center_uv
    try:
        p_base, debug = pixel_to_base_mm(
            u,
            v,
            depth,
            K,
            dist,
            T_ee_cam,
            T_base_ee,
            depth_min_mm=cam_cfg.get("depth_min_mm", 100),
            depth_max_mm=cam_cfg.get("depth_max_mm", 800),
        )
    except CoordTransformError as exc:
        print(f"坐标变换失败: {exc}")
        return 1

    print("\n=== 坐标变换结果 ===")
    print(f"选中: {target.class_name} conf={target.confidence:.3f}")
    print(f"像素 raw: {debug['uv_raw']}")
    print(f"像素 undist: {debug['uv_undist']}")
    print(f"深度 median: {debug['depth_mm']:.1f} mm")
    print(f"相机系 mm: {debug['p_cam_mm']}")
    print(f"末端系 mm: {debug['p_ee_mm']}")
    print(f"基座系 mm: x={p_base[0]:.2f}, y={p_base[1]:.2f}, z={p_base[2]:.2f}")
    print("请与示教器/实物对比验证")

    out_dir = PROJECT_ROOT / "data" / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    vis = detector.draw(color, detections)
    x, y, z = p_base
    cv2.putText(
        vis,
        f"base: ({x:.1f},{y:.1f},{z:.1f}) mm",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    vis_path = out_dir / f"verify_{save_prefix}_{stamp}.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"标注图: {vis_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 坐标变换验证")
    parser.add_argument("--offline", nargs=2, metavar=("COLOR", "DEPTH"))
    parser.add_argument(
        "--pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="离线模式下的臂姿 mm+rad；缺省用 scan_pose.json",
    )
    args = parser.parse_args()

    if args.offline:
        color_path = Path(args.offline[0])
        depth_path = Path(args.offline[1])
        if args.pose:
            pose_6d = list(args.pose)
        else:
            cfg = load_config()
            scan = load_yaml(cfg["poses"]["scan_pose"])["pose"]
            pose_6d = _pose_dict_to_list(scan)
            print("离线模式：使用 scan_pose.json 作为臂姿")
        return run_offline(color_path, depth_path, pose_6d)

    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())
