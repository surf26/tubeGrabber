"""
交互标定试管架平面高度：在 scan 位对空孔位多点点击 → 深度 → base Z → 写入 config。

用法:
  # 真机（臂到 scan_pose + 拍照）
  python scripts/calibrate_rack_height.py

  # 离线（已有 color + depth）
  python scripts/calibrate_rack_height.py --offline \\
      data/captures/scan_xxx_color.png data/captures/scan_xxx_depth.png

  # 只预览不写 config
  python scripts/calibrate_rack_height.py --offline color.png depth.png --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver, ArmDriverError
from drivers.camera_driver import CameraDriver, CameraDriverError
from perception.coord_transform import load_T_ee_cam, load_intrinsics, pose_6d_to_matrix
from utils.config_loader import load_config
from utils.pose_io import load_pose_list
from utils.rack_height import pick_rack_plane_z_interactive, save_rack_plane_z_mm


def main() -> int:
    parser = argparse.ArgumentParser(description="交互标定试管架平面 Z")
    parser.add_argument(
        "--offline",
        nargs=2,
        metavar=("COLOR", "DEPTH"),
        help="离线模式：彩色图 + 16 位深度图",
    )
    parser.add_argument(
        "--pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="离线模式臂姿 mm+rad，默认 scan_pose.json",
    )
    parser.add_argument(
        "--min-clicks",
        type=int,
        default=3,
        help="最少点击点数（默认 3）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只显示结果，不写入 config/rack_layout.yaml",
    )
    args = parser.parse_args()

    cfg = load_config()
    cam_cfg = cfg["camera"]
    rack_layout_path = cfg["calib"]["rack_layout"]

    color = depth = None
    pose_6d: list[float]

    if args.offline:
        color_path = Path(args.offline[0])
        depth_path = Path(args.offline[1])
        color = cv2.imread(str(color_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            print("无法读取 color/depth 图像")
            return 1
        if args.pose:
            pose_6d = list(args.pose)
        else:
            pose_6d = load_pose_list(cfg["poses"]["scan_pose"])
            print("使用 scan_pose.json 作为臂姿")
    else:
        arm_cfg = cfg["arm"]
        pose_6d = load_pose_list(cfg["poses"]["scan_pose"])
        speed = arm_cfg["approach_speed"]

        print("真机标定：移动到 scan_pose 后拍照")
        print(f"  scan_pose: z={pose_6d[2]:.1f} mm")
        print("确认安全后按 Enter 开始（Ctrl+C 取消）...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n已取消")
            return 130

        arm = ArmDriver(ip=arm_cfg["ip"], port=arm_cfg["port"])
        cam = CameraDriver(
            serial=cam_cfg["serial"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
        )
        try:
            arm.connect()
            cam.connect()
            arm.move_p(pose_6d, speed=speed)
            arm.wait_motion_done()
            frame = cam.capture()
            color = frame.color
            depth = frame.depth
            pose_6d = list(arm.get_pose_6d())
            print(f"使用拍照时臂姿: z={pose_6d[2]:.1f} mm")
        except (ArmDriverError, CameraDriverError) as exc:
            print(f"硬件错误: {exc}")
            return 1
        finally:
            cam.disconnect()
            arm.disconnect()

    K, dist = load_intrinsics(cfg["calib"]["camera_intrinsics"])
    T_ee_cam = load_T_ee_cam(cfg["calib"]["hand_eye"])
    T_base_ee = pose_6d_to_matrix(pose_6d)

    picked = pick_rack_plane_z_interactive(
        color,
        depth,
        K,
        dist,
        T_ee_cam,
        T_base_ee,
        depth_min_mm=cam_cfg.get("depth_min_mm", 100),
        depth_max_mm=cam_cfg.get("depth_max_mm", 800),
        min_clicks=args.min_clicks,
    )
    if picked is None:
        return 1

    z_rack, _samples = picked
    if args.no_save:
        print(f"\n(未写入) z_rack = {z_rack:.1f} mm")
        return 0

    out_path = save_rack_plane_z_mm(z_rack, rack_layout_path)
    print(f"\n已写入 {out_path}")
    print(f"  default_rack_plane_z_mm: {z_rack:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
