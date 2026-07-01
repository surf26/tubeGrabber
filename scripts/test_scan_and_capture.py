"""
Phase 2+ 联调：机械臂到 scan_pose → 停稳 → 拍照 → 存图。

比 test_camera_capture 多一步：验证「拍摄位 + 相机」整条链路。
Phase 4 坐标验证也会在这个位姿上做。
"""

import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver, ArmDriverError
from drivers.camera_driver import CameraDriver, CameraDriverError
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


def main() -> int:
    cfg = load_config()
    arm_cfg = cfg["arm"]
    cam_cfg = cfg["camera"]

    scan_data = load_yaml(cfg["poses"]["scan_pose"])
    scan_pose = _pose_dict_to_list(scan_data["pose"])
    speed = arm_cfg["approach_speed"]

    print("目标: 移动到 scan_pose 后拍照")
    print(f"  scan_pose: x={scan_pose[0]:.1f}, y={scan_pose[1]:.1f}, z={scan_pose[2]:.1f}")
    print("确认工作空间安全，按 Enter 开始（Ctrl+C 取消）...")
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

    out_dir = PROJECT_ROOT / "data" / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        print("连接机械臂...")
        arm.connect()

        print("连接相机...")
        cam.connect()

        print(f"move_p → scan_pose (speed={speed}%)...")
        arm.move_p(scan_pose, speed=speed, block=True)
        if not arm.wait_motion_done():
            print("警告: wait_motion_done 超时")

        arm_pose = arm.get_pose_6d()
        print(f"臂到位: x={arm_pose[0]:.1f}, y={arm_pose[1]:.1f}, z={arm_pose[2]:.1f}")

        print("拍照...")
        frame = cam.capture()

        u = cam_cfg["width"] // 2
        v = cam_cfg["height"] // 2
        depth_center = int(frame.depth[v, u])
        depth_min = cam_cfg.get("depth_min_mm", 100)
        depth_max = cam_cfg.get("depth_max_mm", 800)

        print(f"color={frame.color.shape}, depth={frame.depth.shape}")
        print(f"中心 ({u},{v}) 深度 = {depth_center} mm")

        color_path = out_dir / f"scan_{stamp}_color.png"
        depth_path = out_dir / f"scan_{stamp}_depth.png"
        cv2.imwrite(str(color_path), frame.color)
        cv2.imwrite(str(depth_path), frame.depth)
        print(f"已保存: {color_path}")
        print(f"已保存: {depth_path}")

        if depth_center == 0:
            print("验收失败: 中心深度为 0")
            return 1
        if not (depth_min <= depth_center <= depth_max):
            print(f"警告: 中心深度 {depth_center} 不在 [{depth_min}, {depth_max}] mm")
        else:
            print("深度范围检查通过")

        return 0

    except (ArmDriverError, CameraDriverError) as exc:
        print(f"失败: {exc}")
        return 1
    finally:
        cam.disconnect()
        arm.disconnect()
        print("已断开臂与相机")


if __name__ == "__main__":
    raise SystemExit(main())
