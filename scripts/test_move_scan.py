"""Phase 1 验收：慢速移动到全局拍摄位 scan_pose。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver, ArmDriverError
from utils.config_loader import load_config
from utils.pose_io import load_pose_list


def main() -> int:
    cfg = load_config()
    arm = ArmDriver(cfg["arm"]["ip"], cfg["arm"]["port"])

    scan_path = cfg["poses"]["scan_pose"]
    target = load_pose_list(scan_path)
    speed = cfg["arm"]["approach_speed"]

    print(f"目标 scan_pose（来自 {scan_path}）:")
    print(f"  x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}")
    print(f"  rx={target[3]:.3f}, ry={target[4]:.3f}, rz={target[5]:.3f}")
    print(f"速度: {speed}%")
    print("确认工作空间无障碍后，按 Enter 开始移动（Ctrl+C 取消）...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        return 130

    print(f"连接 {cfg['arm']['ip']}:{cfg['arm']['port']} ...")
    try:
        arm.connect()
        print("开始 move_p ...")
        arm.move_p(target, speed=speed, block=True)
        if not arm.wait_motion_done():
            print("警告: wait_motion_done 超时，请检查是否已到位")

        pose = arm.get_pose_6d()
        print("到位后位姿 mm+rad:")
        print(f"  x={pose[0]:.3f}, y={pose[1]:.3f}, z={pose[2]:.3f}")
        print(f"  rx={pose[3]:.3f}, ry={pose[4]:.3f}, rz={pose[5]:.3f}")
        print("请与示教器及 scan_pose.json 对比")
    except ArmDriverError as exc:
        print(f"运动失败: {exc}")
        return 1
    finally:
        arm.disconnect()
        print("已断开")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
