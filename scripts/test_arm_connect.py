"""Phase 1 验收：连接机械臂并读取当前位姿。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver, ArmDriverError
from utils.config_loader import load_config


def main() -> int:
    cfg = load_config()
    arm = ArmDriver(cfg["arm"]["ip"], cfg["arm"]["port"])

    print(f"连接 {cfg['arm']['ip']}:{cfg['arm']['port']} ...")
    if not arm.connect():
        print("连接失败")
        return 1

    try:
        pose = arm.get_pose_6d()
        x, y, z, rx, ry, rz = pose
        print("连接成功")
        print(f"位姿 mm+rad: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        print(f"            rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}")
        print("请与示教器 TCP 坐标对比（位置应为 mm 量级）")
    except ArmDriverError as exc:
        print(f"读位姿失败: {exc}")
        return 1
    finally:
        arm.disconnect()
        print("已断开")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
