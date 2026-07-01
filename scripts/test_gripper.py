"""Phase 3 验收：夹爪 Modbus 读位置 + open/close 循环。"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers.arm_driver import ArmDriver, ArmDriverError
from drivers.gripper_driver import GripperDriver, GripperDriverError
from utils.config_loader import load_config


def main() -> int:
    cfg = load_config()
    arm = ArmDriver(cfg["arm"]["ip"], cfg["arm"]["port"])

    print(f"连接机械臂 {cfg['arm']['ip']}:{cfg['arm']['port']} ...")
    try:
        arm.connect()
        gripper = GripperDriver(arm, cfg["gripper"])

        print("配置末端 Modbus RTU ...")
        gripper.setup_modbus()

        pos = gripper.get_position()
        print(f"当前夹爪位置: {pos} 步")

        print("即将循环 open → close 共 3 次，确认夹爪周围安全后按 Enter ...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n已取消")
            return 130

        for i in range(3):
            print(f"--- 第 {i + 1} 次 open ---")
            gripper.open()
            print(f"  位置: {gripper.get_position()}")
            time.sleep(1.0)

            print(f"--- 第 {i + 1} 次 close ---")
            gripper.close()
            print(f"  位置: {gripper.get_position()}")
            time.sleep(1.0)

        print("验收完成")
        return 0

    except (ArmDriverError, GripperDriverError) as exc:
        print(f"失败: {exc}")
        return 1
    finally:
        arm.disconnect()
        print("已断开机械臂")


if __name__ == "__main__":
    raise SystemExit(main())
