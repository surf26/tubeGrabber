"""Phase 3 验收：夹爪 SDK 初始化 + 不同开度测试。"""

import argparse
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
    parser = argparse.ArgumentParser(description="夹爪不同开度测试")
    parser.add_argument(
        "--positions",
        default="120,130,135,140,115,110",
        help="逗号分隔的夹爪开度，SDK 范围 0=全闭, 1000=全开",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动继续，不等待每一步 Enter",
    )
    args = parser.parse_args()

    positions = _parse_positions(args.positions)

    cfg = load_config()
    arm = ArmDriver(cfg["arm"]["ip"], cfg["arm"]["port"])

    print(f"连接机械臂 {cfg['arm']['ip']}:{cfg['arm']['port']} ...")
    try:
        arm.connect()
        gripper = GripperDriver(arm, cfg["gripper"])

        print("初始化夹爪通信 ...")
        gripper.setup_modbus()

        pos = gripper.get_position()
        print(f"当前夹爪位置: {pos}")

        print()
        print("将依次测试这些夹爪开度：")
        print("  " + " -> ".join(str(p) for p in positions))
        print("说明：0=全闭，1000=全开；观察哪个开度刚好适合抓试管。")
        print("确认夹爪周围安全后按 Enter 开始 ...")
        try:
            input()
        except KeyboardInterrupt:
            print("\n已取消")
            return 130

        for position in positions:
            print(f"--- 开到 {position} ---")
            gripper.move_to(position)
            print(f"  位置: {gripper.get_position()}")
            if not args.auto:
                _wait_enter("观察开度后按 Enter 闭合回 0...")

            print("--- 闭合回 0 ---")
            gripper.close()
            print(f"  位置: {gripper.get_position()}")
            if args.auto:
                time.sleep(1.0)
            else:
                _wait_enter("确认闭合后按 Enter 测下一个开度...")

        print("验收完成")
        return 0

    except (ArmDriverError, GripperDriverError) as exc:
        print(f"失败: {exc}")
        return 1
    finally:
        arm.disconnect()
        print("已断开机械臂")


def _parse_positions(text: str) -> list[int]:
    positions: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value > 1000:
            raise ValueError(f"夹爪开度超出范围 0..1000: {value}")
        positions.append(value)
    if not positions:
        raise ValueError("至少需要一个夹爪开度")
    return positions


def _wait_enter(prompt: str) -> None:
    try:
        input(prompt)
    except KeyboardInterrupt:
        raise


if __name__ == "__main__":
    raise SystemExit(main())
