"""
Phase 8 真机空跑：连接硬件 → 扫描 → 移动 + 精定位，不夹取。

用法:
  python scripts/test_fsm_dry_run.py left.a1 right.b2
  python scripts/test_fsm_dry_run.py left.a1 right.b2 --no-gripper
  python scripts/test_fsm_dry_run.py --interactive

推荐: python main.py dry-run left.a1 right.b2 --no-gripper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.fsm_factory import build_pick_place_fsm


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8 FSM dry-run")
    parser.add_argument("src", nargs="?", default=None, help="源槽 left.a1")
    parser.add_argument("dst", nargs="?", default=None, help="目标槽 right.b2")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互模式（等同 python main.py）",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="不连接夹爪",
    )
    args = parser.parse_args()

    if args.interactive:
        import main as main_cli

        sys.argv = ["main.py"]
        if args.no_gripper:
            sys.argv.append("--no-gripper")
        return main_cli.main()

    fsm = build_pick_place_fsm(dry_run=True, skip_gripper=args.no_gripper)

    print("=== Phase 8 dry-run ===")
    print("确认工作空间安全，按 Enter 开始（Ctrl+C 取消）...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n已取消")
        return 130

    try:
        if not args.src or not args.dst:
            print("需要 src dst，或使用 --interactive / python main.py")
            return 1
        ok = fsm.run_dry_move(f"{args.src} {args.dst}")
        return 0 if ok else 1
    finally:
        fsm.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
