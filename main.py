#!/usr/bin/env python3
"""
tube_ws 主程序 CLI（Phase 9）。

用法:
  python main.py                          # 交互模式
  python main.py move left.a1 right.b2    # 完整抓放
  python main.py dry-run left.a1 right.b2 # 空跑（不夹取）
  python main.py scan                       # 仅扫描（需已交互启动或配合 --once）
"""

from __future__ import annotations

import argparse
import sys

from tasks.fsm_factory import build_pick_place_fsm
from utils.config_loader import load_config


HELP_TEXT = """
命令:
  scan                      重新全局扫描 24 槽
  table                     打印当前状态表
  move SRC DST              完整抓放，如 move left.a1 right.b2
  dry-run SRC DST           空跑（移动+精定位，不夹取）
  quit                      退出

也可直接输入: left.a1 right.b2  （等同 move）
"""


def _continuous_mode(config: dict) -> bool:
    return bool(config.get("runtime", {}).get("continuous_mode", False))


def _confirm_start(config: dict) -> bool:
    if _continuous_mode(config):
        print("连续模式已开启：跳过启动 Enter 确认")
        return True
    print("确认工作空间安全，按 Enter 开始（Ctrl+C 取消）...")
    try:
        input()
        return True
    except KeyboardInterrupt:
        print("\n已取消")
        return False


def _print_table(fsm) -> None:
    print(fsm.registry.to_table_str())


def _run_interactive(fsm, *, config: dict, skip_gripper: bool) -> int:
    if not _confirm_start(config):
        return 130
    if not fsm.connect_and_scan():
        print(f"启动失败: {fsm.fail_reason}")
        return 1

    print(HELP_TEXT)
    _print_table(fsm)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "q", "exit"):
            break
        if cmd == "help":
            print(HELP_TEXT)
            continue
        if cmd == "scan":
            if not fsm.scan():
                print(f"扫描失败: {fsm.fail_reason}")
            else:
                _print_table(fsm)
            continue
        if cmd == "table":
            _print_table(fsm)
            continue
        if cmd == "move" and len(parts) == 3:
            ok = fsm.execute_move(f"{parts[1]} {parts[2]}", dry=False)
            print("完成" if ok else f"失败: {fsm.fail_reason}")
            if ok:
                _print_table(fsm)
            continue
        if cmd in ("dry-run", "dry_run", "dryrun") and len(parts) == 3:
            ok = fsm.execute_move(f"{parts[1]} {parts[2]}", dry=True)
            print("完成" if ok else f"失败: {fsm.fail_reason}")
            if ok:
                _print_table(fsm)
            continue

        # left.a1 right.b2 简写
        if len(parts) == 2 and "." in parts[0] and "." in parts[1]:
            ok = fsm.execute_move(f"{parts[0]} {parts[1]}", dry=False)
            print("完成" if ok else f"失败: {fsm.fail_reason}")
            if ok:
                _print_table(fsm)
            continue

        print(f"无法识别命令: {line!r}")
        print("输入 help 查看帮助")

    fsm.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="tube_ws 试管抓放")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("move", "dry-run", "scan"),
        help="子命令；省略则进入交互模式",
    )
    parser.add_argument("src", nargs="?", help="源槽 left.a1")
    parser.add_argument("dst", nargs="?", help="目标槽 right.b2")
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="不连接夹爪（空跑调试用）",
    )
    args = parser.parse_args()

    config = load_config()
    skip_gripper = args.no_gripper or not config.get("gripper", {}).get("enabled", True)
    fsm = build_pick_place_fsm(config=config, dry_run=False, skip_gripper=skip_gripper)

    try:
        if args.command is None:
            return _run_interactive(fsm, config=config, skip_gripper=skip_gripper)

        if not _confirm_start(config):
            return 130

        if args.command == "scan":
            if not fsm.connect_and_scan():
                print(f"扫描失败: {fsm.fail_reason}")
                return 1
            _print_table(fsm)
            return 0

        if args.command == "move":
            if not args.src or not args.dst:
                print("用法: python main.py move left.a1 right.b2")
                return 1
            ok = fsm.run_once(f"{args.src} {args.dst}")
            if ok:
                _print_table(fsm)
            else:
                print(f"失败: {fsm.fail_reason}")
            return 0 if ok else 1

        if args.command == "dry-run":
            if not args.src or not args.dst:
                print("用法: python main.py dry-run left.a1 right.b2")
                return 1
            ok = fsm.run_dry_move(f"{args.src} {args.dst}")
            if ok:
                _print_table(fsm)
            else:
                print(f"失败: {fsm.fail_reason}")
            return 0 if ok else 1

        return 1
    finally:
        fsm.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
