#!/usr/bin/env python3
"""
tubeGrabber 主程序 CLI。

用法:
  python main.py                          # 交互模式
  python main.py move left.a1 right.b2    # 完整抓放
  python main.py dry-run left.a1 right.b2 # 空跑（不夹取）
  python main.py scan                     # 连接并扫描
"""

from __future__ import annotations

import argparse
import sys

from tasks.fsm_factory import build_pick_place_fsm
from utils.cli_output import AUTO_CONT, PROMPT_START, err, ok, sub
from utils.config_loader import load_config


HELP_TEXT = """
命令:
  scan              重新扫描 24 槽
  table             显示状态表
  move SRC DST      完整抓放
  dry-run SRC DST   空跑 (移动+定位, 不夹取)
  quit              退出

简写: left.a1 right.b2  (等同 move)
"""


def _continuous_mode(config: dict) -> bool:
    return bool(config.get("runtime", {}).get("continuous_mode", False))


def _confirm_start(config: dict) -> bool:
    if _continuous_mode(config):
        print(f"{PROMPT_START}{AUTO_CONT}")
        return True
    print(PROMPT_START)
    try:
        input()
        return True
    except KeyboardInterrupt:
        print("\n[cancelled]")
        return False


def _print_table(fsm) -> None:
    print(fsm.registry.to_table_str())


def _report_result(ok_flag: bool, fsm) -> None:
    if ok_flag:
        ok()
    else:
        err(fsm.fail_reason)


def _run_interactive(fsm, *, config: dict) -> int:
    if not _confirm_start(config):
        return 130
    if not fsm.connect_and_scan():
        err(f"startup failed: {fsm.fail_reason}")
        return 1

    print(HELP_TEXT)
    _print_table(fsm)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
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
                err(f"scan failed: {fsm.fail_reason}")
            else:
                _print_table(fsm)
            continue
        if cmd == "table":
            _print_table(fsm)
            continue
        if cmd == "move" and len(parts) == 3:
            ok_flag = fsm.execute_move(f"{parts[1]} {parts[2]}", dry=False)
            _report_result(ok_flag, fsm)
            if ok_flag:
                _print_table(fsm)
            continue
        if cmd in ("dry-run", "dry_run", "dryrun") and len(parts) == 3:
            ok_flag = fsm.execute_move(f"{parts[1]} {parts[2]}", dry=True)
            _report_result(ok_flag, fsm)
            if ok_flag:
                _print_table(fsm)
            continue

        if len(parts) == 2 and "." in parts[0] and "." in parts[1]:
            ok_flag = fsm.execute_move(f"{parts[0]} {parts[1]}", dry=False)
            _report_result(ok_flag, fsm)
            if ok_flag:
                _print_table(fsm)
            continue

        err(f"unknown command: {line!r}")
        sub("type 'help' for usage")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="tubeGrabber 试管抓放")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("move", "dry-run", "scan"),
        help="子命令; 省略则进入交互模式",
    )
    parser.add_argument("src", nargs="?", help="源槽 left.a1")
    parser.add_argument("dst", nargs="?", help="目标槽 right.b2")
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="不连接夹爪 (空跑调试用)",
    )
    args = parser.parse_args()

    config = load_config()
    skip_gripper = args.no_gripper or not config.get("gripper", {}).get("enabled", True)
    fsm = build_pick_place_fsm(config=config, dry_run=False, skip_gripper=skip_gripper)

    try:
        if args.command is None:
            return _run_interactive(fsm, config=config)

        if not _confirm_start(config):
            return 130

        if args.command == "scan":
            if not fsm.connect_and_scan():
                err(f"scan failed: {fsm.fail_reason}")
                return 1
            _print_table(fsm)
            return 0

        if args.command == "move":
            if not args.src or not args.dst:
                err("usage: python main.py move left.a1 right.b2")
                return 1
            ok_flag = fsm.run_once(f"{args.src} {args.dst}")
            if ok_flag:
                _print_table(fsm)
            else:
                err(fsm.fail_reason)
            return 0 if ok_flag else 1

        if args.command == "dry-run":
            if not args.src or not args.dst:
                err("usage: python main.py dry-run left.a1 right.b2")
                return 1
            ok_flag = fsm.run_dry_move(f"{args.src} {args.dst}")
            if ok_flag:
                _print_table(fsm)
            else:
                err(fsm.fail_reason)
            return 0 if ok_flag else 1

        return 1
    finally:
        fsm.wait_for_viewer_quit()
        fsm.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
