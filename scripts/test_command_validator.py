"""
Phase 7 验收：CommandValidator 解析与规则校验。

用法:
  python scripts/test_command_validator.py
  python scripts/test_command_validator.py "left.a1 right.b2"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning.command_validator import CommandValidator, CommandValidatorError
from utils.config_loader import load_config
from utils.perception_factory import build_registry, build_slot_mapper
from world.tube_registry import TubeRegistry


def _build_demo_registry() -> TubeRegistry:
    cfg = load_config()
    mapper = build_slot_mapper(cfg)
    registry = build_registry(mapper)

    registry.update_slot(
        "left.a1",
        klass="tube",
        confidence=0.95,
        base_xyz=(-180.0, 310.0, 150.0),
        z_source="measured",
    )
    registry.update_slot(
        "left.b2",
        klass="empty",
        confidence=0.98,
        base_xyz=(-150.0, 330.0, 120.0),
        z_source="rack_plane",
    )
    registry.update_slot(
        "right.b2",
        klass="empty",
        confidence=0.97,
        base_xyz=(90.0, 340.0, 120.0),
        z_source="rack_plane",
    )
    registry.update_slot(
        "right.a1",
        klass="tube",
        confidence=0.96,
        base_xyz=(70.0, 320.0, 148.0),
        z_source="measured",
    )
    return registry


def _run_builtin_cases(registry: TubeRegistry, validator: CommandValidator) -> int:
    cases: list[tuple[str, bool, str | None]] = [
        ("left.a1 right.b2", True, None),
        ("LEFT.A1 RIGHT.B2", True, None),
        ("left.a1 left.b2", True, None),
        ("left.b2 left.a1", False, "tube"),
        ("left.a1 right.a1", False, "empty"),
        ("left.a1 left.a1", False, "相同"),
        ("left.a1 right.d3", False, None),
        ("foo bar", False, None),
        ("left.a1", False, None),
    ]

    failed = 0
    print("内置用例:")
    for text, expect_ok, keyword in cases:
        cmd, ok, reason = validator.parse_and_validate(text, registry)
        if keyword and ok == expect_ok and keyword not in reason:
            status = "FAIL"
            failed += 1
        elif ok != expect_ok:
            status = "FAIL"
            failed += 1
        else:
            status = "OK"
        print(f"  [{status}] {text!r:28} -> ok={ok}, reason={reason}")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 CommandValidator 测试")
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help="可选，如 left.a1 right.b2",
    )
    args = parser.parse_args()

    registry = _build_demo_registry()
    validator = CommandValidator(registry.slot_ids())

    failed = _run_builtin_cases(registry, validator)
    print()

    if args.command:
        print(f"用户指令: {args.command!r}")
        try:
            cmd = validator.parse(args.command)
        except CommandValidatorError as exc:
            print(f"解析失败: {exc}")
            return 1
        ok, reason = validator.validate(cmd, registry)
        print(f"  src={cmd.src}, dst={cmd.dst}")
        print(f"  ok={ok}, reason={reason}")
        if not ok:
            return 1
        return failed

    return failed


if __name__ == "__main__":
    raise SystemExit(main())
