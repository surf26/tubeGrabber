"""
Phase 7 验收：MotionPlanner 路点序列。

用法:
  python scripts/test_motion_planner.py left.a1 right.b2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.coord_transform import flange_xyz_to_tip_xyz
from perception.slot_mapper import SlotMapper
from planning.command_validator import CommandValidator
from planning.motion_planner import MotionPlanner, format_waypoints
from utils.config_loader import load_config, load_yaml
from world.tube_registry import TubeRegistry


def _build_demo_registry() -> TubeRegistry:
    cfg = load_config()
    rack = load_yaml(cfg["calib"]["rack_layout"])
    mapper = SlotMapper(rack_config=rack)
    registry = TubeRegistry(mapper.all_slot_ids())

    registry.update_slot(
        "left.a1",
        klass="tube",
        confidence=0.95,
        base_xyz=(-180.0, 310.0, 150.0),
        z_source="measured",
    )
    registry.update_slot(
        "right.b2",
        klass="empty",
        confidence=0.97,
        base_xyz=(90.0, 340.0, 120.0),
        z_source="rack_plane",
    )
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 MotionPlanner 测试")
    parser.add_argument("src", type=str, help="源槽，如 left.a1")
    parser.add_argument("dst", type=str, help="目标槽，如 right.b2")
    args = parser.parse_args()

    registry = _build_demo_registry()
    validator = CommandValidator(registry.slot_ids())
    cmd, ok, reason = validator.parse_and_validate(f"{args.src} {args.dst}", registry)
    if not ok or cmd is None:
        print(f"指令无效: {reason}")
        return 1

    planner = MotionPlanner.from_config()
    from_pose = planner.scan_pose
    tcp = planner.tcp_offset_mm

    src_state = registry.get(cmd.src)
    dst_state = registry.get(cmd.dst)

    print(f"指令: {cmd.src} -> {cmd.dst}")
    print(f"起点: scan_pose z={from_pose[2]:.1f} mm")
    print(f"tcp_offset_mm (ee): {tcp}")
    print()

    pick_wps = planner.plan_pick_transit(src_state, from_pose)
    print("=== 抓取路段 (scan -> src approach) ===")
    print(format_waypoints(pick_wps))

    pick_approach = pick_wps[-1].pose_6d
    pick_insert = planner.build_pick_insert_pose(pick_approach)
    pick_retreat = planner.build_retreat_pose(
        pick_insert,
        float(load_config()["motion"]["pick_retreat_mm"]),
    )
    rx, ry, rz = pick_approach[3:6]
    pick_tip = flange_xyz_to_tip_xyz(pick_approach[:3], (rx, ry, rz), np.array(tcp))
    print()
    print("抓取细节:")
    print(f"  approach 法兰 z={pick_approach[2]:.1f} mm  TCP z={pick_tip[2]:.1f} mm")
    print(f"  insert:  z={pick_insert[2]:.1f} mm")
    print(f"  retreat: z={pick_retreat[2]:.1f} mm")

    place_wps = planner.plan_place_transit(dst_state, pick_retreat)
    print()
    print("=== 放置路段 (retreat -> dst approach) ===")
    print(format_waypoints(place_wps))

    place_approach = place_wps[-1].pose_6d
    place_insert = planner.build_place_insert_pose(place_approach)
    place_retreat = planner.build_retreat_pose(
        place_insert,
        float(load_config()["motion"]["place_retreat_mm"]),
    )
    print()
    print("放置细节:")
    print(f"  insert:  z={place_insert[2]:.1f} mm")
    print(f"  retreat: z={place_retreat[2]:.1f} mm")

    return_wps = planner.plan_to_scan(place_retreat)
    print()
    print("=== 回 scan ===")
    print(format_waypoints(return_wps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
