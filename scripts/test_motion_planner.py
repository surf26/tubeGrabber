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
    print(f"carried_extension_below_tcp={planner.carried_extension_below_tcp_mm:.1f} mm")
    print()

    pick_wps = planner.plan_pick_transit(src_state, from_pose)
    print("=== 抓取路段 (scan -> src approach) ===")
    print(format_waypoints(pick_wps))

    pick_refine = pick_wps[-1].pose_6d
    pick_approach = planner.build_approach_pose(src_state.base_xyz)
    pick_insert = planner.build_pick_insert_pose(pick_approach)
    pick_retreat = planner.build_retreat_pose(
        pick_insert,
        float(load_config()["motion"]["pick_retreat_mm"]),
    )
    motion = load_config()["motion"]
    approach_h = float(motion["approach_height_mm"])
    pick_descend_mm = float(
        motion.get("pick_descend_mm", approach_h + float(motion["pick_insert_mm"]))
    )
    pick_refine_h = float(motion.get("pick_refine_height_mm", motion.get("refine_height_mm", approach_h)))
    tube_z = src_state.base_xyz[2] if src_state.base_xyz else float("nan")

    rx, ry, rz = pick_approach[3:6]
    tcp_off = np.array(tcp)
    tip_refine = flange_xyz_to_tip_xyz(pick_refine[:3], pick_refine[3:6], tcp_off)
    tip_approach = flange_xyz_to_tip_xyz(pick_approach[:3], (rx, ry, rz), tcp_off)
    tip_insert = flange_xyz_to_tip_xyz(pick_insert[:3], (rx, ry, rz), tcp_off)

    print()
    print("抓取细节 (TCP 高度):")
    print(f"  管口 base_z={tube_z:.1f} mm")
    print(f"  refine  TCP z={tip_refine[2]:.1f} mm  (期望 ≈ base_z + {pick_refine_h:.0f})")
    print(f"  approach TCP z={tip_approach[2]:.1f} mm  (期望 ≈ base_z + {approach_h:.0f})")
    print(f"  refine -> approach 下降={tip_refine[2] - tip_approach[2]:.1f} mm")
    print(f"  insert  TCP z={tip_insert[2]:.1f} mm")
    print(f"  TCP 下降量={tip_approach[2] - tip_insert[2]:.1f} mm  (期望 {pick_descend_mm:.0f})")
    if tip_insert[2] >= tip_approach[2]:
        print("  [FAIL] insert TCP 应低于 approach")
        return 1
    print("  [OK] insert TCP z < approach TCP z")

    place_wps = planner.plan_place_transit(dst_state, pick_retreat)
    print()
    print("=== 放置路段 (retreat -> dst approach) ===")
    print(format_waypoints(place_wps))
    rack_z = dst_state.base_xyz[2] if dst_state.base_xyz else float("nan")
    carried_high = place_wps[2].pose_6d if len(place_wps) >= 3 else place_wps[-1].pose_6d
    high_lowest_z = planner.carried_lowest_z(carried_high)
    tube_cfg = load_config().get("tube", {})
    obstacle_top_z = rack_z + float(
        tube_cfg.get("tube_top_above_rack_mm", tube_cfg.get("length_mm", 0.0))
    )
    print(
        f"夹持高位最低点 z={high_lowest_z:.1f} mm, "
        f"其它试管顶部估计 z={obstacle_top_z:.1f} mm, "
        f"净空={high_lowest_z - obstacle_top_z:.1f} mm"
    )

    place_above = place_wps[-1].pose_6d
    planned_place_approach = planner.build_place_approach_pose(dst_state.base_xyz)
    place_approach = (
        place_above[0],
        place_above[1],
        planned_place_approach[2],
        place_above[3],
        place_above[4],
        place_above[5],
    )
    place_insert = planner.build_place_insert_pose(place_approach)
    place_retreat = planner.build_retreat_pose(
        place_insert,
        float(motion["place_retreat_mm"]),
    )
    place_descend_mm = float(
        motion.get(
            "place_descend_mm",
            approach_h + float(motion["place_insert_mm"]),
        )
    )
    rx, ry, rz = place_approach[3:6]
    tip_place_a = flange_xyz_to_tip_xyz(place_approach[:3], (rx, ry, rz), tcp_off)
    tip_place_i = flange_xyz_to_tip_xyz(place_insert[:3], (rx, ry, rz), tcp_off)
    lowest_place_a = planner.carried_lowest_z(place_approach)
    lowest_place_i = planner.carried_lowest_z(place_insert)
    print()
    print("放置细节 (TCP 高度):")
    print(f"  架面 base_z={rack_z:.1f} mm")
    print(f"  approach TCP z={tip_place_a[2]:.1f} mm")
    print(f"  insert  TCP z={tip_place_i[2]:.1f} mm")
    print(f"  TCP 下降量={tip_place_a[2] - tip_place_i[2]:.1f} mm  (期望 {place_descend_mm:.0f})")
    print(f"  approach 夹持最低点 z={lowest_place_a:.1f} mm")
    print(f"  insert  夹持最低点 z={lowest_place_i:.1f} mm")
    if tip_place_i[2] >= tip_place_a[2]:
        print("  [FAIL] insert TCP 应低于 approach")
        return 1
    print("  [OK] insert TCP z < approach TCP z")

    return_wps = planner.plan_to_scan(place_retreat)
    print()
    print("=== 回 scan ===")
    print(format_waypoints(return_wps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
