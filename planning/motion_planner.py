"""运动路点规划（不含 SDK 调用）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.coord_transform import (
    flange_xyz_to_tip_xyz,
    load_tcp_offset_mm,
    rotation_matrix_rpy,
    tip_xyz_to_flange_xyz,
)
from utils.config_loader import load_config, load_yaml
from world.tube_registry import SlotState


class MotionPlannerError(RuntimeError):
    """路点规划失败。"""


@dataclass
class Waypoint:
    label: str
    pose_6d: tuple[float, float, float, float, float, float]
    speed: int | None = None


class MotionPlanner:
    def __init__(
        self,
        config: dict[str, Any],
        poses: dict[str, list[float]],
    ) -> None:
        self._config = config
        self._motion = config.get("motion", {})
        self._arm = config.get("arm", {})
        self._transit_z_offset_mm = float(
            config.get("paths", {}).get("transit_z_offset_mm", 80)
        )
        self._approach_height_mm = float(
            self._motion.get("approach_height_mm", 50)
        )
        self._default_speed = int(self._arm.get("default_speed", 20))
        self._approach_speed = int(self._arm.get("approach_speed", 10))
        self._tcp_offset_mm = load_tcp_offset_mm(config=config)

        self._scan_pose = _as_pose6(poses["scan"])
        self._left_region = _as_pose6(poses["left_region"])
        self._right_region = _as_pose6(poses["right_region"])
        self._vertical_pose = _as_pose6(poses["vertical"])

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> MotionPlanner:
        cfg = config or load_config()
        pose_cfg = cfg["poses"]
        poses = {
            "scan": _load_pose_list(pose_cfg["scan_pose"]),
            "left_region": _load_pose_list(pose_cfg["left_region_pose"]),
            "right_region": _load_pose_list(pose_cfg["right_region_pose"]),
            "vertical": _load_pose_list(pose_cfg["vertical_ee_pose"]),
        }
        return cls(cfg, poses)

    @property
    def scan_pose(self) -> tuple[float, float, float, float, float, float]:
        return self._scan_pose

    @property
    def tcp_offset_mm(self) -> tuple[float, float, float]:
        o = self._tcp_offset_mm
        return (float(o[0]), float(o[1]), float(o[2]))

    def plan_to_scan(
        self,
        from_pose: tuple[float, float, float, float, float, float] | None = None,
    ) -> list[Waypoint]:
        if from_pose is None:
            return [
                Waypoint("scan", self._scan_pose, speed=self._default_speed),
            ]

        waypoints: list[Waypoint] = []
        safe = self._raise_pose(from_pose, self._transit_z_offset_mm)
        waypoints.append(Waypoint("raise_to_transit", safe, speed=self._default_speed))

        scan_at_safe = _with_z(self._scan_pose, safe[2])
        if not _xyz_close(scan_at_safe, safe):
            waypoints.append(
                Waypoint("move_above_scan", scan_at_safe, speed=self._default_speed)
            )
        waypoints.append(
            Waypoint("scan", self._scan_pose, speed=self._approach_speed)
        )
        return waypoints

    def plan_transit_to_slot(
        self,
        slot: SlotState,
        side: str,
        from_pose: tuple[float, float, float, float, float, float],
    ) -> list[Waypoint]:
        if slot.base_xyz is None:
            raise MotionPlannerError(f"{slot.slot_id} 缺少 base_xyz")

        side = side.lower()
        if side not in ("left", "right"):
            raise MotionPlannerError(f"无效 side: {side}")

        region = self._left_region if side == "left" else self._right_region
        approach = self.build_approach_pose(slot.base_xyz, self._approach_height_mm)

        waypoints: list[Waypoint] = []
        safe = self._raise_pose(from_pose, self._transit_z_offset_mm)
        waypoints.append(
            Waypoint(f"{slot.slot_id}_raise", safe, speed=self._default_speed)
        )

        region_safe = _with_z(region, safe[2])
        waypoints.append(
            Waypoint(f"{side}_region_transit", region_safe, speed=self._default_speed)
        )
        waypoints.append(
            Waypoint(f"{side}_region", region, speed=self._approach_speed)
        )

        above_slot = _with_z(approach, safe[2])
        waypoints.append(
            Waypoint(f"{slot.slot_id}_above", above_slot, speed=self._default_speed)
        )
        waypoints.append(
            Waypoint(f"{slot.slot_id}_approach", approach, speed=self._approach_speed)
        )
        return waypoints

    def plan_pick_transit(
        self,
        slot: SlotState,
        from_pose: tuple[float, float, float, float, float, float],
    ) -> list[Waypoint]:
        side = slot.slot_id.split(".", 1)[0]
        return self.plan_transit_to_slot(slot, side, from_pose)

    def plan_place_transit(
        self,
        slot: SlotState,
        from_pose: tuple[float, float, float, float, float, float],
    ) -> list[Waypoint]:
        side = slot.slot_id.split(".", 1)[0]
        return self.plan_transit_to_slot(slot, side, from_pose)

    def build_approach_pose(
        self,
        base_xyz: tuple[float, float, float],
        approach_height_mm: float | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """
        base_xyz：视觉给出的抓取点（夹爪 TCP 目标，基坐标 mm）。
        返回法兰应到的 6D 位姿，使 TCP 停在目标上方 approach_height_mm。
        """
        height = (
            self._approach_height_mm
            if approach_height_mm is None
            else float(approach_height_mm)
        )
        x, y, z = (float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2]))
        rx, ry, rz = self._vertical_pose[3:6]
        tip_xyz = (x, y, z + height)
        flange_xyz = tip_xyz_to_flange_xyz(tip_xyz, (rx, ry, rz), self._tcp_offset_mm)
        return (*flange_xyz, rx, ry, rz)

    def build_retreat_pose(
        self,
        current_pose: tuple[float, float, float, float, float, float],
        retreat_mm: float,
    ) -> tuple[float, float, float, float, float, float]:
        x, y, z, rx, ry, rz = current_pose
        return (x, y, z + float(retreat_mm), rx, ry, rz)

    def build_pick_insert_pose(
        self,
        approach_pose: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        insert = float(self._motion.get("pick_insert_mm", 25))
        return self._build_insert_pose(approach_pose, insert)

    def build_place_insert_pose(
        self,
        approach_pose: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        insert = float(self._motion.get("place_insert_mm", 20))
        return self._build_insert_pose(approach_pose, insert)

    def _build_insert_pose(
        self,
        approach_pose: tuple[float, float, float, float, float, float],
        insert_mm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """沿末端 -Z 下降 insert_mm，使 TCP 下探抓取/放置。"""
        x, y, z, rx, ry, rz = approach_pose
        tip = np.array(
            flange_xyz_to_tip_xyz((x, y, z), (rx, ry, rz), self._tcp_offset_mm),
            dtype=np.float64,
        )
        R = rotation_matrix_rpy(rx, ry, rz)
        tip -= R @ np.array([0.0, 0.0, float(insert_mm)])
        flange_xyz = tip_xyz_to_flange_xyz(tip, (rx, ry, rz), self._tcp_offset_mm)
        return (*flange_xyz, rx, ry, rz)

    def _raise_pose(
        self,
        pose: tuple[float, float, float, float, float, float],
        offset_mm: float,
    ) -> tuple[float, float, float, float, float, float]:
        x, y, z, rx, ry, rz = pose
        return (x, y, z + float(offset_mm), rx, ry, rz)


def _load_pose_list(path: str) -> list[float]:
    data = load_yaml(path)
    pose = data["pose"]
    return [
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
        float(pose["rx"]),
        float(pose["ry"]),
        float(pose["rz"]),
    ]


def _as_pose6(values: list[float]) -> tuple[float, float, float, float, float, float]:
    if len(values) < 6:
        raise MotionPlannerError(f"pose 需要 6 个数: {values!r}")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def _with_z(
    pose: tuple[float, float, float, float, float, float],
    z: float,
) -> tuple[float, float, float, float, float, float]:
    return (pose[0], pose[1], z, pose[3], pose[4], pose[5])


def _xyz_close(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
    tol: float = 0.5,
) -> bool:
    return (
        abs(a[0] - b[0]) < tol
        and abs(a[1] - b[1]) < tol
        and abs(a[2] - b[2]) < tol
    )


def format_waypoints(waypoints: list[Waypoint]) -> str:
    lines = [f"{'#':<3} {'label':<24} {'speed':>5}  pose (mm + rad)"]
    lines.append("-" * 80)
    for i, wp in enumerate(waypoints, start=1):
        speed = "-" if wp.speed is None else str(wp.speed)
        p = wp.pose_6d
        pose_str = (
            f"x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f}, "
            f"rx={p[3]:.3f}, ry={p[4]:.3f}, rz={p[5]:.3f}"
        )
        lines.append(f"{i:<3} {wp.label:<24} {speed:>5}  {pose_str}")
    return "\n".join(lines)
