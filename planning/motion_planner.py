"""运动路点规划（不含 SDK 调用）。"""

from __future__ import annotations

import math
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
        self._safety = config.get("safety", {})
        self._tube = config.get("tube", {})
        self._transit_z_offset_mm = float(
            config.get("paths", {}).get("transit_z_offset_mm", 80)
        )
        self._approach_height_mm = float(
            self._motion.get("approach_height_mm", 50)
        )
        self._refine_height_mm = float(
            self._motion.get("refine_height_mm", self._approach_height_mm)
        )
        self._pick_refine_height_mm = float(
            self._motion.get("pick_refine_height_mm", self._refine_height_mm)
        )
        self._pick_transit_z_mm = self._motion.get("pick_transit_z_mm")
        if self._pick_transit_z_mm is not None:
            self._pick_transit_z_mm = float(self._pick_transit_z_mm)
        self._use_direct_pick_transit = bool(
            self._motion.get("use_direct_pick_transit", True)
        )
        self._carried_extension_below_tcp_mm = float(
            self._tube.get(
                "carried_extension_below_tcp_mm",
                self._tube.get("length_mm", 0.0),
            )
        )
        self._place_bottom_clearance_mm = float(
            self._tube.get("place_bottom_clearance_mm", 10.0)
        )
        self._place_approach_height_mm = float(
            self._motion.get(
                "place_approach_height_mm",
                max(
                    self._approach_height_mm,
                    self._carried_extension_below_tcp_mm
                    + self._place_bottom_clearance_mm,
                ),
            )
        )
        self._carried_obstacle_clearance_mm = float(
            self._tube.get("carried_obstacle_clearance_mm", 30.0)
        )
        self._use_lift_for_place = bool(
            self._motion.get("use_lift_pose_for_place_transit", True)
        )
        self._carried_clearance_z_mm = self._motion.get("carried_clearance_z_mm")
        if self._carried_clearance_z_mm is not None:
            self._carried_clearance_z_mm = float(self._carried_clearance_z_mm)
        self._place_carried_transit_z_mm = self._motion.get("place_carried_transit_z_mm")
        if self._place_carried_transit_z_mm is not None:
            self._place_carried_transit_z_mm = float(self._place_carried_transit_z_mm)
        self._default_speed = int(self._arm.get("default_speed", 20))
        self._approach_speed = int(self._arm.get("approach_speed", 10))
        self._tcp_offset_mm = load_tcp_offset_mm(config=config)

        # 夹爪偏航自适应对齐参数
        gripper_cfg = config.get("gripper", {})
        self._yaw_offset_rad = float(gripper_cfg.get("yaw_offset_rad", 0.0))
        self._finger_default_axis = str(gripper_cfg.get("finger_default_axis", "row"))
        self._max_grasp_yaw_dev_rad = float(
            self._safety.get("max_grasp_yaw_dev_rad", 1.7)
        )

        self._scan_pose = _as_pose6(poses["scan"])
        self._left_region = _as_pose6(poses["left_region"])
        self._right_region = _as_pose6(poses["right_region"])
        self._lift_pose = _as_pose6(poses.get("lift", poses["scan"]))
        self._vertical_pose = _as_pose6(poses["vertical"])
        self._max_tool_tilt_deg = float(self._safety.get("max_tool_tilt_deg", 5.0))
        self._assert_vertical_tool(self._vertical_pose[3:6])

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> MotionPlanner:
        cfg = config or load_config()
        pose_cfg = cfg["poses"]
        poses = {
            "scan": _load_pose_list(pose_cfg["scan_pose"]),
            "left_region": _load_pose_list(pose_cfg["left_region_pose"]),
            "right_region": _load_pose_list(pose_cfg["right_region_pose"]),
            "lift": _load_pose_list(pose_cfg.get("lift_pose", pose_cfg["scan_pose"])),
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

    @property
    def carried_extension_below_tcp_mm(self) -> float:
        return self._carried_extension_below_tcp_mm

    def carried_lowest_z(
        self,
        pose: tuple[float, float, float, float, float, float],
    ) -> float:
        """夹着试管时，工具/试管组合最低点的基坐标 Z。"""
        x, y, z, rx, ry, rz = pose
        R = rotation_matrix_rpy(rx, ry, rz)
        flange = np.array([x, y, z], dtype=np.float64)
        tcp = flange + R @ self._tcp_offset_mm
        tube_bottom = flange + R @ self._carried_tip_offset_mm()
        return float(min(tcp[2], tube_bottom[2]))

    def plan_to_scan(
        self,
        from_pose: tuple[float, float, float, float, float, float] | None = None,
    ) -> list[Waypoint]:
        if from_pose is None:
            return [
                Waypoint("scan", self._scan_pose, speed=self._default_speed),
            ]

        safe_z = max(float(from_pose[2]), self._scan_pose[2])
        if self._carried_clearance_z_mm is not None:
            safe_z = max(safe_z, self._carried_clearance_z_mm)

        waypoints = [
            Waypoint("scan_return_raise", _with_z(from_pose, safe_z), speed=self._default_speed),
        ]
        scan_at_safe = _with_z(self._scan_pose, safe_z)
        if not _xyz_close(scan_at_safe, waypoints[-1].pose_6d):
            waypoints.append(
                Waypoint("move_above_scan", scan_at_safe, speed=self._default_speed)
            )
        waypoints.append(Waypoint("scan", self._scan_pose, speed=self._approach_speed))
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
        safe = self._transit_pose(
            from_pose,
            region[2],
            approach[2],
        )
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

        # Keep the slot approach at the local work height. Reusing the scan
        # height here can make far rack positions unreachable with the vertical
        # grasp posture.
        above_slot = _with_z(approach, max(region[2], approach[2]))
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
        if slot.base_xyz is None:
            raise MotionPlannerError(f"{slot.slot_id} 缺少 base_xyz")

        refine = self.build_pick_refine_pose(slot.base_xyz)
        if self._use_direct_pick_transit:
            # Empty-hand pick transit does not need the carried-tube clearance.
            # Keeping the vertical tool pose at the scan height can make far
            # right-side slots unreachable, so use a lower configurable
            # transit height before descending to the refine observation pose.
            safe_z = max(refine[2], self._pick_transit_z_mm or refine[2])
            return [
                Waypoint(
                    f"{slot.slot_id}_pick_raise",
                    _with_z(from_pose, safe_z),
                    speed=self._default_speed,
                ),
                Waypoint(
                    f"{slot.slot_id}_above_refine",
                    _with_z(refine, safe_z),
                    speed=self._default_speed,
                ),
                Waypoint(
                    f"{slot.slot_id}_pick_refine",
                    refine,
                    speed=self._approach_speed,
                ),
            ]

        side = slot.slot_id.split(".", 1)[0]
        waypoints = self.plan_transit_to_slot(slot, side, from_pose)
        if len(waypoints) >= 2:
            waypoints[-2] = Waypoint(
                f"{slot.slot_id}_above_refine",
                _with_z(refine, max(waypoints[-2].pose_6d[2], refine[2])),
                speed=waypoints[-2].speed,
            )
        waypoints[-1] = Waypoint(
            f"{slot.slot_id}_pick_refine",
            refine,
            speed=self._approach_speed,
        )
        return waypoints

    def plan_place_transit(
        self,
        slot: SlotState,
        from_pose: tuple[float, float, float, float, float, float],
    ) -> list[Waypoint]:
        if self._use_lift_for_place:
            side = slot.slot_id.split(".", 1)[0]
            return self.plan_carried_transit_to_slot(slot, side, from_pose)
        side = slot.slot_id.split(".", 1)[0]
        return self.plan_transit_to_slot(slot, side, from_pose)

    def plan_carried_transit_to_slot(
        self,
        slot: SlotState,
        side: str,
        from_pose: tuple[float, float, float, float, float, float],
    ) -> list[Waypoint]:
        """
        带试管转移到目标槽。

        与普通 transit 相比，先进入示教的 lift_pose 走廊，再进入目标侧
        region。这个路段用于夹着试管时的第二段运动，避免从源槽附近直接
        横穿到目标孔上方。
        """
        if slot.base_xyz is None:
            raise MotionPlannerError(f"{slot.slot_id} 缺少 base_xyz")

        side = side.lower()
        if side not in ("left", "right"):
            raise MotionPlannerError(f"无效 side: {side}")

        refine = self.build_place_approach_pose(slot.base_xyz)

        target_z = max(refine[2], self._required_carried_flange_z(slot.base_xyz[2]))
        if self._place_carried_transit_z_mm is not None:
            target_z = max(target_z, self._place_carried_transit_z_mm)
        departure_z = max(from_pose[2], target_z)
        waypoints: list[Waypoint] = [
            Waypoint(
                f"{slot.slot_id}_carry_raise",
                _with_z(from_pose, departure_z),
                speed=self._default_speed,
            ),
        ]

        # 夹着试管时先在高位走廊横移到目标正上方。最后靠近由
        # PLACE_REFINE 后单独做同 XY 竖直下降，避免斜向插入。
        above_slot = _with_z(refine, target_z)
        waypoints.append(
            Waypoint(f"{slot.slot_id}_above_high", above_slot, speed=self._default_speed)
        )
        return waypoints

    def build_refine_pose(
        self,
        base_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        """视觉精定位位姿：TCP 停在目标上方 refine_height_mm。"""
        return self.build_approach_pose(base_xyz, self._refine_height_mm)

    def build_pick_refine_pose(
        self,
        base_xyz: tuple[float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        """抓取二次定位观察位：比真实抓取 approach 更高，给相机留视野。"""
        return self.build_approach_pose(base_xyz, self._pick_refine_height_mm)

    def build_place_approach_pose(
        self,
        base_xyz: tuple[float, float, float],
        *,
        yaw_rad: float | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """放置位姿：按夹持试管底端留安全高度，而不是只按夹爪 TCP。"""
        return self.build_approach_pose(
            base_xyz, self._place_approach_height_mm, yaw_rad=yaw_rad
        )

    def build_approach_pose(
        self,
        base_xyz: tuple[float, float, float],
        approach_height_mm: float | None = None,
        *,
        yaw_rad: float | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """
        base_xyz：视觉给出的抓取点（夹爪 TCP 目标，基坐标 mm）。
        返回法兰应到的 6D 位姿，使 TCP 停在目标上方 approach_height_mm。
        yaw_rad 非空时覆盖腕部偏航 rz（绕竖直工具 Z，不影响竖直性）。
        """
        height = (
            self._approach_height_mm
            if approach_height_mm is None
            else float(approach_height_mm)
        )
        x, y, z = (float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2]))
        rx, ry, rz = self._vertical_pose[3:6]
        if yaw_rad is not None:
            rz = float(yaw_rad)
        self._assert_vertical_tool((rx, ry, rz))
        tip_xyz = (x, y, z + height)
        flange_xyz = tip_xyz_to_flange_xyz(tip_xyz, (rx, ry, rz), self._tcp_offset_mm)
        return (*flange_xyz, rx, ry, rz)

    def choose_grasp_yaw(self, slot_id: str, registry: Any) -> float | None:
        """
        依据架面 θ 与相邻 tube 的方向，选夹爪偏航 rz（避免手指压到相邻试管）。

        - 左右有邻管 → 手指沿竖直轴 φ_v = θ+90°；
        - 上下有邻管 → 手指沿水平轴 φ_h = θ；
        - 无邻管 → 按 finger_default_axis 默认；
        夹爪 180° 对称，取与中性 rz 就近的等价角；超出 max_grasp_yaw_dev 返回 None。
        θ 未知返回 None（由上层回退到默认竖直姿态）。
        """
        side = slot_id.split(".", 1)[0]
        theta = registry.rack_theta.get(side)
        if theta is None:
            return None

        h, v = registry.neighbor_tube_axes(slot_id)
        phi_h = float(theta)
        phi_v = float(theta) + np.pi / 2.0
        if h:
            psi = phi_v
        elif v:
            psi = phi_h
        else:
            psi = phi_v if self._finger_default_axis == "col" else phi_h

        neutral_rz = float(self._vertical_pose[5])
        rz = _nearest_equiv_angle(psi - self._yaw_offset_rad, neutral_rz, np.pi)
        if abs(_wrap_pi(rz - neutral_rz)) > self._max_grasp_yaw_dev_rad:
            return None
        return rz

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
        if "pick_descend_mm" in self._motion:
            return self.build_descend_pose(
                approach_pose,
                float(self._motion["pick_descend_mm"]),
            )
        insert = float(self._motion.get("pick_insert_mm", 25))
        return self._build_insert_pose(approach_pose, insert)

    def build_place_insert_pose(
        self,
        approach_pose: tuple[float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float]:
        if "place_descend_mm" in self._motion:
            return self.build_descend_pose(
                approach_pose,
                self._checked_place_descend_mm(),
            )
        insert = float(self._motion.get("place_insert_mm", 20))
        return self._build_insert_pose(approach_pose, insert)

    def build_descend_pose(
        self,
        approach_pose: tuple[float, float, float, float, float, float],
        descend_mm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """从当前 approach 位沿 EE +Z 方向下探指定距离。"""
        x, y, z, rx, ry, rz = approach_pose
        tip = np.array(
            flange_xyz_to_tip_xyz((x, y, z), (rx, ry, rz), self._tcp_offset_mm),
            dtype=np.float64,
        )
        R = rotation_matrix_rpy(rx, ry, rz)
        tip += R @ np.array([0.0, 0.0, float(descend_mm)])
        flange_xyz = tip_xyz_to_flange_xyz(tip, (rx, ry, rz), self._tcp_offset_mm)
        return (*flange_xyz, rx, ry, rz)

    def _checked_place_descend_mm(self) -> float:
        descend = float(self._motion.get("place_descend_mm", 0))
        rack_depth = self._tube.get("rack_depth_mm")
        if rack_depth is None:
            return descend

        margin = float(self._tube.get("place_depth_margin_mm", 0))
        max_descend = max(
            0.0,
            self._place_bottom_clearance_mm + float(rack_depth) - margin,
        )
        if descend > max_descend:
            raise MotionPlannerError(
                "place_descend_mm="
                f"{descend:.1f}mm 超过孔深安全下探 {max_descend:.1f}mm "
                f"(rack_depth_mm={float(rack_depth):.1f}, "
                f"bottom_clearance={self._place_bottom_clearance_mm:.1f}, "
                f"margin={margin:.1f})"
            )
        return descend

    def _carried_tip_offset_mm(self) -> np.ndarray:
        return self._tcp_offset_mm + np.array(
            [0.0, 0.0, self._carried_extension_below_tcp_mm],
            dtype=np.float64,
        )

    def _build_insert_pose(
        self,
        approach_pose: tuple[float, float, float, float, float, float],
        insert_mm: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        从 approach 下降到管口，再沿 EE +Z 插入 insert_mm（竖直向下时 TCP 在基坐标下降）。
        pick_insert_mm / place_insert_mm 表示进入管内深度，不含 approach 悬空段。
        """
        x, y, z, rx, ry, rz = approach_pose
        tip = np.array(
            flange_xyz_to_tip_xyz((x, y, z), (rx, ry, rz), self._tcp_offset_mm),
            dtype=np.float64,
        )
        R = rotation_matrix_rpy(rx, ry, rz)
        total_descent = self._approach_height_mm + float(insert_mm)
        tip += R @ np.array([0.0, 0.0, total_descent])
        flange_xyz = tip_xyz_to_flange_xyz(tip, (rx, ry, rz), self._tcp_offset_mm)
        return (*flange_xyz, rx, ry, rz)

    def _raise_pose(
        self,
        pose: tuple[float, float, float, float, float, float],
        offset_mm: float,
    ) -> tuple[float, float, float, float, float, float]:
        x, y, z, rx, ry, rz = pose
        return (x, y, z + float(offset_mm), rx, ry, rz)

    def _transit_pose(
        self,
        from_pose: tuple[float, float, float, float, float, float],
        *clearance_z_values: float,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Build a conservative transit pose without blindly adding height.

        The scan pose is already a high camera pose on this setup; adding another
        transit offset can exceed the arm workspace. Keep the current height when
        it already clears the next segment, otherwise raise only as much as needed.
        """
        x, y, z, rx, ry, rz = from_pose
        safe_z = max(float(z), *(float(v) for v in clearance_z_values))
        return (x, y, safe_z, rx, ry, rz)

    def _assert_vertical_tool(self, rpy: tuple[float, float, float]) -> None:
        """确保末端工具 Z 轴接近基坐标 -Z，避免斜插孔。"""
        R = rotation_matrix_rpy(*rpy)
        tool_z = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cos_angle = float(np.clip(tool_z @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        if angle_deg > self._max_tool_tilt_deg:
            raise MotionPlannerError(
                f"末端姿态不够竖直: tool_z={tool_z.tolist()} "
                f"tilt={angle_deg:.2f}deg > {self._max_tool_tilt_deg:.2f}deg"
            )

    def _required_carried_flange_z(self, rack_z: float) -> float:
        tube_top_above_rack = float(
            self._tube.get(
                "tube_top_above_rack_mm",
                self._tube.get("length_mm", 0.0),
            )
        )
        obstacle_top_z = float(rack_z) + tube_top_above_rack
        required_bottom_z = obstacle_top_z + self._carried_obstacle_clearance_mm
        rx, ry, rz = self._vertical_pose[3:6]
        delta_z = float((rotation_matrix_rpy(rx, ry, rz) @ self._carried_tip_offset_mm())[2])
        return required_bottom_z - delta_z


def _wrap_pi(angle: float) -> float:
    """把角度归一到 (-π, π]。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _nearest_equiv_angle(angle: float, ref: float, period: float) -> float:
    """在 angle + k*period（k∈Z）中取最接近 ref 的等价角。"""
    k = round((float(ref) - float(angle)) / float(period))
    return float(angle) + k * float(period)


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
