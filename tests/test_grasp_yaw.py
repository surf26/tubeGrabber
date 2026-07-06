"""夹爪偏航自适应对齐单测：neighbor_tube_axes + choose_grasp_yaw + build_approach_pose。"""

from __future__ import annotations

import math

import pytest

from planning.motion_planner import MotionPlanner
from world.tube_registry import TubeRegistry

_THETA = 0.3  # rad
_NEUTRAL_RZ = 0.0


def _all_slots():
    ids = []
    for side, rows, cols in (
        ("left", "abcd", [3, 2, 1]),
        ("right", "dcba", [1, 2, 3]),
    ):
        for r in rows:
            for c in cols:
                ids.append(f"{side}.{r}{c}")
    return ids


def _make_planner(*, finger_default_axis="row", max_grasp_yaw_dev_rad=1.7):
    config = {
        "arm": {"default_speed": 20, "approach_speed": 10},
        "motion": {},
        "safety": {"max_tool_tilt_deg": 2.0, "max_grasp_yaw_dev_rad": max_grasp_yaw_dev_rad},
        "tube": {},
        "paths": {},
        "gripper": {
            "tcp_offset_mm": [0, 0, 220],
            "yaw_offset_rad": 0.0,
            "finger_default_axis": finger_default_axis,
        },
    }
    vertical = [59.7, 312.2, 297.7, math.pi, 0.0, _NEUTRAL_RZ]
    poses = {
        "scan": [0.0, 300.0, 300.0, math.pi, 0.0, 0.0],
        "left_region": [0.0, 300.0, 300.0, math.pi, 0.0, 0.0],
        "right_region": [0.0, 300.0, 300.0, math.pi, 0.0, 0.0],
        "lift": [0.0, 300.0, 300.0, math.pi, 0.0, 0.0],
        "vertical": vertical,
    }
    return MotionPlanner(config, poses)


def _make_registry(tube_slots=(), theta=_THETA, set_theta=True):
    reg = TubeRegistry(_all_slots())
    for sid in tube_slots:
        reg.update_slot(sid, klass="tube")
    if set_theta:
        reg.set_rack_theta({"left": theta, "right": theta})
    return reg


def _axis_equal(a, b, tol=1e-4):
    """夹爪 180° 对称：判断两角在 mod π 意义下相等。"""
    d = (a - b) % math.pi
    return min(d, math.pi - d) < tol


# --- neighbor_tube_axes ---

def test_neighbor_horizontal_only():
    reg = _make_registry(tube_slots=["left.b1"])
    assert reg.neighbor_tube_axes("left.b2") == (True, False)


def test_neighbor_vertical_only():
    reg = _make_registry(tube_slots=["left.a2"])
    assert reg.neighbor_tube_axes("left.b2") == (False, True)


def test_neighbor_none():
    reg = _make_registry(tube_slots=[])
    assert reg.neighbor_tube_axes("left.b2") == (False, False)


def test_neighbor_both():
    reg = _make_registry(tube_slots=["left.b3", "left.c2"])
    assert reg.neighbor_tube_axes("left.b2") == (True, True)


# --- choose_grasp_yaw ---

def test_yaw_horizontal_neighbor_uses_vertical_axis():
    """左右有邻管 → 手指沿 φ_v = θ+90°（≡ θ-90°）。"""
    planner = _make_planner()
    reg = _make_registry(tube_slots=["left.b1"])
    yaw = planner.choose_grasp_yaw("left.b2", reg)
    assert yaw is not None
    assert _axis_equal(yaw, _THETA + math.pi / 2)


def test_yaw_vertical_neighbor_uses_horizontal_axis():
    """上下有邻管 → 手指沿 φ_h = θ。"""
    planner = _make_planner()
    reg = _make_registry(tube_slots=["left.a2"])
    yaw = planner.choose_grasp_yaw("left.b2", reg)
    assert yaw is not None
    assert _axis_equal(yaw, _THETA)


def test_yaw_no_neighbor_default_row_axis():
    """无邻管 + 默认 row → φ_h = θ。"""
    planner = _make_planner(finger_default_axis="row")
    reg = _make_registry(tube_slots=[])
    yaw = planner.choose_grasp_yaw("left.b2", reg)
    assert yaw is not None
    assert _axis_equal(yaw, _THETA)


def test_yaw_no_neighbor_default_col_axis():
    planner = _make_planner(finger_default_axis="col")
    reg = _make_registry(tube_slots=[])
    yaw = planner.choose_grasp_yaw("left.b2", reg)
    assert yaw is not None
    assert _axis_equal(yaw, _THETA + math.pi / 2)


def test_yaw_theta_unknown_returns_none():
    planner = _make_planner()
    reg = _make_registry(tube_slots=["left.a2"], set_theta=False)
    assert planner.choose_grasp_yaw("left.b2", reg) is None


def test_yaw_exceeds_max_dev_returns_none():
    planner = _make_planner(max_grasp_yaw_dev_rad=0.05)
    reg = _make_registry(tube_slots=["left.a2"])  # vertical → rz≈θ=0.3
    assert planner.choose_grasp_yaw("left.b2", reg) is None


# --- build_approach_pose(yaw_rad=) ---

def test_build_approach_pose_overrides_rz_keeps_vertical():
    planner = _make_planner()
    pose = planner.build_approach_pose((100.0, 200.0, 300.0), yaw_rad=0.7)
    assert pose[5] == pytest.approx(0.7)
    assert pose[3] == pytest.approx(math.pi)  # rx 保持竖直姿态
    assert pose[4] == pytest.approx(0.0)


def test_build_place_approach_pose_accepts_yaw():
    planner = _make_planner()
    pose = planner.build_place_approach_pose((100.0, 200.0, 300.0), yaw_rad=-1.2)
    assert pose[5] == pytest.approx(-1.2)
    assert pose[3] == pytest.approx(math.pi)
