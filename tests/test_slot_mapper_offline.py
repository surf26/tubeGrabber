"""SlotMapper 度量栅格指派离线单测：合成旋转栅格 + 缺列/缺角/抖动 + 在线估步距。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from perception.slot_mapper import SlotMapper
from perception.yolo_detector import Detection

# --- 合成相机（俯拍针孔，无畸变，架面 z=0）---
_F = 800.0
_CX = 640.0
_CY = 360.0
_H = 500.0          # 相机高出架面 mm
_S = _H / _F        # mm / px
_Z_RACK = 0.0

_PITCH_COL = 24.0
_PITCH_ROW = 20.0

_RACK_CONFIG = {
    "rows": 4,
    "cols": 3,
    "sides": ["left", "right"],
    "left_col_order": [3, 2, 1],
    "left_row_order": ["a", "b", "c", "d"],
    "right_col_order": [1, 2, 3],
    "right_row_order": ["d", "c", "b", "a"],
    "tube_above_rack_mm": 22,
    "default_rack_plane_z_mm": None,
    "slot_pitch_mm": {"col": None, "row": None},
}

_SIDE_ORIGIN = {"left": (-170.0, -30.0), "right": (110.0, -30.0)}


def _calib():
    K = np.array([[_F, 0.0, _CX], [0.0, _F, _CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_base_cam[:3, 3] = [0.0, 0.0, _H]
    T_ee_cam = T_base_cam
    T_base_ee = np.eye(4, dtype=np.float64)
    return K, dist, T_ee_cam, T_base_ee


def _world_to_pixel(x_mm: float, y_mm: float) -> tuple[float, float]:
    u = _CX + x_mm / _S
    v = _CY - y_mm / _S
    return u, v


def _build_side(side, theta, missing_cells, rng):
    """返回 (detections, expected_slot_ids)。"""
    rows = _RACK_CONFIG[f"{side}_row_order"]
    cols = _RACK_CONFIG[f"{side}_col_order"]
    x0, y0 = _SIDE_ORIGIN[side]
    c, s = math.cos(theta), math.sin(theta)

    dets = []
    expected = set()
    for ci in range(len(cols)):
        for ri in range(len(rows)):
            if (ci, ri) in missing_cells:
                continue
            bx = ci * _PITCH_COL + rng.uniform(-0.2, 0.2)
            by = ri * _PITCH_ROW + rng.uniform(-0.2, 0.2)
            xw = x0 + bx * c - by * s
            yw = y0 + bx * s + by * c
            u, v = _world_to_pixel(xw, yw)
            dets.append(
                Detection(
                    class_name="tube",
                    confidence=0.9,
                    bbox=(u - 5.0, v - 5.0, u + 5.0, v + 5.0),
                    center_uv=(u, v),
                )
            )
            expected.add(f"{side}.{rows[ri]}{cols[ci]}")
    return dets, expected


@pytest.mark.parametrize("theta_deg", [0.0, 5.0, -5.0, 10.0, 15.0, -12.0])
def test_grid_assignment_and_theta(theta_deg):
    theta = math.radians(theta_deg)
    rng = np.random.default_rng(1234 + int(theta_deg))
    # 缺中列 (ci=1) + 缺一个角 (0,0)
    missing = {(1, 0), (1, 1), (1, 2), (1, 3), (0, 0)}

    K, dist, T_ee_cam, T_base_ee = _calib()
    left_dets, left_expected = _build_side("left", theta, missing, rng)
    right_dets, right_expected = _build_side("right", theta, missing, rng)
    dets = left_dets + right_dets
    expected = left_expected | right_expected

    mapper = SlotMapper(rack_config=_RACK_CONFIG, image_width=1280)
    obs = mapper.map(
        dets,
        depth=None,
        K=K,
        dist=dist,
        T_ee_cam=T_ee_cam,
        T_base_ee=T_base_ee,
        z_rack_override=_Z_RACK,
    )

    # 指派正确：期望槽全部为 tube，其余保持 unknown（一对一，无串扰）
    assigned = {sid for sid, o in obs.items() if o.klass != "unknown"}
    assert assigned == expected
    for sid in expected:
        assert obs[sid].klass == "tube"
        assert obs[sid].confidence == pytest.approx(0.9)

    # θ 估计误差 < 1°（每侧）
    for side in ("left", "right"):
        est = mapper.last_rack_theta[side]
        d = (est - theta + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < math.radians(1.0), f"{side} θ err={math.degrees(d):.3f}deg"


def test_online_pitch_recovers_missing_column():
    """缺整列时仍应恢复真实步距（近平局偏多层）。"""
    from perception.slot_mapper import _estimate_axis_pitch

    # 3 列只出现列 0 和列 2，间距 2*pitch
    xs = [0.0, 0.0, 0.0, 48.0, 48.0, 48.0]
    pitch = _estimate_axis_pitch(xs, k=3)
    assert pitch == pytest.approx(24.0, abs=1e-6)


def test_empty_detections_returns_all_unknown():
    mapper = SlotMapper(rack_config=_RACK_CONFIG, image_width=1280)
    obs = mapper.map([], depth=None)
    assert len(obs) == 24
    assert all(o.klass == "unknown" for o in obs.values())
