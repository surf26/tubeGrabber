"""grid_fit 与 slot_mapper 点阵吸附的离线测试（无硬件、无 YOLO）。

直接跑: python tests/test_grid_fit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.detection import Detection
from perception.grid_fit import fit_grid
from perception.slot_mapper import SlotMapper


RACK = {
    "left_col_order": [3, 2, 1],
    "left_row_order": ["a", "b", "c", "d"],
    "right_col_order": [1, 2, 3],
    "right_row_order": ["d", "c", "b", "a"],
}


def _make_grid(n_cols, n_rows, pitch, theta_deg, origin=(0.0, 0.0)):
    """生成旋转 theta 度的规整网格点，返回 (points, truth[(c,r)])。"""
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts, truth = [], []
    for c in range(n_cols):
        for r in range(n_rows):
            p = R @ np.array([c * pitch, r * pitch]) + np.array(origin)
            pts.append(p)
            truth.append((c, r))
    return np.array(pts), truth


def _bijective(pairs) -> bool:
    """真 (c,r) → 拟合 (c,r) 是否为一一映射（允许整体翻转/换轴）。"""
    fwd = {}
    for t, f in pairs:
        if f is None:
            return False
        if t in fwd and fwd[t] != f:
            return False
        fwd[t] = f
    return len(set(fwd.values())) == len(fwd)


def test_rotation():
    for theta in (0, 8, 20, 35, -15):
        pts, truth = _make_grid(3, 4, 40.0, theta)
        pts = pts + np.random.default_rng(0).normal(0, 1.0, pts.shape)  # 1mm 噪声
        res = fit_grid(pts, 3, 4)
        assert res.ok, f"theta={theta} 拟合失败"
        assert _bijective(list(zip(truth, res.indices))), f"theta={theta} 非一一映射"
    print("test_rotation OK")


def test_missing_row_keeps_absolute_index():
    """整行漏检时，剩余点的行索引应保留'缺口'（绝对索引），而非被挤密。"""
    pts, truth = _make_grid(3, 4, 40.0, 12)
    keep = [i for i, (c, r) in enumerate(truth) if r != 2]  # 删掉第 2 行
    pts_k = pts[keep]
    truth_k = [truth[i] for i in keep]
    res = fit_grid(pts_k, 3, 4)
    assert res.ok
    # 每个真 row 对应的拟合 row 应仍是 3 个不同值，且首尾跨度=3（存在缺口），k-means 会得到跨度 2
    row_map = {}
    for (c, r), f in zip(truth_k, res.indices):
        assert f is not None
        row_map.setdefault(r, set()).add(f[1])
    fitted_rows = sorted({list(v)[0] for v in row_map.values()})
    assert len(fitted_rows) == 3, f"应有 3 个行索引: {fitted_rows}"
    assert max(fitted_rows) - min(fitted_rows) == 3, f"应保留缺口(跨度3): {fitted_rows}"
    print("test_missing_row_keeps_absolute_index OK")


def _det(u, v, klass, conf=0.9):
    return Detection(class_name=klass, confidence=conf, bbox=(u - 5, v - 5, u + 5, v + 5), center_uv=(u, v))


def test_slot_mapper_pixel_orientation():
    """无深度 → 像素坐标拟合，验证左右分架 + 方向 + 槽编号。"""
    mapper = SlotMapper(rack_config=RACK, image_width=640, method="lattice")
    dets = []
    # 左架 u=100/140/180, v=100/140/180/220
    for cu, u in enumerate((100, 140, 180)):
        for rv, v in enumerate((100, 140, 180, 220)):
            dets.append(_det(u, v, "tube"))
    # 右架 u=400/440/480
    for cu, u in enumerate((400, 440, 480)):
        for rv, v in enumerate((100, 140, 180, 220)):
            dets.append(_det(u, v, "empty"))

    obs, z_rack = mapper.map(dets)  # 无 geom → z_rack None
    assert z_rack is None
    # 左架最小 u/v 的框 (100,100): col_idx0→col_order[0]=3, row_idx0→'a' → left.a3
    assert obs["left.a3"].klass == "tube", obs["left.a3"]
    # 左架 (180,220): col_idx2→1, row_idx3→'d' → left.d1
    assert obs["left.d1"].klass == "tube"
    # 右架 (400,100): col_idx0→right_col_order[0]=1, row_idx0→right_row_order[0]='d' → right.d1
    assert obs["right.d1"].klass == "empty"
    assert obs["right.a3"].klass == "empty"  # (480,220)
    n_tube = sum(1 for o in obs.values() if o.klass == "tube")
    n_empty = sum(1 for o in obs.values() if o.klass == "empty")
    assert n_tube == 12 and n_empty == 12, (n_tube, n_empty)
    print("test_slot_mapper_pixel_orientation OK")


def test_pixel_plane_to_base():
    from perception.coord_transform import pixel_plane_to_base_mm

    K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    I = np.eye(4)  # 相机=法兰=base 同系，光心在原点，朝 +Z
    # 画面中心 → 视线 [0,0,1] → 打 z=500 平面 → (0,0,500)
    p = pixel_plane_to_base_mm(320.0, 240.0, 500.0, K, dist, I, I)
    assert np.allclose(p, [0, 0, 500], atol=1e-6), p
    # u 右移一个 fx → 方向 x/z=1 → x=500
    p2 = pixel_plane_to_base_mm(320.0 + 400.0, 240.0, 500.0, K, dist, I, I)
    assert np.allclose(p2, [500, 0, 500], atol=1e-6), p2
    print("test_pixel_plane_to_base OK")


if __name__ == "__main__":
    test_rotation()
    test_missing_row_keeps_absolute_index()
    test_slot_mapper_pixel_orientation()
    test_pixel_plane_to_base()
    print("ALL OK")
