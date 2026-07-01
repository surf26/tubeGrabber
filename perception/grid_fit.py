"""纯几何：把应落在 cols×rows 规整点阵上的一组 2D 点吸附到整数 (col, row) 索引。

本模块**不涉及**槽位、YOLO、深度、坐标系或配置——只做点阵拟合，方便单元测试。
思路：PCA 求网格两个主方向（自动带出旋转角）→ 沿每轴做 1D 等距栅格吸附（绝对索引，
漏点不影响其余点）→ 残差过大者判为孤儿。适用于架子平放转角 + 相机透视（在真实
XY 平面上做时透视已消除）。

方向说明：返回的 col/row 索引方向由 PCA 主轴符号决定，是任意的；调用方若需要与图像
u/v 或物理约定对齐，应在拿到索引后自行校正（见 slot_mapper）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridFitResult:
    ok: bool
    indices: list[tuple[int, int] | None] = field(default_factory=list)  # 每点 (col,row) 或孤儿 None
    col_pitch: float = 0.0
    row_pitch: float = 0.0
    message: str = ""


def _fit_axis_1d(vals: np.ndarray, count: int) -> tuple[np.ndarray, float, np.ndarray]:
    """
    把一维投影 vals 吸附到 count 条等距栅格线。
    返回 (整数索引, 栅格间距 pitch, 每点残差绝对值)。
    索引以最小投影处为 0（绝对方向由调用方另行校正）。
    """
    n = len(vals)
    vmin = float(vals.min())
    vspan = float(vals.max() - vmin)
    if vspan < 1e-6:  # 全部落在同一条线
        return np.zeros(n, dtype=int), 0.0, np.zeros(n)

    rough = vspan / max(1, count - 1)
    # 稳健 pitch：相邻(排序后)非零间隔中，超过噪声阈值者为"跨线间隔"。
    # 漏一条内部线会让某个间隔变成 ~2*pitch，故取最小值（而非中位数）最接近真实 pitch。
    diffs = np.diff(np.sort(vals))
    nz = diffs[diffs > 0.25 * rough]
    pitch = float(np.min(nz)) if nz.size else rough
    if pitch < 1e-6:
        pitch = rough

    origin = vmin
    idx = np.rint((vals - origin) / pitch).astype(int)
    # 用当前整数索引最小二乘重估 origin+pitch，再吸附，迭代收敛
    for _ in range(3):
        A = np.stack([np.ones(n), idx.astype(float)], axis=1)
        sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
        origin, pitch = float(sol[0]), float(sol[1])
        if abs(pitch) < 1e-6:
            break
        idx = np.rint((vals - origin) / pitch).astype(int)

    # 归一化到 0 起，再夹到 [0,count-1]。残差按夹后的合法格点算：
    # 若某轴的自然层数 > count（指派错误），被夹的点会产生大残差，从而被淘汰。
    origin_n = origin + float(idx.min()) * pitch
    labels_raw = idx - int(idx.min())
    labels = np.clip(labels_raw, 0, count - 1)
    residual = np.abs(vals - (origin_n + labels * pitch))
    return labels, abs(pitch), residual


def fit_grid(
    points: np.ndarray,
    n_cols: int,
    n_rows: int,
    *,
    residual_max_ratio: float = 0.35,
    min_points: int = 6,
) -> GridFitResult:
    """
    把 points (N,2) 吸附到 n_cols×n_rows 规整点阵。

    residual_max_ratio: 点到吸附格点距离 > 此比例×min(格距) 判孤儿(None)。
    min_points: 少于此不拟合（数据不足，结果不可信）。
    """
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    if n < min_points or n < max(n_cols, n_rows):
        return GridFitResult(False, [None] * n, message=f"点数不足({n})")

    Q = P - P.mean(axis=0)
    # PCA 主轴（协方差特征向量，升序）
    _, vecs = np.linalg.eigh(Q.T @ Q)
    axis1, axis2 = vecs[:, -1], vecs[:, -2]
    a = Q @ axis1
    b = Q @ axis2

    # 两种轴→行列指派，取总残差小者（3≠4 时可区分谁是列、谁是行）
    candidates = []
    # 指派 A：a=列, b=行
    la, pa, ra = _fit_axis_1d(a, n_cols)
    lb, pb, rb = _fit_axis_1d(b, n_rows)
    candidates.append((float(np.hypot(ra, rb).sum()), la, lb, pa, pb, ra, rb))
    # 指派 B：a=行, b=列
    la2, pa2, ra2 = _fit_axis_1d(a, n_rows)
    lb2, pb2, rb2 = _fit_axis_1d(b, n_cols)
    candidates.append((float(np.hypot(ra2, rb2).sum()), lb2, la2, pb2, pa2, rb2, ra2))

    candidates.sort(key=lambda c: c[0])
    _, col_labels, row_labels, col_pitch, row_pitch, col_res, row_res = candidates[0]

    thr = residual_max_ratio * max(1e-6, min(col_pitch, row_pitch))
    indices: list[tuple[int, int] | None] = []
    for i in range(n):
        if float(np.hypot(col_res[i], row_res[i])) > thr:
            indices.append(None)
        else:
            indices.append((int(col_labels[i]), int(row_labels[i])))

    return GridFitResult(True, indices, col_pitch, row_pitch, "")
