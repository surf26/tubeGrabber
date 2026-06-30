"""像素 + 深度 + 手眼 + 臂姿 → 基坐标系。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from utils.config_loader import load_yaml


class CoordTransformError(RuntimeError):
    """坐标变换失败。"""


def load_intrinsics(path: str = "config/camera_intrinsics.yaml") -> tuple[np.ndarray, np.ndarray]:
    """返回 K (3x3), dist (5,)。"""
    data = load_yaml(path)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["distortion_coeffs"], dtype=np.float64)
    return K, dist


def load_T_ee_cam(path: str = "config/hand_eye.yaml") -> np.ndarray:
    """
    加载 eye-in-hand 手眼矩阵 T_ee_cam（4x4）。
    yaml 中平移单位为 m，返回矩阵平移部分转为 mm。
    """
    data = load_yaml(path)
    T = np.array(data["T_ee_cam"], dtype=np.float64)
    unit = data.get("translation_unit", "m")
    # 手眼矩阵平移单位为 m，返回矩阵平移部分转为 mm
    if unit == "m":
        T = T.copy()
        T[:3, 3] *= 1000.0
    return T


def undistort_uv(u: float, v: float, K: np.ndarray, dist: np.ndarray) -> tuple[float, float]:
    """畸变像素 → 归一化平面上的无畸变像素（仍用 K 内参）。"""
    pts = np.array([[[u, v]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return float(undist[0, 0, 0]), float(undist[0, 0, 1])


def sample_depth_mm(
    depth: np.ndarray,
    u: int,
    v: int,
    *,
    window: int = 5,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
) -> float | None:
    """在 (u,v) 邻域取有效深度 median，单位 mm。"""
    h, w = depth.shape[:2]
    half = window // 2
    u0 = max(0, u - half)
    u1 = min(w, u + half + 1)
    v0 = max(0, v - half)
    v1 = min(h, v + half + 1)

    patch = depth[v0:v1, u0:u1].astype(np.float64)
    valid = patch[(patch > 0) & (patch >= depth_min_mm) & (patch <= depth_max_mm)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_camera_mm(u: float, v: float, depth_mm: float, K: np.ndarray) -> np.ndarray:
    """无畸变像素 + 深度 → 相机坐标系点 (3,) mm。"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth_mm / fx
    y = (v - cy) * depth_mm / fy
    z = depth_mm
    return np.array([x, y, z], dtype=np.float64)


def load_tcp_offset_mm(
    gripper_cfg: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """加载法兰→夹爪 TCP 平移（ee 坐标系，mm）。"""
    if gripper_cfg is None:
        if config is None:
            from utils.config_loader import load_config

            config = load_config()
        gripper_cfg = config.get("gripper", {})
    offset = gripper_cfg.get("tcp_offset_mm", [0.0, 0.0, 0.0])
    return np.array(offset, dtype=np.float64)


def tip_xyz_to_flange_xyz(
    tip_xyz: tuple[float, float, float] | np.ndarray,
    rpy: tuple[float, float, float],
    tcp_offset_mm: np.ndarray,
) -> tuple[float, float, float]:
    """
    夹爪 TCP 目标点 → 法兰应在的基坐标位置。
    tcp_offset_mm：ee 系下 TCP 相对法兰的平移，满足 tip = flange + R @ offset。
    """
    R = _rotation_matrix_rpy(*rpy)
    tip = np.asarray(tip_xyz, dtype=np.float64)
    flange = tip - R @ tcp_offset_mm
    return (float(flange[0]), float(flange[1]), float(flange[2]))


def flange_xyz_to_tip_xyz(
    flange_xyz: tuple[float, float, float] | np.ndarray,
    rpy: tuple[float, float, float],
    tcp_offset_mm: np.ndarray,
) -> tuple[float, float, float]:
    """法兰位置 → 对应夹爪 TCP 在基坐标下的位置。"""
    R = _rotation_matrix_rpy(*rpy)
    flange = np.asarray(flange_xyz, dtype=np.float64)
    tip = flange + R @ tcp_offset_mm
    return (float(tip[0]), float(tip[1]), float(tip[2]))


def pose_6d_to_matrix(pose_6d: list[float] | tuple[float, ...]) -> np.ndarray:
    """
    法兰位姿 → T_base_ee（4x4）。
    pose: x,y,z (mm), rx,ry,rz (rad)，欧拉角采用 R = Rz @ Ry @ Rx。
    """
    x, y, z, rx, ry, rz = (float(v) for v in pose_6d[:6])
    R = _rotation_matrix_rpy(rx, ry, rz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def transform_point(T: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    """4x4 变换矩阵 × 3D 点 → 3D 点。"""
    p = np.ones(4, dtype=np.float64)
    p[:3] = point_xyz
    out = T @ p
    return out[:3]


def pixel_to_base_mm(
    u: float,
    v: float,
    depth: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
    depth_window: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    完整链路：像素 + 深度图 + 标定 → 基坐标 (x,y,z) mm。
    结果为抓取目标点（作夹爪 TCP 目标；运动规划时再换算法兰位姿）。
    返回 (point_base_mm, debug_info)。
    """
    u_undist, v_undist = undistort_uv(u, v, K, dist)
    ui, vi = int(round(u_undist)), int(round(v_undist))

    depth_mm = sample_depth_mm(
        depth,
        ui,
        vi,
        window=depth_window,
        depth_min_mm=depth_min_mm,
        depth_max_mm=depth_max_mm,
    )
    if depth_mm is None:
        raise CoordTransformError(f"({ui},{vi}) 邻域无有效深度")

    p_cam = pixel_to_camera_mm(u_undist, v_undist, depth_mm, K)
    p_ee = transform_point(T_ee_cam, p_cam)
    p_base = transform_point(T_base_ee, p_ee)

    debug = {
        "uv_raw": (u, v),
        "uv_undist": (u_undist, v_undist),
        "uv_sample": (ui, vi),
        "depth_mm": depth_mm,
        "p_cam_mm": p_cam.tolist(),
        "p_ee_mm": p_ee.tolist(),
        "p_base_mm": p_base.tolist(),
    }
    return p_base, debug


def _rotation_matrix_rpy(rx: float, ry: float, rz: float) -> np.ndarray:
    """R = Rz(rz) @ Ry(ry) @ Rx(rx)。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


rotation_matrix_rpy = _rotation_matrix_rpy
