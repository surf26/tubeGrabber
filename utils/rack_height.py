"""试管架平面高度：交互点击 + 深度 → base Z，写入 config。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.opencv_gui import cv2

from perception.coord_transform import CoordTransformError, pixel_to_base_mm
from utils.config_loader import PROJECT_ROOT, load_yaml


class RackHeightError(RuntimeError):
    """试管架高度标定失败。"""


@dataclass
class RackClickSample:
    uv: tuple[int, int]
    depth_mm: float
    base_xyz: tuple[float, float, float]


def z_median_from_clicks(
    clicks: list[tuple[int, int]],
    depth: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
) -> tuple[float, list[RackClickSample]]:
    """多点点击 → 基坐标 Z 中位数。"""
    if not clicks:
        raise RackHeightError("至少点击 1 个空孔位")

    samples: list[RackClickSample] = []
    errors: list[str] = []
    for u, v in clicks:
        try:
            p_base, dbg = pixel_to_base_mm(
                float(u),
                float(v),
                depth,
                K,
                dist,
                T_ee_cam,
                T_base_ee,
                depth_min_mm=depth_min_mm,
                depth_max_mm=depth_max_mm,
            )
        except CoordTransformError as exc:
            errors.append(f"({u},{v}): {exc}")
            continue

        xyz = (float(p_base[0]), float(p_base[1]), float(p_base[2]))
        samples.append(
            RackClickSample(
                uv=(int(u), int(v)),
                depth_mm=float(dbg["depth_mm"]),
                base_xyz=xyz,
            )
        )

    if not samples:
        detail = "; ".join(errors) if errors else "无有效点"
        raise RackHeightError(f"所有点击点深度/transform 均失败: {detail}")

    z_rack = float(np.median([s.base_xyz[2] for s in samples]))
    return z_rack, samples


def load_rack_plane_z_mm(path: str | Path = "config/rack_layout.yaml") -> float | None:
    data = load_yaml(path)
    value = data.get("default_rack_plane_z_mm")
    if value is None:
        return None
    return float(value)


def save_rack_plane_z_mm(
    z_mm: float,
    path: str | Path = "config/rack_layout.yaml",
) -> Path:
    """写入 rack_layout.yaml 的 default_rack_plane_z_mm，保留其余内容与注释。"""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.is_file():
        raise RackHeightError(f"找不到配置文件: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    new_line = f"default_rack_plane_z_mm: {z_mm:.1f}"
    if re.search(r"^default_rack_plane_z_mm:", text, flags=re.MULTILINE):
        text = re.sub(
            r"^default_rack_plane_z_mm:.*$",
            new_line,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = text.rstrip() + "\n" + new_line + "\n"

    file_path.write_text(text, encoding="utf-8")
    return file_path


def pick_rack_plane_z_interactive(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_ee_cam: np.ndarray,
    T_base_ee: np.ndarray,
    *,
    depth_min_mm: float = 100,
    depth_max_mm: float = 800,
    min_clicks: int = 3,
    window_name: str = "标定试管架高度",
) -> tuple[float, list[RackClickSample]] | None:
    """
    弹出 imshow 窗口，左键点击空孔位，Enter 确认，r 清空，q 取消。

    返回 (z_rack_mm, samples)；用户取消则返回 None。
    """
    clicks: list[tuple[int, int]] = []
    canvas = color_bgr.copy()

    help_lines = [
        "左键: 点击空孔位(可多次)",
        f"Enter: 确认(至少{min_clicks}点)",
        "r: 清空  q: 退出",
    ]

    def _redraw() -> None:
        nonlocal canvas
        canvas = color_bgr.copy()
        for i, (u, v) in enumerate(clicks, start=1):
            cv2.circle(canvas, (u, v), 6, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                str(i),
                (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        y0 = 24
        for line in help_lines:
            cv2.putText(
                canvas,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            y0 += 22
        cv2.putText(
            canvas,
            f"已选 {len(clicks)} 点",
            (10, y0 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        _redraw()

    _redraw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, canvas)
    # 须先 imshow 创建窗口，再注册鼠标回调（否则 Linux 上 null window handler）
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, _on_mouse)

    result: tuple[float, list[RackClickSample]] | None = None
    while True:
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if len(clicks) < min_clicks:
                print(f"请至少点击 {min_clicks} 个空孔位（当前 {len(clicks)}）")
                continue
            try:
                z_rack, samples = z_median_from_clicks(
                    clicks,
                    depth,
                    K,
                    dist,
                    T_ee_cam,
                    T_base_ee,
                    depth_min_mm=depth_min_mm,
                    depth_max_mm=depth_max_mm,
                )
            except RackHeightError as exc:
                print(exc)
                continue

            print(f"\n有效点 {len(samples)}/{len(clicks)}:")
            for i, s in enumerate(samples, start=1):
                x, y, z = s.base_xyz
                print(
                    f"  #{i} uv=({s.uv[0]},{s.uv[1]}) "
                    f"depth={s.depth_mm:.1f}mm base=({x:.1f},{y:.1f},{z:.1f})"
                )
            print(f"\nz_rack (median Z) = {z_rack:.1f} mm")
            result = (z_rack, samples)
            break

        if key == ord("r"):
            clicks.clear()
            _redraw()
            print("已清空点击")

        if key == ord("q") or key == 27:
            print("已取消")
            break

    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
    return result
