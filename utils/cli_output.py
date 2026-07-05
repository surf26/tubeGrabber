"""CLI 终端输出格式（main / FSM 共用）。"""

from __future__ import annotations

PROMPT_START = "确认工作空间安全, 按 Enter 开始 (Ctrl+C 取消)"
PROMPT_GRASP = "gripper open, 确认下探安全后按 Enter 夹取 (Ctrl+C 取消)"
AUTO_CONT = " [continuous_mode: auto continue]"


def stage(tag: str, msg: str = "") -> None:
    """阶段日志, 如 [CHECK_HW] connecting hardware"""
    print(f"[{tag}] {msg}" if msg else f"[{tag}]")


def sub(msg: str) -> None:
    print(f"  {msg}")


def ok(msg: str = "done") -> None:
    print(f"[OK] {msg}")


def err(msg: str) -> None:
    print(f"[FAILED] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def move_line(
    label: str,
    target: tuple[float, float, float, float, float, float],
    *,
    cur_xyz: tuple[float, float, float] | None = None,
    delta_mm: float | None = None,
    speed: int | None = None,
) -> None:
    """move_p 目标位姿日志。"""
    x, y, z, rx, ry, rz = target
    parts = [
        f"move_p [{label}]",
        f"target xyz=({x:.1f},{y:.1f},{z:.1f})",
        f"rpy=({rx:.3f},{ry:.3f},{rz:.3f})",
    ]
    if cur_xyz is not None:
        cx, cy, cz = cur_xyz
        parts.append(f"cur xyz=({cx:.1f},{cy:.1f},{cz:.1f})")
    if delta_mm is not None:
        parts.append(f"delta={delta_mm:.1f}mm")
    if speed is not None:
        parts.append(f"speed={speed}")
    sub(" | ".join(parts))


def move_done(
    label: str,
    xyz: tuple[float, float, float],
    moved_mm: float,
) -> None:
    sub(
        f"move_p [{label}] reached "
        f"xyz=({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}), moved={moved_mm:.1f}mm"
    )
