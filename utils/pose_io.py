"""示教位姿读取工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.config_loader import load_yaml


def pose_dict_to_list(pose: Mapping[str, Any]) -> list[float]:
    """pose dict -> [x, y, z, rx, ry, rz]，单位保持配置中的 mm + rad。"""
    return [
        float(pose["x"]),
        float(pose["y"]),
        float(pose["z"]),
        float(pose["rx"]),
        float(pose["ry"]),
        float(pose["rz"]),
    ]


def load_pose_list(path: str | Path) -> list[float]:
    """读取 config/poses/*.json 风格文件，返回 6D 位姿列表。"""
    return pose_dict_to_list(load_yaml(path)["pose"])
