"""24 槽试管状态表（跨扫描持久）。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from perception.slot_mapper import SlotObservation


class TubeRegistryError(RuntimeError):
    """状态表操作失败。"""


@dataclass
class SlotState:
    slot_id: str
    klass: str = "unknown"  # tube | empty | unknown
    confidence: float = 0.0
    pixel_uv: tuple[float, float] | None = None
    base_xyz: tuple[float, float, float] | None = None  # 抓取目标（夹爪 TCP），基坐标 mm
    z_source: str = "missing"  # measured | rack_plane | missing | unknown
    updated_at: float = 0.0


def estimate_z_rack(
    observations: dict[str, SlotObservation],
    *,
    tube_above_rack_mm: float = 30.0,
    default_z_mm: float | None = None,
) -> float:
    """
    从当前扫描估计 rack 平面高度（mm）。
    优先：tube 的 base_z 中位数 - tube_above_rack_mm；
    其次：default_z_mm（由 scripts/calibrate_rack_height.py 标定写入）。
    """
    tube_zs = [
        o.base_xyz[2]
        for o in observations.values()
        if o.klass == "tube" and o.base_xyz is not None
    ]
    if tube_zs:
        return float(np.median(tube_zs)) - tube_above_rack_mm

    if default_z_mm is not None:
        return default_z_mm

    raise TubeRegistryError("无法估计 z_rack，请提供 depth 图或 --z-rack")


def _resolve_base_xyz(
    obs: SlotObservation,
    z_rack: float,
) -> tuple[tuple[float, float, float] | None, str]:
    if obs.klass == "empty":
        if obs.base_xyz is not None:
            x, y, _ = obs.base_xyz
            return (float(x), float(y), float(z_rack)), "rack_plane"
        return None, "rack_plane"
    if obs.klass == "tube":
        if obs.base_xyz is not None:
            xyz = (float(obs.base_xyz[0]), float(obs.base_xyz[1]), float(obs.base_xyz[2]))
            return xyz, "measured"
        return None, "missing"
    return None, "missing"


class TubeRegistry:
    def __init__(self, all_slot_ids: list[str]) -> None:
        if not all_slot_ids:
            raise TubeRegistryError("all_slot_ids 不能为空")
        self._slot_order = list(all_slot_ids)
        self._slots: dict[str, SlotState] = {
            slot_id: SlotState(slot_id=slot_id) for slot_id in self._slot_order
        }
        self._z_rack: float | None = None

    @property
    def z_rack(self) -> float | None:
        return self._z_rack

    def slot_ids(self) -> list[str]:
        return list(self._slot_order)

    def get(self, slot_id: str) -> SlotState:
        if slot_id not in self._slots:
            raise TubeRegistryError(f"未知 slot_id: {slot_id}")
        return self._slots[slot_id]

    def update_slot(self, slot_id: str, **kwargs) -> None:
        if slot_id not in self._slots:
            raise TubeRegistryError(f"未知 slot_id: {slot_id}")
        state = self._slots[slot_id]
        for key, value in kwargs.items():
            if not hasattr(state, key):
                raise TubeRegistryError(f"无效字段: {key}")
            setattr(state, key, value)
        state.updated_at = time.time()

    def update_from_scan(
        self,
        observations: dict[str, SlotObservation],
        z_rack: float,
    ) -> None:
        """用 SlotMapper 输出刷新 24 槽状态。"""
        self._z_rack = z_rack
        now = time.time()
        for slot_id in self._slot_order:
            obs = observations.get(slot_id)
            if obs is None or obs.klass == "unknown" or obs.pixel_uv is None:
                self._slots[slot_id] = SlotState(slot_id=slot_id, updated_at=now)
                continue

            base_xyz, z_source = _resolve_base_xyz(obs, z_rack)
            self._slots[slot_id] = SlotState(
                slot_id=slot_id,
                klass=obs.klass,
                confidence=obs.confidence,
                pixel_uv=obs.pixel_uv,
                base_xyz=base_xyz,
                z_source=z_source,
                updated_at=now,
            )

    def find_empty_slots(self) -> list[str]:
        return [
            slot_id
            for slot_id in self._slot_order
            if self._slots[slot_id].klass == "empty"
        ]

    def find_tube_slots(self) -> list[str]:
        return [
            slot_id
            for slot_id in self._slot_order
            if self._slots[slot_id].klass == "tube"
        ]

    def to_table_str(self) -> str:
        lines = [
            f"{'slot_id':<12} {'class':<8} {'conf':>6}  {'uv':<18} {'base_xyz':<28} z_src",
            "-" * 90,
        ]
        for slot_id in self._slot_order:
            state = self._slots[slot_id]
            uv = (
                f"({state.pixel_uv[0]:.0f},{state.pixel_uv[1]:.0f})"
                if state.pixel_uv
                else "-"
            )
            if state.base_xyz:
                xyz = (
                    f"({state.base_xyz[0]:.1f},{state.base_xyz[1]:.1f},"
                    f"{state.base_xyz[2]:.1f})"
                )
            else:
                xyz = "-"
            lines.append(
                f"{slot_id:<12} {state.klass:<8} {state.confidence:>6.3f}  "
                f"{uv:<18} {xyz:<28} {state.z_source}"
            )
        if self._z_rack is not None:
            lines.append("")
            lines.append(f"z_rack = {self._z_rack:.1f} mm")
        return "\n".join(lines)
