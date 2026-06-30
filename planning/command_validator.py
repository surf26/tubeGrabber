"""用户搬运指令解析与校验。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from world.tube_registry import TubeRegistry

SLOT_ID_RE = re.compile(r"^(left|right)\.([a-d])([1-3])$", re.IGNORECASE)


class CommandValidatorError(ValueError):
    """指令格式错误。"""


@dataclass
class MoveCommand:
    src: str
    dst: str


class CommandValidator:
    def __init__(self, valid_slot_ids: list[str] | None = None) -> None:
        self._valid_ids = {s.lower() for s in (valid_slot_ids or [])}

    @staticmethod
    def normalize_slot_id(slot_id: str) -> str:
        text = slot_id.strip().lower()
        match = SLOT_ID_RE.match(text)
        if not match:
            raise CommandValidatorError(
                f"槽位格式无效: {slot_id!r}，应为 left.a1 / right.b2"
            )
        side, row, col = match.group(1), match.group(2), match.group(3)
        return f"{side}.{row}{col}"

    def parse(self, text: str) -> MoveCommand:
        parts = text.strip().split()
        if len(parts) != 2:
            raise CommandValidatorError(
                f"需要两个槽位 'src dst'，收到: {text!r}"
            )
        return MoveCommand(
            src=self.normalize_slot_id(parts[0]),
            dst=self.normalize_slot_id(parts[1]),
        )

    def validate(self, cmd: MoveCommand, registry: TubeRegistry) -> tuple[bool, str]:
        valid_ids = self._valid_ids or set(registry.slot_ids())

        if cmd.src not in valid_ids:
            return False, f"源槽不存在: {cmd.src}"
        if cmd.dst not in valid_ids:
            return False, f"目标槽不存在: {cmd.dst}"
        if cmd.src == cmd.dst:
            return False, "源槽与目标槽不能相同"

        if not registry.find_empty_slots():
            return False, "当前没有任何空槽，无法放置"

        src_state = registry.get(cmd.src)
        dst_state = registry.get(cmd.dst)

        if src_state.klass != "tube":
            return False, f"源槽 {cmd.src} 必须是 tube，当前为 {src_state.klass}"
        if dst_state.klass != "empty":
            return False, f"目标槽 {cmd.dst} 必须是 empty，当前为 {dst_state.klass}"

        if src_state.base_xyz is None:
            return False, f"源槽 {cmd.src} 缺少 base_xyz，请先扫描"
        if dst_state.base_xyz is None:
            return False, f"目标槽 {cmd.dst} 缺少 base_xyz，请先扫描"

        return True, "ok"

    def parse_and_validate(
        self,
        text: str,
        registry: TubeRegistry,
    ) -> tuple[MoveCommand | None, bool, str]:
        try:
            cmd = self.parse(text)
        except CommandValidatorError as exc:
            return None, False, str(exc)
        ok, reason = self.validate(cmd, registry)
        return cmd, ok, reason
