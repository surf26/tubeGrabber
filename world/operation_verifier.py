"""抓取/放置结果验证规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from world.tube_registry import TubeRegistry


class OperationVerifierError(RuntimeError):
    """验证配置或状态异常。"""


@dataclass(frozen=True)
class SlotCheck:
    slot_id: str
    expected_klass: str
    actual_klass: str
    confidence: float
    min_confidence: float
    ok: bool
    reason: str


@dataclass(frozen=True)
class VerificationResult:
    stage: str
    ok: bool
    checks: tuple[SlotCheck, ...]

    def summary(self) -> str:
        if self.ok:
            detail = ", ".join(
                f"{c.slot_id}={c.actual_klass}(conf={c.confidence:.2f})"
                for c in self.checks
            )
            tag = "VERIFY_PICK" if self.stage == "pick" else "VERIFY_PLACE"
            return f"{tag} OK: {detail}"
        return "; ".join(c.reason for c in self.checks if not c.ok)


class OperationVerifier:
    """
    根据配置验证一次搬运阶段是否成功。

    默认规则：
    - pick:  源槽 src 应为 empty
    - place: 目标槽 dst 应为 tube
    """

    _DEFAULTS = {
        "pick": {
            "rules": [
                {"slot": "src", "expected": "empty", "min_confidence": 0.0},
            ]
        },
        "place": {
            "rules": [
                {"slot": "dst", "expected": "tube", "min_confidence": 0.0},
            ]
        },
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def verify(
        self,
        stage: str,
        *,
        src: str,
        dst: str,
        registry: TubeRegistry,
    ) -> VerificationResult:
        stage_key = stage.lower()
        rules = self._rules_for(stage_key)
        checks: list[SlotCheck] = []

        for rule in rules:
            slot_id = _resolve_slot_ref(str(rule.get("slot", "")), src=src, dst=dst)
            expected = str(rule.get("expected", "")).lower()
            if expected not in ("tube", "empty", "unknown"):
                raise OperationVerifierError(
                    f"verification.{stage_key} expected 无效: {expected!r}"
                )
            min_conf = float(rule.get("min_confidence", 0.0))

            state = registry.get(slot_id)
            klass_ok = state.klass == expected
            conf_ok = state.confidence >= min_conf
            ok = klass_ok and conf_ok
            if ok:
                reason = "ok"
            elif not klass_ok:
                reason = (
                    f"VERIFY_{stage_key.upper()} failed: {slot_id} "
                    f"expected={expected}, actual={state.klass}"
                )
            else:
                reason = (
                    f"VERIFY_{stage_key.upper()} failed: {slot_id} "
                    f"conf={state.confidence:.2f} < min_confidence={min_conf:.2f}"
                )

            checks.append(
                SlotCheck(
                    slot_id=slot_id,
                    expected_klass=expected,
                    actual_klass=state.klass,
                    confidence=float(state.confidence),
                    min_confidence=min_conf,
                    ok=ok,
                    reason=reason,
                )
            )

        return VerificationResult(
            stage=stage_key,
            ok=all(c.ok for c in checks),
            checks=tuple(checks),
        )

    def _rules_for(self, stage: str) -> list[dict[str, Any]]:
        stage_cfg = self._config.get(stage)
        if stage_cfg is None:
            stage_cfg = self._DEFAULTS.get(stage)
        if not isinstance(stage_cfg, dict):
            raise OperationVerifierError(f"verification.{stage} 必须是 dict")

        rules = stage_cfg.get("rules")
        if not isinstance(rules, list) or not rules:
            raise OperationVerifierError(f"verification.{stage}.rules 不能为空")
        return rules


def _resolve_slot_ref(value: str, *, src: str, dst: str) -> str:
    key = value.strip().lower()
    if key == "src":
        return src
    if key == "dst":
        return dst
    if "." in key:
        return key
    raise OperationVerifierError(f"无效 slot 引用: {value!r}")
