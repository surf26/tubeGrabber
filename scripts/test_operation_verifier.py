"""OperationVerifier 离线自检。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world.operation_verifier import OperationVerifier
from world.tube_registry import TubeRegistry


def main() -> int:
    registry = TubeRegistry(["left.a1", "right.b2"])
    registry.update_slot("left.a1", klass="empty", confidence=0.91)
    registry.update_slot("right.b2", klass="tube", confidence=0.88)

    verifier = OperationVerifier()
    pick = verifier.verify("pick", src="left.a1", dst="right.b2", registry=registry)
    place = verifier.verify("place", src="left.a1", dst="right.b2", registry=registry)
    print(pick.summary())
    print(place.summary())

    if not pick.ok or not place.ok:
        return 1

    strict = OperationVerifier(
        {
            "place": {
                "rules": [
                    {"slot": "dst", "expected": "tube", "min_confidence": 0.95}
                ]
            }
        }
    )
    strict_result = strict.verify(
        "place", src="left.a1", dst="right.b2", registry=registry
    )
    print(strict_result.summary())
    return 0 if not strict_result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
