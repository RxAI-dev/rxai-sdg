"""Segment-structured writer (spec §5.6).

The :class:`SegmentWriter` serialises :class:`ConversationRecord` objects (each
turn already carries typed ``query`` / ``reasoning`` / ``answer`` segments) to
the output schema (§7) and to JSONL.

The Data Factory emits exactly **one** reasoning-mode record per conversation.
Deriving instruct / mixed training variants is a separate post-processing step
(see :mod:`rxai_sdg.factory.variants`), not part of this writer. The
self-containment helper :func:`flag_dangling_references` lives here because it is
a generic answer-quality check reused by that post-processing step.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from .schemas import ConversationRecord

# Phrases that indicate an answer leans on reasoning that may have been stripped.
_DANGLING_PATTERNS = [
    r"\bas (?:computed|shown|derived|calculated|mentioned|noted|discussed) above\b",
    r"\bfrom step\s+\d+\b",
    r"\bin step\s+\d+\b",
    r"\bas (?:we|i) (?:computed|saw|found|noted) (?:above|earlier)\b",
    r"\busing the (?:above|previous) (?:reasoning|calculation|steps?)\b",
    r"\bper my reasoning\b",
    r"\bas reasoned\b",
]
_DANGLING_RE = re.compile("|".join(_DANGLING_PATTERNS), re.IGNORECASE)


def flag_dangling_references(answer: str) -> list[str]:
    """Return the list of dangling-reference phrases found in ``answer``."""
    return [m.group(0) for m in _DANGLING_RE.finditer(answer or "")]


@dataclass
class SegmentWriter:
    def to_dict(self, record: ConversationRecord) -> dict:
        return record.to_dict()

    def write_jsonl(self, records: Iterable[ConversationRecord], path: str) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                n += 1
        return n
