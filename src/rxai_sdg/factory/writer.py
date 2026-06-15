"""Segment-structured writer + reasoning post-processing (spec §5.6, §8).

The :class:`SegmentWriter` serialises :class:`ConversationRecord` objects to the
output schema (§7) and derives the multiple training records from a single
reasoning-mode generation:

* ``reasoning`` - all reasoning segments kept;
* ``instruct``  - all reasoning segments removed;
* ``mixed``     - a sampled subset of turns keep reasoning, the rest stripped.

It also enforces the **self-containment rule**: after stripping reasoning, the
answer must stand alone. A light regex pass flags answers with dangling
references to removed reasoning ("as computed above", "from step 2", ...).
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .schemas import ConversationRecord, Segment, Turn

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
    rng: Optional[random.Random] = None
    mixed_mode_keep_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    # ------------------------------------------------------- variant derivation
    def derive_variants(
        self,
        record: ConversationRecord,
        variants: Iterable[str],
    ) -> list[ConversationRecord]:
        """Produce the requested derived records from one reasoning-mode record."""
        out: list[ConversationRecord] = []
        for variant in variants:
            if variant == "reasoning":
                out.append(self._as_reasoning(record))
            elif variant == "instruct":
                out.append(self._as_instruct(record))
            elif variant == "mixed":
                out.append(self._as_mixed(record))
            else:
                raise ValueError(f"unknown derived variant: {variant!r}")
        return out

    def _clone(self, record: ConversationRecord, mode: str) -> ConversationRecord:
        new = copy.deepcopy(record)
        new.mode = mode  # type: ignore[assignment]
        # Distinct conversation_id per derived variant keeps records independent.
        new.conversation_id = f"{record.conversation_id}:{mode}"
        return new

    def _as_reasoning(self, record: ConversationRecord) -> ConversationRecord:
        return self._clone(record, "reasoning")

    def _as_instruct(self, record: ConversationRecord) -> ConversationRecord:
        new = self._clone(record, "instruct")
        for turn in new.turns:
            self._strip_reasoning(turn)
        self._annotate_self_containment(new)
        return new

    def _as_mixed(self, record: ConversationRecord) -> ConversationRecord:
        new = self._clone(record, "mixed")
        for turn in new.turns:
            keep = self.rng.random() < self.mixed_mode_keep_ratio
            if not keep:
                self._strip_reasoning(turn)
        self._annotate_self_containment(new)
        return new

    @staticmethod
    def _strip_reasoning(turn: Turn) -> None:
        turn.segments = [s for s in turn.segments if s.segment_type != "reasoning"]
        turn.reasoning_flag = False

    def _annotate_self_containment(self, record: ConversationRecord) -> None:
        flags = []
        for turn in record.turns:
            if turn.reasoning_flag:
                continue  # reasoning kept; nothing was stripped
            dangling = flag_dangling_references(turn.answer or "")
            if dangling:
                flags.append({"turn_index": turn.turn_index, "phrases": dangling})
        if flags:
            record.cross_turn_checks = dict(record.cross_turn_checks)
            record.cross_turn_checks["self_containment"] = flags

    # --------------------------------------------------------------- emit / io
    def to_dict(self, record: ConversationRecord) -> dict:
        return record.to_dict()

    def write_jsonl(self, records: Iterable[ConversationRecord], path: str) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                n += 1
        return n
