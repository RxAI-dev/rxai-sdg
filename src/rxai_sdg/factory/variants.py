"""Reasoning -> derived-variant post-processing (spec §8) - **standalone**.

Generation always happens in reasoning mode and the Data Factory emits exactly
**one** reasoning-mode record per conversation. Deriving the instruct / mixed
training variants is a separate, deterministic post-processing step that can be
run independently *later* over an already-generated dataset - it is intentionally
NOT part of the live generation pipeline.

Usage::

    from rxai_sdg.factory.variants import derive_variants
    instruct, mixed = derive_variants(record, ["instruct", "mixed"])

* ``reasoning`` - all reasoning segments kept;
* ``instruct``  - all reasoning segments removed;
* ``mixed``     - a sampled subset of turns keep reasoning, the rest stripped.

The **self-containment rule** is enforced: after stripping reasoning, answers
that still reference removed reasoning ("as computed above", "from step 2") are
flagged under ``record.cross_turn_checks["self_containment"]``.
"""

from __future__ import annotations

import copy
import random
from typing import Iterable, Optional

from .schemas import ConversationRecord, Turn
from .writer import flag_dangling_references


def derive_variants(
    record: ConversationRecord,
    variants: Iterable[str] = ("reasoning", "instruct", "mixed"),
    rng: Optional[random.Random] = None,
    mixed_mode_keep_ratio: float = 0.5,
) -> list[ConversationRecord]:
    """Produce the requested derived records from one reasoning-mode record."""
    rng = rng or random.Random()
    out: list[ConversationRecord] = []
    for variant in variants:
        if variant == "reasoning":
            out.append(_clone(record, "reasoning"))
        elif variant == "instruct":
            new = _clone(record, "instruct")
            for turn in new.turns:
                _strip_reasoning(turn)
            _annotate_self_containment(new)
            out.append(new)
        elif variant == "mixed":
            new = _clone(record, "mixed")
            for turn in new.turns:
                if rng.random() >= mixed_mode_keep_ratio:
                    _strip_reasoning(turn)
            _annotate_self_containment(new)
            out.append(new)
        else:
            raise ValueError(f"unknown derived variant: {variant!r}")
    return out


def _clone(record: ConversationRecord, mode: str) -> ConversationRecord:
    new = copy.deepcopy(record)
    new.mode = mode  # type: ignore[assignment]
    new.conversation_id = f"{record.conversation_id}:{mode}"
    return new


def _strip_reasoning(turn: Turn) -> None:
    turn.segments = [s for s in turn.segments if s.segment_type != "reasoning"]
    turn.reasoning_flag = False


def _annotate_self_containment(record: ConversationRecord) -> None:
    flags = []
    for turn in record.turns:
        if turn.reasoning_flag:
            continue  # reasoning kept; nothing stripped from this turn
        dangling = flag_dangling_references(turn.answer or "")
        if dangling:
            flags.append({"turn_index": turn.turn_index, "phrases": dangling})
    if flags:
        record.cross_turn_checks = dict(record.cross_turn_checks)
        record.cross_turn_checks["self_containment"] = flags
