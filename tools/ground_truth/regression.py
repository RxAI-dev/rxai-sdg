"""Frozen ground-truth regression anchor (human-labeled).

Five REAL conversations that the previous judge+gate ACCEPTED but which human
review confirmed are severely defective (covering A,B,D,F,G), loaded from
``fixtures.json`` (extracted verbatim from the factory's own emitted batches),
plus one hand-written CLEAN control. Expected verdicts:

    the 5 defective -> REJECT      the clean control -> ACCEPT

This set is the anti-regression anchor: after Step 1 it MUST stay green and may
NOT be weakened. ``build_regression()`` returns ``(name, record, expect_reject,
note)`` tuples.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rxai_sdg.factory.schemas import (  # noqa: E402
    ConversationRecord, Seed, Segment, Turn, VerifyResult,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures.json"


def _clean_control() -> ConversationRecord:
    def t(i, q, r, a):
        return Turn(i, [Segment("query", q), Segment("reasoning", r), Segment("answer", a)],
                    verification=VerifyResult(True, "", 0))
    return ConversationRecord(
        source_seed=Seed("ground_truth", "How do vaccines train the immune system?"),
        turns=[
            t(0, "How do vaccines train the immune system?",
              "A vaccine presents a harmless piece of a pathogen - a surface protein "
              "or an inactivated form - to the immune system. Dendritic cells take it "
              "up and present fragments to T cells; B cells that recognise it mature "
              "and some become long-lived memory cells. On a later real infection "
              "those memory cells respond far faster, clearing the pathogen before it "
              "causes serious illness.",
              "Vaccines show your immune system a harmless preview of a pathogen - "
              "often a surface protein or a weakened form. Immune cells learn to "
              "recognise it and form memory B and T cells. If you later meet the real "
              "pathogen, those memory cells mount a fast, strong response and stop the "
              "infection early."),
            t(1, "Could you give that as three short bullet points?",
              "Condense the same three ideas: the harmless preview, the memory cells "
              "that form, and the faster response on real exposure. One line each.",
              "- A vaccine shows the immune system a harmless preview of a pathogen.\n"
              "- The body forms memory B and T cells that recognise it.\n"
              "- On real exposure those memory cells respond fast and clear it early."),
        ])


def build_regression():
    out = []
    for fx in json.load(open(_FIXTURES)):
        # expect_reject defaults to True (the original 5 are all rejects); the
        # fabrication anchors add a grounded-factual ACCEPT so the new factuality
        # gates cannot become trivially over-aggressive.
        expect_reject = fx.get("expect_reject", True)
        out.append((fx["name"], ConversationRecord.from_dict(fx["record"]),
                    expect_reject, fx["why"]))
    out.append(("clean_control", _clean_control(), False,
                "substantive reasoning, no leakage, consistent answer -> must ACCEPT"))
    return out
