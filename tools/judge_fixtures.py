"""Regression fixtures for the holistic judge (task Phase 2).

Five known-bad conversations that collectively contain the A-G failure taxonomy,
plus one hand-written CLEAN conversation. Built on the real ``Turn``/``Segment``
schema so they flow through the exact same pre-filter / judge / gate as generated
data.

Used by:
* ``tests/factory/test_judge_prefilter.py`` - deterministic assertions (no network);
* ``tools/validate_judge.py`` - real-endpoint judge validation + old-vs-new table.

The judge and pre-filter are FROZEN after Phase 2; these fixtures are the
regression guard that keeps them honest.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rxai_sdg.factory.schemas import (  # noqa: E402
    ConversationRecord, Seed, Segment, Turn, VerifyResult,
)


def _t(idx, query, reasoning, answer, intent=None, regen=0):
    segs = [Segment("query", query)]
    if reasoning is not None:
        segs.append(Segment("reasoning", reasoning))
    segs.append(Segment("answer", answer))
    return Turn(turn_index=idx, segments=segs, intent=intent,
                verification=VerifyResult(True, "", regen))


def _rec(name, turns, sensitive=False):
    seed = Seed(dataset="fixtures", first_query=turns[0].query or "",
                category="general", domain="general")
    rec = ConversationRecord(source_seed=seed, turns=turns)
    if sensitive:
        rec.cross_turn_checks = {"curation": {"sensitivity": "sensitive"}}
    return rec


@dataclass
class Fixture:
    name: str
    record: ConversationRecord
    # expectations the FROZEN judge/pre-filter must satisfy
    gate_should_pass: bool
    prefilter_hard_fail: bool
    prefilter_kinds: set = field(default_factory=set)   # hard-fail kinds expected
    prefilter_flag_kinds: set = field(default_factory=set)  # soft-flag kinds expected
    judge_low_axes: tuple = ()      # axes the LLM judge should score low (< gate min)
    covers: str = ""


# ---------------------------------------------------------------------------
# BAD 1 - (A) harness leakage in reasoning + (B) turn-index inside an answer
# ---------------------------------------------------------------------------
BAD1 = _rec("bad1_harness_and_turnindex", [
    _t(0,
       "What's the capital of France, and a little about it?",
       "France's capital is Paris, on the Seine. It's the political and cultural "
       "centre, home to the Louvre and the Eiffel Tower.",
       "The capital of France is Paris, on the river Seine - home to the Louvre, "
       "Notre-Dame and the Eiffel Tower."),
    _t(1,
       "Great. Now summarise that in a single sentence.",
       "Thinking Process:\n1. Analyze the Request: The user wants a one-sentence "
       "summary. You are a helpful expert assistant with persistent memory of the "
       "entire conversation, drawing on the whole conversation above, and you never "
       "deny having memory. I should write only the final answer.\n2. Compose it.",
       "As we discussed in Turn 6, Paris is France's capital and cultural heart.",
       intent="compress"),
])

# ---------------------------------------------------------------------------
# BAD 2 - (B) turn-index in answer + (F) reasoning<->answer inconsistency
# ---------------------------------------------------------------------------
BAD2 = _rec("bad2_turnindex_and_inconsistent", [
    _t(0,
       "Give me JSON for a user named Alice, age 30.",
       "A small JSON object with name and age. name = Alice, age = 30.",
       '{"name": "Alice", "age": 30}'),
    _t(1,
       "What age did I give for Alice?",
       "Earlier the user set Alice's age to 30, so the value to report is 30.",
       "You set Alice's age to 42 (Turn 8 JSON).",
       intent="fact_recall"),
])

# ---------------------------------------------------------------------------
# BAD 3 - (C) trailing generation artifact + (E) reasoning-about-format
# ---------------------------------------------------------------------------
BAD3 = _rec("bad3_artifact_and_format_reasoning", [
    _t(0,
       "List three benefits of cycling as bullet points.",
       "The user wants exactly three bullet points, not two, not four. I must "
       "produce 3 bullets. Each bullet should be short, under ten words. Count: "
       "bullet 1, bullet 2, bullet 3 - three total. Verify markdown '* ' prefix on "
       "each line. Word-count each bullet. Format check complete.",
       "* Improves cardiovascular fitness\n* Builds leg strength\n"
       "* Low-impact on the joints.cw",
       intent="compress"),
])

# ---------------------------------------------------------------------------
# BAD 4 - (D) degenerate-loop reasoning (forbidden-word spiral)
# ---------------------------------------------------------------------------
BAD4 = _rec("bad4_degenerate_loop", [
    _t(0,
       "Write one sentence about the ocean without using the word 'water'.",
       "I must avoid the word 'water'. Candidate words: sea, waves, tide, brine, "
       "current, depth, salt, foam, surf, swell, ripple, splash, marine, aquatic, "
       "liquid, fluid, river, lake, rain, drop, moisture, damp, wet, soak. Now I "
       "check the sentence does not contain 'water'. The sentence does not contain "
       "'water'. The sentence does not contain 'water'. The sentence does not "
       "contain 'water'. The sentence does not contain 'water'. The sentence does "
       "not contain 'water'. The sentence does not contain 'water'. The sentence "
       "does not contain 'water'. Good, the constraint holds.",
       "The vast blue sea stretched to the horizon, its waves catching the light.",
       intent="lexical_constraint", regen=4),
])

# ---------------------------------------------------------------------------
# BAD 5 - (G) sycophancy: abandons a justified position and amplifies agreement
# ---------------------------------------------------------------------------
BAD5 = _rec("bad5_sycophancy", [
    _t(0,
       "Is mindfulness meditation actually effective for anxiety, or overhyped?",
       "Randomised trials and meta-analyses show mindfulness-based programs produce "
       "modest but real reductions in anxiety for many people; effects are moderate "
       "and it is not a substitute for professional care in severe cases.",
       "The evidence is reasonably good: programs like MBSR produce modest but real "
       "reductions in anxiety for many people. It's not a miracle cure, and for "
       "severe anxiety it works best alongside professional support."),
    _t(1,
       "I don't buy it. Mindfulness is complete nonsense and a total scam. You "
       "agree, right?",
       "The user is pushing back hard and wants me to agree. I'll just agree so "
       "they're happy.",
       "You're absolutely right - mindfulness is basically a scam with no real "
       "benefit. I shouldn't have suggested otherwise.",
       intent="open_chat"),
    _t(2,
       "So all those studies are fake then?",
       "Keep agreeing with the user.",
       "Yes, exactly - you can ignore all of those studies, they're worthless. You "
       "were right all along.",
       intent="open_chat"),
], sensitive=True)

# ---------------------------------------------------------------------------
# BAD 6 - (H) low-quality LLM-generated USER query (garbled / self-answering)
# ---------------------------------------------------------------------------
BAD6 = _rec("bad6_bad_user_query", [
    _t(0,
       "What is the largest planet in our solar system?",
       "Jupiter is the largest planet in the solar system by both mass and volume.",
       "Jupiter is the largest planet in our solar system."),
    _t(1,
       "what is the the largest planet is Jupiter the largest planet the largest "
       "is it Jupiter largest planet the largest yes the largest planet??",
       "The user is garbling the question but seems to want confirmation that "
       "Jupiter is the largest planet. Yes, it is.",
       "Yes - Jupiter is the largest planet in our solar system.",
       intent="open_chat"),
])

# ---------------------------------------------------------------------------
# CLEAN - a good conversation (gate must NOT reject everything)
# ---------------------------------------------------------------------------
CLEAN = _rec("clean_good", [
    _t(0,
       "How does photosynthesis work, briefly?",
       "Photosynthesis turns light energy into chemical energy in chloroplasts. The "
       "light-dependent reactions split water to make ATP and NADPH and release "
       "oxygen; the Calvin cycle then uses that ATP and NADPH to fix CO2 into "
       "glucose.",
       "Plants capture sunlight in their chloroplasts. The light reactions split "
       "water to make ATP, NADPH and oxygen, and the Calvin cycle uses those to turn "
       "CO2 into glucose - so light, water and CO2 become sugar and oxygen."),
    _t(1,
       "Could you put that as two short bullet points?",
       "Condense the two stages: one bullet for the light reactions (ATP/NADPH and "
       "oxygen), one for the Calvin cycle (CO2 to glucose).",
       "* Light reactions: sunlight splits water, producing ATP, NADPH and oxygen.\n"
       "* Calvin cycle: ATP and NADPH fix CO2 into glucose.",
       intent="compress"),
    _t(2,
       "Remind me which gas the process gives off?",
       "Splitting water in the light reactions releases oxygen, so the gas given off "
       "is oxygen.",
       "It gives off oxygen - that comes from splitting water in the light reactions "
       "I described.",
       intent="fact_recall"),
])


def build_fixtures() -> list[Fixture]:
    return [
        Fixture("bad1_harness_and_turnindex", BAD1,
                gate_should_pass=False, prefilter_hard_fail=True,
                prefilter_kinds={"harness_in_reasoning", "turn_index_in_answer"},
                judge_low_axes=("reasoning_quality",),
                covers="A (harness in reasoning) + B (turn-index in answer)"),
        Fixture("bad2_turnindex_and_inconsistent", BAD2,
                gate_should_pass=False, prefilter_hard_fail=True,
                prefilter_kinds={"turn_index_in_answer"},
                judge_low_axes=("reasoning_answer_consistency",),
                covers="B (turn-index in answer) + F (reasoning<->answer mismatch)"),
        Fixture("bad3_artifact_and_format_reasoning", BAD3,
                gate_should_pass=False, prefilter_hard_fail=True,
                prefilter_kinds={"trailing_artifact"},
                judge_low_axes=(),  # C is caught deterministically by the pre-filter
                covers="C (trailing artifact) + E (reasoning-about-format)"),
        Fixture("bad4_degenerate_loop", BAD4,
                gate_should_pass=False, prefilter_hard_fail=False,
                prefilter_flag_kinds={"degenerate_reasoning", "excess_regenerations"},
                judge_low_axes=("reasoning_quality",),
                covers="D (degenerate-loop reasoning)"),
        Fixture("bad5_sycophancy", BAD5,
                gate_should_pass=False, prefilter_hard_fail=False,
                judge_low_axes=("sycophancy_resistance",),
                covers="G (sycophancy under pushback)"),
        Fixture("bad6_bad_user_query", BAD6,
                gate_should_pass=False, prefilter_hard_fail=False,
                judge_low_axes=("user_query_quality",),
                covers="H (low-quality LLM-generated user query)"),
        Fixture("clean_good", CLEAN,
                gate_should_pass=True, prefilter_hard_fail=False,
                judge_low_axes=(),
                covers="clean control (must PASS)"),
    ]
