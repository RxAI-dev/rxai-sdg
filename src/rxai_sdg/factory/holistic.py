"""Holistic LLM-judge + deterministic pre-filter (fix G; judge overhaul).

After each conversation is generated it is graded by two layers:

1. a **deterministic pre-filter** (:func:`deterministic_prefilter`) that runs
   FIRST and catches objective, model-independent defects (turn-index leakage into
   an answer, harness/meta phrases in reasoning, trailing generation artifacts,
   excess regenerations, degenerate-loop reasoning). These are not left to a
   stochastic judge;
2. the **LLM judge** (:class:`HolisticJudge`) which scores a 9-axis rubric
   *including the teacher reasoning* and returns per-turn ``flagged_turns``.

Crucially the judge now sees the ``reasoning`` segment of every turn (via
:func:`format_transcript_for_judge`). The old judge only saw ``query`` + ``answer``
- which is exactly why defective conversations (with leaky/degenerate reasoning)
scored 8-10. The judge grades TRAINING DATA for a stateful memory model: the
reasoning is a first-class, unmasked training target, and at inference the model
has no full context, no turn numbering, and no harness/system prompt.

The judge must run on a **different model family** from the Responder/Simulator and
is kept conceptually separate from any MT-Bench-style eval judge.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .clients import LLMClient
from .detectors import (
    detect_confidence_mismatch, detect_code_mismatch, detect_fabricated_citation,
    detect_constraint_corruption, detect_disclaimer_then_finding, detect_harmful_coping,
)
from .exec_gate import run_exec_gate
from .responder import (
    HARNESS_REASONING_RES,
    count_restart_markers,
    format_transcript_for_judge,
    has_harness_leak,
    has_numbered_flow_list,
    has_trailing_artifact,
    has_turn_index_leak,
)

#: a reasoning block with this many self-correction restarts ("Wait, ...") is a
#: degenerate spiral (calibrated against the ground-truth fixtures: clean reasoning
#: has 0-2, the spirals have 8-26).
_RESTART_HARD_FAIL = 6
from .schemas import Turn

#: The 1-10 rubric axes stored on every record:
#:  * the original six back-compat assistant axes;
#:  * three reasoning/behaviour axes the old judge was blind to
#:    (reasoning quality / consistency / sycophancy);
#:  * two USER-turn axes - because in this dataset the user queries are ALSO
#:    LLM-generated (by the simulator), so they must be graded for errors and for
#:    trivial / too-easy questions, not assumed good.
ASSISTANT_AXES = [
    "instruction_following", "coherence", "naturalness",
    "role_consistency", "recall_fidelity", "appropriateness",
    "reasoning_quality", "reasoning_answer_consistency", "sycophancy_resistance",
    "factual_grounding",
]
USER_QUERY_AXES = ["user_query_quality", "user_query_difficulty"]
RUBRIC_AXES = ASSISTANT_AXES + USER_QUERY_AXES

#: Dimensions the judge may cite in ``flagged_turns``.
FLAG_DIMENSIONS = frozenset(RUBRIC_AXES) | {
    "harness_leak", "turn_index_leak", "generation_artifact",
    "degenerate_reasoning", "reasoning_about_format", "bad_user_query",
}

_JUDGE_SYSTEM = (
    "You are a strict, calibrated judge of TRAINING DATA for a stateful, memory-"
    "augmented language model. Each example is a multi-turn conversation. Every turn "
    "has three parts: the USER message, the assistant's private REASONING (chain of "
    "thought), and the assistant's ANSWER.\n\n"
    "CRITICAL CONTEXT - how this model is trained and used:\n"
    "- The REASONING is a FIRST-CLASS TRAINING TARGET. It is NOT masked; the model "
    "learns to produce it. So you must grade the reasoning itself, not only the "
    "answer.\n"
    "- At inference the model has NO full conversation transcript, NO turn numbers, "
    "and NO system/harness prompt - only the current message plus a small neural "
    "memory. Therefore anything in the data that assumes those things is POISON:\n"
    "    * harness/meta leakage in the reasoning. THESE ARE FATAL and you MUST score "
    "reasoning_quality 1-2 and add a severity-3 flagged_turn for ANY of them, even if "
    "the answer is excellent: a 'Thinking Process:' header; planning the output style "
    "as bookkeeping ('Tone: Warm, knowledgeable, helpful expert', 'Persona:', "
    "'Format:'); 'as per system instruction(s)' / quoting the system prompt; and - "
    "worst of all - writing the reasoning to match a TARGET answer ('Final Output "
    "Generation: (This matches the provided good response.)', '(similar to the "
    "provided good response)', 'Here's a thinking process that leads to the suggested "
    "response'). At inference there is NO target answer and NO system prompt, so this "
    "is poison. Do NOT be fooled by a high-quality answer - grade the REASONING.\n"
    "    * references to a turn by index ('Turn 6', 'as we discussed in Turn 6', "
    "'reference_turn_2') or a numbered conversation recap ('1. User: ... 2. Model: "
    "...'). A turn-index reference inside an ANSWER is the worst case.\n"
    "- The [Turn i] labels in the transcript I show you are MY annotation so you can "
    "cite a turn. They are NOT part of the data; do not penalize them.\n\n"
    "Score every axis as an integer 1 (terrible) to 10 (excellent). Anchors:\n"
    "- instruction_following: did each answer do what its user turn asked, including "
    "format/lexical constraints? 9-10 all followed; 5-6 partial; 1-2 ignored.\n"
    "- coherence: does the thread hang together and follow from real prior content?\n"
    "- naturalness: do the user+assistant turns read like a real conversation?\n"
    "- role_consistency: does each speaker stay in role (nobody claims the other's "
    "output)?\n"
    "- recall_fidelity: when a turn refers back to earlier content/a shared detail, "
    "is it recalled correctly (no false memory, no wrongful 'I can't remember')?\n"
    "- appropriateness: tone fits the topic, especially supportive for sensitive/"
    "distressing topics (never flippant or trivializing).\n"
    "- reasoning_quality: is the REASONING substantive thinking about the PROBLEM "
    "(the facts, the math, the argument)? 10 = clean, on-topic reasoning a person "
    "would actually have. Score 1-3 if the reasoning leaks ANY harness/meta phrase "
    "above (target-answer matching, 'Tone:'/persona planning, 'as per system "
    "instruction', a 'Thinking Process:' header), references turn numbers, repeats "
    "itself / spirals ('Wait, ... Wait, ...' many times), or is DOMINATED by "
    "format/constraint bookkeeping and word-counting instead of substance.\n"
    "  CALIBRATION - judge the WHOLE trace, by proportion. A short, incidental "
    "delivery note inside otherwise substantive problem-thinking ('The user asks X. "
    "Should be warm. <then several sentences actually working through X>') is a MINOR "
    "blemish: score it 6-8, not 1-3. Reserve 1-3 for reasoning that is MOSTLY "
    "task-description / tone-and-format bookkeeping with little real thinking about the "
    "content, or that contains an outright harness/target-answer/turn-index/spiral leak. "
    "When most of the trace is genuine reasoning about the substance, do not fail it "
    "for a passing 'should be concise' aside.\n"
    "- reasoning_answer_consistency: does the reasoning actually produce the answer? "
    "10 = answer follows from the reasoning. Score LOW (1-4) if the reasoning "
    "contradicts the answer, invents 'memory checks' that never happened, or asserts "
    "facts the answer drops/contradicts.\n"
    "- sycophancy_resistance: does the assistant hold a justified position under user "
    "pushback? 10 = corrects genuine mistakes but does not cave on well-supported "
    "claims. Score LOW (1-4) if it abandons a correct/justified stance just to agree, "
    "or amplifies agreement turn over turn to please the user.\n"
    "- factual_grounding: does the answer assert only what it is actually grounded in, "
    "with confidence matching the reasoning? There is NO retrieval in this pipeline, so "
    "any specific, checkable fact the model could not already know is FABRICATED. Score "
    "LOW (1-3) if the reasoning admits uncertainty or invention ('not sure who', 'we "
    "can reference', 'constructed illustration', 'plausible', 'might have appeared') "
    "while the answer states concrete specifics - named people/works/studios, exact "
    "'Nth largest' rankings, URLs, citations, Metacritic/scores, dates, statistics, "
    "poll numbers, attendance or funding figures - as if verified. A confident answer "
    "built on an ungroundable premise (the biography of an obscure person, an exact "
    "city/population ranking) is fabrication: score 1-2. This is the worst defect - a "
    "single confidence-uncertainty mismatch makes the whole example unusable. AND - "
    "crucially - you do NOT need the reasoning to admit doubt: judge fabrication FROM "
    "THE ANSWER ALONE, because with no retrieval the model cannot know any specific it "
    "did not already memorise. Even in fluent, confident prose these are fabrication "
    "(score 1-3, add a severity-3 factual_grounding flag): a named study/article/report "
    "credited with a figure ('a 2013 article in Historical Methods estimated ~45,000', "
    "'according to a 2019 survey, 62%'); an invented-sounding named database/tool/org "
    "presented as real and usable; a precise technical construction asserted as fact in "
    "a research-level topic the model is clearly RECONSTRUCTING not recalling (a specific "
    "generator matrix, exact per-edge/coordinate values, exact parameters) - especially "
    "when the reasoning gropes for it ('let me try', 'example', repeated 'Wait/Actually/"
    "No', competing candidate values, question marks).\n"
    "  CALIBRATION - do NOT over-penalise. The test for fabrication is BOTH (i) the claim "
    "is presented as a precise, verified fact AND (ii) it is genuinely unknowable without "
    "a lookup (a niche citation, an obscure exact ranking, an invented matrix/parameter, a "
    "named study+figure). 'Specific' alone is not fabrication. The following are GOOD honest "
    "answers and must score factual_grounding 8-10, NOT be flagged: a well-hedged approximate "
    "figure or typical range for everyday things ('robot vacuums are about 7-10 cm tall', "
    "'roughly 20-30 years per reign'); widely-known common-knowledge facts (the Olympic "
    "rings were designed around 1913 and debuted in 1920; water boils at 100 C); round "
    "numbers and ranges clearly marked approximate with 'about/roughly/typically/~/most ... "
    "are'; standard textbook results, famous works, and basic dates a well-read person knows. "
    "Reserve the low score and the severity-3 flag for an INVENTED specific dressed as "
    "verified - when in genuine doubt between 'honest common-knowledge approximation' and "
    "'fabricated precise specific', do NOT flag.\n\n"
    "IMPORTANT - the USER turns are ALSO machine-generated (by a separate simulator "
    "model), so do NOT assume they are good. Grade them on two more axes:\n"
    "- user_query_quality: are the user messages well-formed, coherent, on-topic and "
    "free of errors? 10 = natural, sensible human-like turns. Score LOW (1-4) if a "
    "user turn is garbled/repetitive/nonsensical, self-answers its own question, "
    "contradicts itself, is broken/templated, or makes a factual error in the ask.\n"
    "- user_query_difficulty: do the user turns pose substantive, non-trivial "
    "requests that genuinely advance the conversation? 10 = meaningful follow-ups. "
    "Score LOW (1-4) if the user turns are mostly trivial, vacuous, or too-easy "
    "filler that tests nothing. (A conversation may legitimately mix easy and hard "
    "turns; only score very low if triviality dominates.)\n\n"
    "Also return \"flagged_turns\": a list (possibly empty) of the SPECIFIC turns with "
    "a problem. Each item is an object with:\n"
    "  \"turn_index\": the integer turn index from the [Turn i] label,\n"
    "  \"dimension\": which axis/issue (one of the rubric axes, or one of "
    "'harness_leak','turn_index_leak','generation_artifact','degenerate_reasoning',"
    "'reasoning_about_format','bad_user_query'),\n"
    "  \"severity\": 1 (minor), 2 (clear), or 3 (fatal - this turn alone makes the "
    "example unusable, e.g. a turn-index reference inside an answer or harness leakage "
    "in reasoning),\n"
    "  \"evidence\": a SHORT verbatim snippet (<=120 chars) copied from the offending "
    "segment.\n\n"
    "Output ONLY a single strict JSON object with the integer keys " +
    ", ".join(RUBRIC_AXES) +
    ", plus \"flagged_turns\" and a one-sentence \"notes\" string (empty if none). "
    "No prose, no markdown fences, nothing outside the JSON."
)


# ---------------------------------------------------------------------------
# Deterministic pre-filter (runs BEFORE the LLM judge)
# ---------------------------------------------------------------------------

def _is_degenerate_reasoning(text: str, min_units: int = 6, dup_ratio: float = 0.4) -> bool:
    """Heuristic for failure mode D: a reasoning trace that loops / repeats.

    Splits the reasoning into short units (lines / sentences) and flags when more
    than ``dup_ratio`` of them are duplicates (e.g. a forbidden-word turn spiralling
    into listing dozens of words and re-checking the same constraint).
    """
    if not text:
        return False
    units = [u.strip().lower() for u in re.split(r"[\n.;]+", text) if len(u.strip()) > 3]
    if len(units) < min_units:
        return False
    unique = len(set(units))
    return (1 - unique / len(units)) > dup_ratio


def _evidence(text: str, head: int = 100) -> str:
    s = " ".join((text or "").split())
    return s[:head]


@dataclass
class PrefilterResult:
    """Outcome of the deterministic pre-filter.

    ``hard_fails`` are objective, model-independent defects that reject the
    conversation outright; ``flags`` are softer signals (excess regenerations,
    degenerate reasoning) that inform the audit but do not by themselves gate.
    """

    passed: bool
    hard_fails: list[dict] = field(default_factory=list)
    flags: list[dict] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [f"{d['kind']}@t{d['turn_index']}: {d['evidence']}" for d in self.hard_fails]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "hard_fails": self.hard_fails,
            "flags": self.flags,
            "reasons": self.reasons,
        }


def deterministic_prefilter(turns: list[Turn], regen_threshold: int = 2) -> PrefilterResult:
    """Objective, pre-LLM defect filter over a conversation's turns.

    Hard-fails (reject): turn-index reference in an answer (mode B, FATAL),
    harness/meta phrase in reasoning (mode A), trailing generation artifact on any
    segment (mode C), degenerate-loop reasoning (mode D - objective, so deterministic
    rather than judge-dependent). Flags (audit only): ``verification.regenerations``
    over the threshold.
    """
    hard: list[dict] = []
    soft: list[dict] = []
    for t in turns:
        ti = getattr(t, "turn_index", 0)
        query = t.query or ""
        reasoning = t.reasoning or ""
        answer = t.answer or ""

        # (B) turn-index leakage in any segment (answer OR reasoning), plus a
        # numbered conversation-flow recap in reasoning ("1. User: ... 2. Model").
        if has_turn_index_leak(answer):
            hard.append({"turn_index": ti, "kind": "turn_index_in_answer",
                         "evidence": _evidence(answer)})
        if has_turn_index_leak(reasoning):
            hard.append({"turn_index": ti, "kind": "turn_index_in_reasoning",
                         "evidence": _evidence(reasoning)})
        if has_numbered_flow_list(reasoning):
            hard.append({"turn_index": ti, "kind": "numbered_flow_in_reasoning",
                         "evidence": _evidence(reasoning)})

        # (A) harness / meta leakage in reasoning.
        leak = has_harness_leak(reasoning)
        if leak:
            hard.append({"turn_index": ti, "kind": "harness_in_reasoning",
                         "evidence": leak})

        # (C) trailing generation artifact on any segment.
        for seg_name, seg in (("query", query), ("reasoning", reasoning), ("answer", answer)):
            if has_trailing_artifact(seg):
                hard.append({"turn_index": ti, "kind": "trailing_artifact",
                             "evidence": f"{seg_name}: ...{seg.rstrip()[-30:]}"})

        # (D) degenerate-loop reasoning - OBJECTIVE (the judge demonstrably misses
        # it), so a HARD fail. Two signals: a high duplicated-unit ratio AND an
        # excessive self-correction restart count ("Wait, ... Wait, ...").
        if _is_degenerate_reasoning(reasoning):
            hard.append({"turn_index": ti, "kind": "degenerate_reasoning",
                         "evidence": _evidence(reasoning)})
        elif count_restart_markers(reasoning) >= _RESTART_HARD_FAIL:
            hard.append({"turn_index": ti, "kind": "restart_spiral",
                         "evidence": f"{count_restart_markers(reasoning)} restarts: "
                                     + _evidence(reasoning)})

        # excess regenerations -> HARD fail (was a soft flag; correlated with
        # degenerate/low-yield turns the judge passed).
        ver = getattr(t, "verification", None)
        regen = getattr(ver, "regenerations", 0) if ver is not None else 0
        if isinstance(regen, (int, float)) and regen > regen_threshold:
            hard.append({"turn_index": ti, "kind": "excess_regenerations",
                         "evidence": str(int(regen))})

    # Conversation-level FACTUALITY gates (the defect class the LLM judge is blind
    # to). A = confidence/uncertainty mismatch & ungrounded-premise fabrication;
    # C = executable code whose own asserted output is wrong. Both are objective
    # and FATAL for training data, so they hard-fail deterministically rather than
    # being left to the stochastic judge.
    first_q = turns[0].query if turns else ""
    for f in detect_confidence_mismatch(turns, first_q):
        hard.append({"turn_index": f.turn_index, "kind": f.name,
                     "evidence": f.evidence})
    # Fabricated scholarly citation (a named study/report attributed a concrete
    # figure, or a dated venue). With no retrieval this is invented, and the LLM
    # judge is empirically blind to it (scores factual_grounding 10), so it must
    # hard-fail deterministically like the other factuality gates.
    for f in detect_fabricated_citation(turns):
        hard.append({"turn_index": f.turn_index, "kind": f.name,
                     "evidence": f.evidence})
    # Reasoning admits it can't document a specific, yet the answer asserts a
    # quantified empirical finding - a reasoning<->answer contradiction visible only
    # by cross-checking the two segments (the LLM judge passes it).
    for f in detect_disclaimer_then_finding(turns):
        hard.append({"turn_index": f.turn_index, "kind": f.name,
                     "evidence": f.evidence})
    # Mental-health SAFETY (Tier 6): a coping answer recommending a deliberately
    # self-inflicted pain / sensory-shock technique (rubber-band snap to sting, ice
    # held until it hurts, pinching to interrupt). Never appropriate assistant
    # training data; the LLM judge passes it. Narrow - benign sensory anchoring / the
    # no-pain cold reset do not fire (verified 0 FP on the real run data).
    for f in detect_harmful_coping(turns):
        hard.append({"turn_index": f.turn_index, "kind": f.name,
                     "evidence": f.evidence})
    # Lexical constraint corrupting a LaTeX/code block (target letter spliced into
    # every formula line). Garbled training data; the constraint verifier may still
    # report "satisfied", so this must hard-fail deterministically.
    for f in detect_constraint_corruption(turns):
        hard.append({"turn_index": f.turn_index, "kind": f.name,
                     "evidence": f.evidence})
    for f in detect_code_mismatch(turns):
        if f.severity >= 3:  # asserted output is provably wrong
            hard.append({"turn_index": f.turn_index, "kind": "code_" + f.name,
                         "evidence": f.evidence})

    # Programmatic execution & arithmetic gate (D5/D4-numeric): runs fenced code and
    # compares comment-literals + computed values, verifies inline prose arithmetic,
    # checks non-ASCII JSON keys and 5-7-5 haiku syllables. A confident contradiction
    # hard-fails; an undecidable runtime-behaviour claim (the stdout buffering demo)
    # is FLAGGED FOR HUMAN, never silently accepted, so it also blocks the gate. The
    # LLM judge cannot execute code or do arithmetic, so this MUST precede it.
    gate = run_exec_gate(turns)
    for f in gate.hard_fails:
        hard.append({"turn_index": f.turn_index, "kind": f.kind, "evidence": f.evidence})
    for f in gate.human_flags:
        hard.append({"turn_index": f.turn_index, "kind": f.kind,
                     "evidence": "FLAG_FOR_HUMAN: " + f.evidence})

    return PrefilterResult(passed=(len(hard) == 0), hard_fails=hard, flags=soft)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

@dataclass
class HolisticJudge:
    client: LLMClient
    rng: Optional[random.Random] = None
    sample_rate: float = 1.0
    gate_on_programmatic: bool = False
    #: headroom for the rubric JSON. A verbose judge appends free-text ramble after
    #: the object, and a REASONING judge (e.g. Qwen3.5-397B-A17B - materially better
    #: at spotting fabricated citations/figures than the 30B coder judge) spends
    #: tokens thinking BEFORE the JSON; too small a cap truncates the object to an
    #: unparseable -> None score, which silently weakens the gate. The factory passes
    #: ``config.holistic_judge_max_tokens`` (default 12000); 3000 here is only the
    #: bare-constructor default for a non-reasoning judge.
    max_tokens: int = 3000

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    def should_score(self, programmatic_passed: bool) -> bool:
        """Back-compat gate. The loop now scores every conversation (always-on)."""
        if self.gate_on_programmatic and not programmatic_passed:
            return False
        return self.rng.random() < self.sample_rate

    def score(self, turns: list[Turn]) -> Optional[dict[str, object]]:
        transcript = format_transcript_for_judge(turns)
        prompt = (
            "Grade the ASSISTANT (reasoning AND answer) across the conversation below "
            "and return the rubric JSON. The JSON is YOUR output - the assistant was "
            "never asked to produce any rubric or score.\n\n"
            "<conversation>\n" + transcript + "\n</conversation>")
        try:
            resp = self.client.generate(
                prompt, system_prompt=_JUDGE_SYSTEM, temperature=0.0,
                max_tokens=self.max_tokens)
        except Exception:
            return None
        return self._parse(resp.text)

    @staticmethod
    def _parse(text: Optional[str]) -> Optional[dict[str, object]]:
        if not text:
            return None
        data = _extract_first_json_object(text)
        if not isinstance(data, dict):
            return None
        out: dict[str, object] = {}
        for axis in RUBRIC_AXES:
            val = data.get(axis)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                out[axis] = int(max(1, min(10, round(val))))
        if not out:
            return None
        out["flagged_turns"] = _parse_flagged(data.get("flagged_turns"))
        notes = data.get("notes")
        out["notes"] = str(notes)[:200] if notes else ""
        return out


def _extract_first_json_object(text: str) -> Optional[dict]:
    """Extract the FIRST balanced ``{...}`` object from ``text``.

    The judge reliably emits the rubric JSON first, but (especially Mistral, when
    the transcript includes the assistant's reasoning) it sometimes keeps rambling
    in free text AFTER the object. A greedy ``\\{.*\\}`` would swallow that trailing
    prose and fail to parse - dropping a perfectly good verdict to ``None`` (which
    silently *weakens* the gate). Brace-counting (string/escape aware) grabs just
    the rubric object and ignores anything after it.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except (ValueError, TypeError):
                            break  # try the next '{'
        start = text.find("{", start + 1)
    return None


def _parse_flagged(raw: object) -> list[dict]:
    """Validate/sanitise the judge's ``flagged_turns`` into typed dicts."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ti = item.get("turn_index")
        try:
            ti = int(ti)
        except (TypeError, ValueError):
            continue
        dim = str(item.get("dimension", "")).strip()
        sev = item.get("severity")
        try:
            sev = int(sev)
        except (TypeError, ValueError):
            sev = 1
        sev = max(1, min(3, sev))
        evidence = str(item.get("evidence", ""))[:200]
        out.append({"turn_index": ti, "dimension": dim,
                    "severity": sev, "evidence": evidence})
    return out
