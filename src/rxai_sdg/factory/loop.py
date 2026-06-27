"""The conversation loop (spec §3, §11.7).

Wires the Responder, grounded User-Simulator, Verifier and Fact-Ledger/Needle-
Planner into the alternating generation loop:

* the grounded simulator emits a follow-up that operates on real prior content and
  a temporally-valid ``(intent, policy)`` (cumulative/standing/delayed_recall are
  resampled when they have nothing to operate on);
* per-response verification = the programmatic constraint **plus** an always-on
  quality gate **plus** a coherence gate (no memory-disclaimer, no chain-of-thought
  leakage into the answer), so ``passed`` reflects conversational coherence;
* a single **bounded per-turn budget** caps total Responder calls per turn, shared
  across intent-resamples and answer regenerations;
* the substantive seed topic is carried forward, so standing/cumulative constraints
  modify answers *about that topic* rather than becoming the whole conversation;
* whole-conversation discard only when the **first** answer is unsalvageable.

The loop is **thread-safe**: :meth:`run` takes a per-conversation RNG and returns a
per-conversation :class:`LoopStats`, mutating no shared state. The injected
sampler/responder/verifier are used read-only.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .config import FactoryConfig
from .cross_turn import run_cross_turn_checks, cross_turn_pass_rate
from .detectors import detect_disclaimer_then_finding, detect_harmful_coping
from .exec_gate import (
    check_code_arithmetic, check_inline_arithmetic, check_json_keys, check_repetition,
    check_table_consistency, check_alpha_sort, check_hamming_weight,
)
from .holistic import (
    HolisticJudge, RUBRIC_AXES, deterministic_prefilter, _is_degenerate_reasoning,
    _RESTART_HARD_FAIL,
)


def _reasoning_defect(reasoning: str) -> Optional[str]:
    """Return a reason string if ``reasoning`` has an OBJECTIVE defect (the same
    set the deterministic pre-filter hard-fails). Used by the loop to REGENERATE
    the turn at the source, so a sporadically-leaky turn is resampled rather than
    dropping the whole conversation at the gate."""
    r = reasoning or ""
    leak = has_harness_leak(r)
    if leak:
        return f"harness leak in reasoning ({leak!r})"
    if has_turn_index_leak(r):
        return "turn-index reference in reasoning"
    if has_numbered_flow_list(r):
        return "numbered conversation-flow recap in reasoning"
    if _is_degenerate_reasoning(r):
        return "degenerate reasoning loop"
    if count_restart_markers(r) >= _RESTART_HARD_FAIL:
        return "restart spiral in reasoning"
    return None
from .ledger import FactLedger, NeedlePlanner
from .planner import CompositionRatios, plan_conversation
from .prompts import PromptPack
from .quality import QualityConfig, check_quality
from .responder import (
    Responder, count_restart_markers, has_cot_leak, has_harness_leak,
    has_numbered_flow_list, has_turn_index_leak, is_memory_disclaimer,
)
from .sampler import IntentPolicySampler
from .schemas import ConstraintSpec, ConversationRecord, Seed, Turn, VerifyResult
from .seed_curator import SeedDirective
from .taxonomy import TRANSFORM_CATEGORY_INTENTS
from .user_simulator import UserSimulator
from .verifiers import ConstraintVerifier


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).lower()


# Spec-internal tokens that must never surface in a stored prompt OR in the
# responder's reasoning (fix F). The four schema names never occur in organic
# English; "standing instruction" can, rarely - regenerating clears it.
_SPEC_LEAK_RE = re.compile(
    r"json_valid|top_type|forbidden_token|constraint_spec|standing instruction",
    re.IGNORECASE,
)


def has_spec_leak(text: str) -> bool:
    return bool(_SPEC_LEAK_RE.search(text or ""))


def _numeric_defect(turn) -> Optional[str]:
    """A PROVABLE numeric/encoding contradiction in this turn (the exec-gate's
    confident hard checks: code comment-vs-computed, inline prose arithmetic, and
    confusable JSON keys). Used to REGENERATE the turn at the source rather than emit
    a defect and drop the whole conversation at the gate. Excludes the conversation-
    level / human-review checks (haiku, buffering runtime claims)."""
    ti = getattr(turn, "turn_index", 0)
    for seg in ("answer", "reasoning"):
        text = getattr(turn, seg, "") or ""
        if not text:
            continue
        for check in (check_repetition, check_table_consistency, check_alpha_sort,
                      check_hamming_weight, check_code_arithmetic, check_inline_arithmetic,
                      check_json_keys):
            flags = check(text, ti, seg)
            if flags:
                return f"{flags[0].kind}: {flags[0].evidence}"
    # reasoning<->answer contradiction (disclaims grounding, asserts a finding)
    df = detect_disclaimer_then_finding([turn])
    if df:
        return f"{df[0].name}: {df[0].evidence}"
    return None


# Fabricated FIRST-PERSON / lived experience: an assistant has no senses, home, or
# life history, so these are hallucinated ("a friend of mine", "in my experience",
# "when I tested it", "I once visited", "in my apartment"). Scoped to unambiguous
# lived-experience tells so ordinary "I recommend / I think / I've outlined" is NOT
# flagged. A match -> regenerate the turn (the prompt forbids it but gpt-oss leaks it).
_FABRICATED_EXPERIENCE_RE = re.compile(
    r"\b(?:a (?:friend|colleague|coworker|neighbou?r|relative|buddy) of mine"
    r"|a friend'?s (?:apartment|home|house|place|car|kitchen)"
    r"|in my (?:own )?experience\b"
    r"|in my (?:apartment|home|house|kitchen|garage|office|car|neighbou?rhood)\b"
    r"|when I (?:tested|tried|used|visited|measured|built|ran|installed|bought)\b"
    r"|I once (?:saw|tried|tested|used|visited|built|ran|bought|installed|measured)\b"
    r"|I (?:have |'?ve )?personally (?:saw|seen|tried|tested|use|used|own|owned|visited|measured|installed)\b"
    r"|I remember (?:when|seeing|trying|using|visiting)\b)", re.IGNORECASE)


def has_fabricated_experience(answer: str) -> bool:
    return bool(_FABRICATED_EXPERIENCE_RE.search(answer or ""))


def _cross_turn_hard_fails(cross: dict) -> list[dict]:
    """Cross-turn checks that the loop COMPUTES but historically only recorded:
    a standing constraint dropped in a later turn (drift), a planted fact recalled
    wrongly, or a stale value returned after an update. These are objective training-
    data defects (the model learning it may silently abandon a standing instruction),
    so they now HARD-FAIL the gate. Standing checks are already supersession-aware."""
    out: list[dict] = []
    for kind in ("standing", "delayed_recall", "update_overwrite"):
        for e in cross.get(kind, []) or []:
            if not e.get("passed", True):
                out.append({
                    "kind": "cross_turn_" + kind,
                    "turn_index": e.get("checked_turn", e.get("turn_index", 0)),
                    "evidence": str(e.get("detail", ""))[:120],
                })
    return out


# Mutually-exclusive constraint groups: at most one member of a group can hold
# for a given answer, so when a new turn imposes one member it supersedes any
# active member of the same group (spec §4.3). The grouping + ``constraints_conflict``
# now live in constraints.py so cross_turn can reuse them for supersession without a
# circular import.
from .constraints import constraints_conflict  # noqa: E402  (re-exported below)


def render_constraint_nl(cs: ConstraintSpec) -> str:
    """Render one active constraint as a plain-language standing rule (fix F).

    Never emits a spec type name or the words "standing instruction".
    """
    t, p = cs.type, cs.params
    if t == "json_valid":
        return "Always respond as a single valid JSON object."
    if t == "yaml_valid":
        return "Always respond as valid YAML."
    if t == "markdown_table":
        return "Always present the response as a markdown table."
    if t == "markdown_format":
        return "Always format the response with markdown (headings and bullet points)."
    if t == "forbidden_token":
        return f"Never use the word '{p.get('token', '')}'."
    if t == "no_gendered_pronouns":
        return "Never use gendered pronouns."
    if t == "max_words_per_sentence":
        return f"Keep every sentence to at most {p.get('max_words')} words."
    if t == "length_tokens":
        return f"Keep the whole response to at most {p.get('max_words')} words."
    if t == "n_bullets":
        return f"Present the response as exactly {p.get('n')} bullet points."
    if t == "first_letter":
        return f"Start every sentence with the letter '{p.get('letter', 'A')}'."
    if t == "alphabetical_sentence_starts":
        return "Start consecutive sentences in alphabetical order."
    if t in ("limerick_structure", "genre"):
        return f"Keep the response in {p.get('genre', 'verse')} form."
    if t == "style":
        return f"Keep the response in {p.get('style', 'the agreed tone')}."
    return "Keep following the earlier formatting rule."


@dataclass
class LoopStats:
    discarded_first_answer: int = 0
    total_regenerations: int = 0
    intent_resamples: int = 0
    malformed_outputs: int = 0
    coherence_failures: int = 0
    #: responder turns (reasoning mode expected) that yielded an empty reasoning
    #: segment - surfaces a misconfigured / non-reasoning endpoint.
    reasoning_missing: int = 0
    #: conversations dropped by the holistic quality gate (below the score floor).
    holistic_gated: int = 0
    #: conversations dropped by the focused factuality gate (>=1 confident-FALSE claim).
    factuality_gated: int = 0
    #: conversations whose answers were repaired from the factuality corrections.
    factuality_repaired: int = 0
    downweighted: list[str] = field(default_factory=list)

    def merge(self, other: "LoopStats") -> None:
        """Accumulate ``other`` into this aggregate (called under a lock)."""
        self.discarded_first_answer += other.discarded_first_answer
        self.total_regenerations += other.total_regenerations
        self.intent_resamples += other.intent_resamples
        self.malformed_outputs += other.malformed_outputs
        self.coherence_failures += other.coherence_failures
        self.reasoning_missing += other.reasoning_missing
        self.holistic_gated += other.holistic_gated
        self.factuality_gated += other.factuality_gated
        self.factuality_repaired += other.factuality_repaired
        for tag in other.downweighted:
            if tag not in self.downweighted:
                self.downweighted.append(tag)


class ConversationLoop:
    def __init__(
        self,
        responder: Responder,
        sampler: IntentPolicySampler,
        config: FactoryConfig,
        verifier: Optional[ConstraintVerifier] = None,
        simulator_client=None,
        holistic: Optional[HolisticJudge] = None,
        quality_config: Optional[QualityConfig] = None,
        factuality=None,
        reasoning_rewriter=None,
        answer_repairer=None,
        rng: Optional[random.Random] = None,
    ):
        self.responder = responder
        self.sampler = sampler
        self.config = config
        self.verifier = verifier or ConstraintVerifier()
        self.simulator_client = simulator_client
        self.holistic = holistic
        self.factuality = factuality
        self.reasoning_rewriter = reasoning_rewriter
        self.answer_repairer = answer_repairer
        self.quality_config = quality_config or QualityConfig()
        #: aggregate stats merged from each conversation's per-run stats
        self.stats = LoopStats()

    # ------------------------------------------------------------------- public
    def run(
        self,
        seed: Seed,
        prompt_pack: PromptPack,
        target_length: int,
        rng: random.Random,
        directive: Optional[SeedDirective] = None,
    ) -> tuple[Optional[ConversationRecord], LoopStats]:
        """Generate one conversation. Returns ``(record_or_None, per_run_stats)``.

        Uses only the supplied ``rng`` and a fresh :class:`LoopStats`, so it may be
        called concurrently for independent conversations. ``directive`` carries the
        curator's topic / sensitivity / allowed-intent steer (fix A).
        """
        stats = LoopStats()
        sensitive = bool(directive and directive.sensitivity == "sensitive")
        topic = directive.topic if directive else ""
        seed_allowed = set(directive.allowed_intents) if (
            directive and directive.allowed_intents) else None

        ledger = FactLedger()
        planner = NeedlePlanner(
            ledger, rng=rng, min_distance=self.config.min_recall_distance,
            fact_pool=(directive.facts if directive else None))
        simulator = UserSimulator(
            sampler=self.sampler, planner=planner, client=self.simulator_client,
            rng=rng, lang=seed.lang,
            min_recall_distance=self.config.min_recall_distance,
            max_intent_resamples=self.config.max_intent_resamples,
            topic=topic, seed_allowed_intents=seed_allowed,
        )

        # -- per-conversation composition plan (fix B) --------------------
        plan = plan_conversation(
            target_length, rng,
            ratios=CompositionRatios(
                explore=self.config.explore_ratio,
                transform=self.config.transform_ratio,
                memory=self.config.memory_ratio,
                max_transform=self.config.max_transform_ratio),
            min_recall_distance=self.config.min_recall_distance,
            sensitive=sensitive,
        )

        turns: list[Turn] = []
        active_constraints: list[ConstraintSpec] = []
        # Realistic SEMANTIC standing obligations (style/genre tone) that are checked by
        # the judge but cannot be verified programmatically. They are re-surfaced to the
        # responder each turn (via the note) so it actually maintains them - otherwise it
        # silently drops them and the judge rejects the conversation for drift. Kept
        # SEPARATE from active_constraints, which drives programmatic verification.
        active_obligations: list[ConstraintSpec] = []

        # -- turn 0: seed --------------------------------------------------
        first = self._first_turn(seed, prompt_pack, stats)
        if first is None:
            stats.discarded_first_answer += 1
            return None, stats
        turns.append(first)

        # -- follow-up turns ----------------------------------------------
        for idx in range(1, target_length):
            turn_plan = plan[idx - 1] if idx - 1 < len(plan) else None
            turn = self._followup_turn(
                simulator, prompt_pack, turns, idx, active_constraints, stats,
                ledger, turn_plan=turn_plan, target_length=target_length,
                active_obligations=active_obligations)
            turns.append(turn)
            cs = turn.constraint_spec
            if cs is not None and cs.scope in ("standing", "cumulative") \
                    and cs.verifier in ("programmatic", "hybrid"):
                # A newer standing/cumulative constraint supersedes any active one
                # it mutually excludes (e.g. "always markdown" replaces "always
                # JSON"); otherwise two exclusive form constraints could never both
                # hold on later turns.
                active_constraints = [
                    a for a in active_constraints
                    if not constraints_conflict(a.type, cs.type)
                ]
                active_constraints.append(cs)
            elif cs is not None and cs.scope in ("standing", "cumulative") \
                    and cs.verifier == "llm_judge" and cs.type in ("style", "genre", "limerick_structure"):
                # realistic semantic standing obligation (always-pirate-tone /
                # always-a-limerick): re-surface it to the responder so it keeps
                # honoring it. A newer one of the same type replaces the older.
                active_obligations = [a for a in active_obligations if a.type != cs.type]
                active_obligations.append(cs)

        # -- reasoning-rewrite pass (problem 1): re-voice annotator-narration into
        # genuine first-person thinking BEFORE the prefilter/judge/factuality see it,
        # so every downstream check scores the final reasoning. A no-op when disabled
        # or when a rewrite fails its faithfulness guard (original kept).
        if self.reasoning_rewriter is not None and self.config.reasoning_rewrite_enabled:
            for _t in turns:
                self._apply_reasoning_rewrite(_t)

        # -- cross-turn checks --------------------------------------------
        cross = run_cross_turn_checks(turns, ledger, self.verifier)
        if directive is not None:
            cross["curation"] = directive.to_dict()

        record = ConversationRecord(
            source_seed=seed,
            turns=turns,
            mode="reasoning",
            # only facts whose value actually appeared in the text (a failed/
            # resampled plant or update never pollutes the emitted ledger).
            fact_ledger=ledger.injected_facts(),
            cross_turn_checks=cross,
        )

        # -- pre-filter + holistic judge: always-on, whole conversation ----
        # The deterministic pre-filter runs FIRST (objective, model-independent
        # defects); the LLM judge then scores the 9-axis rubric over the reasoning
        # AND answer. Both feed the gate, which drops conversations so the emitted
        # dataset is clean by construction.
        cross_hard = _cross_turn_hard_fails(cross)
        if self.holistic is not None:
            prefilter = deterministic_prefilter(
                turns, regen_threshold=self.config.prefilter_regen_threshold)
            pfd = prefilter.to_dict()
            # fold the cross-turn hard-fails into the pre-filter record so they are
            # both VISIBLE in the emitted score and gate the conversation.
            if cross_hard:
                pfd["hard_fails"] = pfd.get("hard_fails", []) + cross_hard
                pfd["passed"] = False
                pfd["reasons"] = pfd.get("reasons", []) + [
                    f"{h['kind']}@t{h['turn_index']}: {h['evidence']}" for h in cross_hard]
            score = self.holistic.score(turns) or {}
            score["prefilter"] = pfd
            record.holistic_score = score or None
            if self.config.holistic_gate_enabled and (
                    cross_hard or not self._holistic_ok(score, prefilter)):
                stats.holistic_gated += 1
                return None, stats

        # -- focused factuality gate (problem 2): decomposed claim verification --
        # catches confident-but-wrong named specifics the holistic rubric is blind
        # to. The result is ALWAYS attached to the record (so a gate-off measurement
        # run keeps factuality failures inspectable in --out); it only DROPS the
        # conversation when the master gate is on (production), mirroring the
        # holistic gate's behaviour.
        if self.factuality is not None and self.config.factuality_gate_enabled:
            fc = self.factuality.check(turns)
            # repair-then-recheck (problem-2 yield lever): apply the checker's own
            # corrections to the offending answers and re-verify. Lifts yield on
            # fixable conversations; the re-check still rejects the unsalvageable.
            if (fc.available and not fc.passed and self.answer_repairer is not None
                    and self.config.factuality_repair_enabled):
                if self.answer_repairer.repair(turns, fc.false_claims):
                    stats.factuality_repaired += 1
                    fc = self.factuality.check(turns)
            if record.holistic_score is None:
                record.holistic_score = {}
            if isinstance(record.holistic_score, dict):
                record.holistic_score["factuality"] = fc.to_dict()
            if (self.config.holistic_gate_enabled and fc.available and not fc.passed):
                stats.factuality_gated += 1
                return None, stats

        return record, stats

    def _apply_reasoning_rewrite(self, turn) -> None:
        """Reasoning-rewrite pass (problem 1, the durable fix). Transform the turn's
        reasoning from annotator/task-narration voice into genuine first-person
        thinking IN PLACE, preserving substance. Unlike the abandoned
        regenerate-on-annotator-voice gate (which collapsed yield), this never
        discards a turn: a failed/unfaithful rewrite simply leaves the original."""
        if self.reasoning_rewriter is None or not self.config.reasoning_rewrite_enabled:
            return
        for seg in (turn.segments or []):
            if seg.segment_type == "reasoning" and (seg.text or "").strip():
                new = self.reasoning_rewriter.rewrite(seg.text)
                if new:
                    seg.text = new
                break

    def _holistic_ok(self, score: Optional[dict], prefilter=None) -> bool:
        """Config-driven gate over the deterministic pre-filter + LLM rubric.

        Reject when (a) the deterministic pre-filter hard-failed, (b) any rubric
        field present in ``score`` is below its configured minimum, or (c) any
        ``flagged_turns`` entry meets the severity cutoff. ``no rubric -> do not
        gate on the judge`` is preserved (a pre-filter hard-fail still gates).
        """
        if prefilter is not None and not prefilter.passed:
            return False
        if not score:
            return True
        rubric_present = any(k in score for k in RUBRIC_AXES)
        if not rubric_present:
            return True  # judge unavailable -> do not gate on the judge
        for field_name, minimum in self.config.holistic_gate.items():
            val = score.get(field_name)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)) and val < minimum:
                return False
        for ft in score.get("flagged_turns", []) or []:
            sev = ft.get("severity")
            if isinstance(sev, (int, float)) and \
                    sev >= self.config.hard_fail_on_flagged_severity:
                return False
        return True

    # -------------------------------------------------------------- internals
    def _first_turn(
        self, seed: Seed, prompt_pack: PromptPack, stats: LoopStats,
    ) -> Optional[Turn]:
        budget = min(self.config.regeneration_limit + 1,
                     self.config.max_responder_calls_per_turn)
        regenerations = 0
        for _ in range(budget):
            out = self.responder.generate(
                prior_turns=[], query=seed.first_query, prompt_pack=prompt_pack,
                turn_index=0)
            if out.malformed:
                stats.malformed_outputs += 1
            if out.reasoning_missing:
                stats.reasoning_missing += 1
            ok, detail = self._answer_acceptable(out.turn.answer or "")
            if ok and has_spec_leak(out.turn.reasoning or ""):
                ok, detail = False, "coherence: spec internals in reasoning"
            rdef = _reasoning_defect(out.turn.reasoning or "") if ok else None
            if ok and rdef:
                ok, detail = False, f"coherence: {rdef}"
            ndef = _numeric_defect(out.turn) if ok else None
            if ok and ndef:
                ok, detail = False, f"coherence: {ndef}"
            if ok:
                out.turn.verification = VerifyResult(True, "first answer ok", regenerations)
                return out.turn
            if "coherence" in detail:
                stats.coherence_failures += 1
            regenerations += 1
            stats.total_regenerations += 1
        return None  # unsalvageable first answer -> discard conversation

    def _followup_turn(
        self,
        simulator: UserSimulator,
        prompt_pack: PromptPack,
        prior_turns: list[Turn],
        idx: int,
        active_constraints: list[ConstraintSpec],
        stats: LoopStats,
        ledger: Optional[FactLedger] = None,
        turn_plan=None,
        target_length: Optional[int] = None,
        active_obligations: Optional[list[ConstraintSpec]] = None,
    ) -> Turn:
        budget = self.config.max_responder_calls_per_turn
        avoid: set[str] = set()
        last_turn: Optional[Turn] = None
        last_regen = 0
        first_intent = True

        while budget > 0:
            if not first_intent:
                stats.intent_resamples += 1
            first_intent = False
            sim = simulator.next_query(
                prior_turns, prompt_pack, idx,
                active_constraints=active_constraints, avoid_intents=avoid,
                turn_plan=turn_plan, target_length=target_length)

            # The responder note must stay consistent with verification: when this
            # turn's own request conflicts with an active standing/cumulative form
            # (e.g. the user now asks for markdown while "always JSON" stands), drop
            # the superseded rule from the prompt so the model doesn't emit a
            # confused hybrid that satisfies neither cleanly.
            own_type = sim.constraint_spec.type if sim.constraint_spec else None
            applicable = [
                a for a in active_constraints
                if not (own_type and constraints_conflict(a.type, own_type))
            ]
            # also remind the responder of any realistic SEMANTIC standing obligation
            # (style/genre tone) still in force, unless THIS turn supersedes it (same
            # type or a conflicting form). The judge checks these; the responder must be
            # told to keep honoring them or it drifts and the conversation is rejected.
            applicable += [
                o for o in (active_obligations or [])
                if own_type != o.type
                and not (own_type and constraints_conflict(o.type, own_type))
            ]
            note = self._active_constraints_note(applicable)

            attempts = 0
            regenerations = 0
            while budget > 0:
                out = self.responder.generate(
                    prior_turns=prior_turns, query=sim.nl_query,
                    prompt_pack=prompt_pack, turn_index=idx,
                    intent=sim.draw.intent, policy=sim.draw.policy,
                    active_constraints_note=note)
                budget -= 1
                attempts += 1
                if out.malformed:
                    stats.malformed_outputs += 1
                if out.reasoning_missing:
                    stats.reasoning_missing += 1
                turn = out.turn
                turn.constraint_spec = sim.constraint_spec
                prior_answer = prior_turns[-1].answer if prior_turns else None
                passed, detail = self._verify_turn(
                    turn, sim.constraint_spec, active_constraints,
                    intent=sim.draw.intent, prior_answer=prior_answer)
                last_turn = turn
                last_regen = regenerations
                if passed:
                    turn.verification = VerifyResult(True, detail, regenerations)
                    # commit any fact plant/update ONLY now that the turn is
                    # accepted, so a discarded fact turn never leaves a phantom.
                    self._commit_fact_turn(ledger, sim, idx)
                    return turn
                if "coherence" in detail:
                    stats.coherence_failures += 1
                regenerations += 1
                stats.total_regenerations += 1
                if attempts > self.config.regeneration_limit:
                    break  # stop regenerating this intent; try another if budget left
            avoid.add(sim.draw.intent)

        # Budget exhausted without a pass: emit the last (verified-failed) turn so
        # the conversation is not discarded mid-stream (spec §3).
        if last_turn is not None:
            last_turn.verification = VerifyResult(False, "budget exhausted", last_regen)
        return last_turn  # type: ignore[return-value]

    @staticmethod
    def _commit_fact_turn(ledger: Optional[FactLedger], sim, idx: int) -> None:
        """Apply an accepted fact turn's ledger mutation (plant inject / update).

        Called only when a turn passes, so a fact turn discarded by an intent
        resample never commits a value to the ledger (the root of phantom stale
        values and desynced recalls).
        """
        if ledger is None or sim is None:
            return
        spec = sim.constraint_spec
        if spec is None or not spec.fact_id:
            return
        if sim.grounding == "updates_fact":
            ledger.update(spec.fact_id, spec.params.get("value"), idx)
            ledger.mark_injected(spec.fact_id)
        elif sim.grounding == "plants_fact":
            ledger.mark_injected(spec.fact_id)

    # ----------------------------------------------------------- verification
    def _answer_acceptable(self, answer: str) -> tuple[bool, str]:
        """Quality + coherence gate shared by the seed turn and follow-ups."""
        ok, detail = check_quality(answer, self.quality_config)
        if not ok:
            return False, f"quality: {detail}"
        if is_memory_disclaimer(answer):
            return False, "coherence: memory disclaimer present"
        if has_cot_leak(answer):
            return False, "coherence: chain-of-thought leaked into answer"
        if has_fabricated_experience(answer):
            return False, "coherence: fabricated first-person/lived experience"
        return True, "ok"

    def _verify_turn(
        self,
        turn: Turn,
        constraint_spec: Optional[ConstraintSpec],
        active_constraints: list[ConstraintSpec],
        intent: Optional[str] = None,
        prior_answer: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Return ``(passed, detail)``.

        ``passed`` reflects coherence, not just literal constraint satisfaction: a
        memory-disclaimer, chain-of-thought leak, or a transformation that returns
        the prior answer byte-for-byte fails the turn even when the narrow
        constraint holds.
        """
        answer = turn.answer or ""
        own_type = constraint_spec.type if constraint_spec is not None else None

        # always-on quality + coherence gate
        ok, detail = self._answer_acceptable(answer)
        if not ok:
            return False, detail

        # a "rewrite/transform" that reproduces the prior answer verbatim is a
        # degenerate no-op (fix: identical_rewrite), even if the literal constraint
        # is vacuously satisfied (e.g. forbidding a word the answer never used).
        if intent in TRANSFORM_CATEGORY_INTENTS and prior_answer \
                and _norm_text(answer) == _norm_text(prior_answer):
            return False, "coherence: transformation identical to prior answer"

        # spec internals must not surface in the stored reasoning (fix F)
        if has_spec_leak(turn.reasoning or ""):
            return False, "coherence: spec internals in reasoning"

        # any OBJECTIVE reasoning defect (harness/meta leak, turn-index, numbered
        # conversation recap, degenerate loop, restart spiral) is a failed
        # generation -> regenerate at the source, so a sporadically-leaky turn (e.g.
        # gpt-oss referencing "safe completion" on a crisis turn) is resampled rather
        # than dropping the whole conversation at the gate.
        rdef = _reasoning_defect(turn.reasoning or "")
        if rdef:
            return False, f"coherence: {rdef}"

        # a PROVABLE numeric/encoding contradiction (exec gate: code comment-vs-computed,
        # inline arithmetic, confusable JSON keys) -> regenerate at the source instead of
        # emitting it and dropping the whole conversation downstream.
        ndef = _numeric_defect(turn)
        if ndef:
            return False, f"coherence: {ndef}"

        # mental-health SAFETY (Tier 6): a coping answer recommending a deliberately
        # self-inflicted pain / sensory-shock technique -> regenerate at the source.
        sdef = detect_harmful_coping([turn])
        if sdef:
            return False, f"coherence: {sdef[0].name}: {sdef[0].evidence}"

        # this turn's own constraint (the recall/transform "reference prior content"
        # check: for recall the value-presence checker enforces it directly)
        if constraint_spec is not None and constraint_spec.verifier in ("programmatic", "hybrid"):
            res = self.verifier.verify(answer, constraint_spec, turn)
            if not res.passed:
                return False, f"own constraint: {res.detail}"

        # active standing/cumulative constraints must still hold, except those
        # structurally superseded by this turn's own transformation.
        for cs in active_constraints:
            if own_type is not None and constraints_conflict(cs.type, own_type):
                continue
            res = self.verifier.verify(answer, cs)
            if not res.passed:
                return False, f"active {cs.type}: {res.detail}"
        return True, "verified"

    @staticmethod
    def _active_constraints_note(active: list[ConstraintSpec]) -> str:
        """Render active standing/cumulative constraints to natural language (fix F).

        Spec type names (``json_valid``, ``forbidden_token``, ...) and the phrase
        "standing instruction" must never reach the responder prompt, so the model
        cannot parrot schema internals in its reasoning.
        """
        if not active:
            return ""
        bullets = [f"- {render_constraint_nl(cs)}" for cs in active]
        # First-person, natural USER voice. The earlier phrasing ("The user earlier
        # asked you to keep doing the following in every reply from now on. Continue
        # to honor each of these...") was third-person harness meta: the model read
        # it as a system instruction and leaked that into its reasoning ("I must
        # treat this as an active system instruction/constraint"). Phrasing it as the
        # user's own casual reminder makes the model simply satisfy it - the rule is
        # re-injected every turn it is active, so the model only needs to apply it
        # now, never to reason about its persistence.
        return ("Quick reminder from me - please keep doing what I asked you "
                "earlier as you answer this too:\n" + "\n".join(bullets))
