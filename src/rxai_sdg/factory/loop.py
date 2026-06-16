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
from .holistic import HolisticJudge
from .ledger import FactLedger, NeedlePlanner
from .planner import CompositionRatios, plan_conversation
from .prompts import PromptPack
from .quality import QualityConfig, check_quality
from .responder import Responder, has_cot_leak, is_memory_disclaimer
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


# Mutually-exclusive constraint groups: at most one member of a group can hold
# for a given answer, so when a new turn imposes one member it supersedes any
# active member of the same group (spec §4.3 - cumulative/standing stacking only
# makes sense for compatible constraints).
_CONFLICT_GROUPS: list[frozenset[str]] = [
    frozenset({"json_valid", "yaml_valid", "markdown_table", "markdown_format",
               "limerick_structure"}),  # answer "form"
    frozenset({"first_letter", "alphabetical_sentence_starts"}),  # sentence starts
]


def constraints_conflict(type_a: str, type_b: str) -> bool:
    """True when two constraint types cannot both hold on the same answer."""
    if type_a == type_b:
        return False
    return any(type_a in g and type_b in g for g in _CONFLICT_GROUPS)


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
        rng: Optional[random.Random] = None,
    ):
        self.responder = responder
        self.sampler = sampler
        self.config = config
        self.verifier = verifier or ConstraintVerifier()
        self.simulator_client = simulator_client
        self.holistic = holistic
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
            ledger, rng=rng, min_distance=self.config.min_recall_distance)
        simulator = UserSimulator(
            sampler=self.sampler, planner=planner, client=self.simulator_client,
            rng=rng, lang=seed.lang,
            min_recall_distance=self.config.min_recall_distance,
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
                ledger, turn_plan=turn_plan, target_length=target_length)
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

        # -- holistic judge: always-on, whole conversation (fix G) --------
        if self.holistic is not None:
            record.holistic_score = self.holistic.score(turns)
            # Quality gate: the judge is the semantic gate (it catches failures the
            # programmatic verifier misses). Drop conversations below the floor so
            # the emitted dataset is clean by construction.
            if self.config.holistic_gate_enabled and not self._holistic_ok(
                    record.holistic_score):
                stats.holistic_gated += 1
                return None, stats

        return record, stats

    def _holistic_ok(self, score: Optional[dict]) -> bool:
        if not score:
            return True  # no score (judge unavailable) -> do not gate
        coh = score.get("coherence")
        appr = score.get("appropriateness")
        if isinstance(coh, (int, float)) and coh < self.config.holistic_min_coherence:
            return False
        if isinstance(appr, (int, float)) and appr < self.config.holistic_min_appropriateness:
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
        return ("Also keep doing the following in your reply, as the user asked "
                "earlier, while still answering the new request:\n" + "\n".join(bullets))
