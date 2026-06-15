"""The conversation loop (spec §3, §11.7).

Wires the Responder, User-Simulator, Verifier and Fact-Ledger/Needle-Planner
into the alternating generation loop with:

* per-response verification fired only for machine-checkable constraints, plus a
  general quality gate that always applies;
* regeneration of the *Responder's answer* up to ``K`` times on failure;
* intent-resampling on terminal failure mid-conversation (the conversation is
  **not** discarded);
* whole-conversation discard only when the **first** answer is unsalvageable;
* enforcement of active ``standing``/``cumulative`` constraints on later answers;
* low-yield down-weighting of chronically failing constraint types.

The loop produces a single reasoning-mode :class:`ConversationRecord`; derived
instruct/mixed variants are created afterwards by the
:class:`~rxai_sdg.factory.writer.SegmentWriter`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .config import FactoryConfig
from .cross_turn import run_cross_turn_checks, cross_turn_pass_rate
from .holistic import HolisticJudge
from .ledger import FactLedger, NeedlePlanner
from .prompts import PromptPack
from .quality import QualityConfig, check_quality
from .responder import Responder
from .sampler import IntentPolicySampler
from .schemas import ConstraintSpec, ConversationRecord, Seed, Turn, VerifyResult
from .user_simulator import UserSimulator
from .verifiers import ConstraintVerifier


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


@dataclass
class LoopStats:
    discarded_first_answer: int = 0
    total_regenerations: int = 0
    intent_resamples: int = 0
    downweighted: list[str] = field(default_factory=list)


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
        self.rng = rng or random.Random(config.seed)
        self.stats = LoopStats()

    # ------------------------------------------------------------------- public
    def run(
        self,
        seed: Seed,
        prompt_pack: PromptPack,
        target_length: Optional[int] = None,
    ) -> Optional[ConversationRecord]:
        if target_length is None:
            band = self.config.band()
            target_length = self.rng.randint(band.min, band.max)

        ledger = FactLedger()
        planner = NeedlePlanner(
            ledger, rng=self.rng, min_distance=self.config.min_recall_distance)
        simulator = UserSimulator(
            sampler=self.sampler, planner=planner, client=self.simulator_client,
            rng=self.rng, lang=seed.lang,
            min_recall_distance=self.config.min_recall_distance,
            naturalize=self.simulator_client is not None,
        )

        turns: list[Turn] = []
        active_constraints: list[ConstraintSpec] = []
        dmpo_pairs: list[dict] = []

        # -- turn 0: seed --------------------------------------------------
        first = self._first_turn(seed, prompt_pack)
        if first is None:
            self.stats.discarded_first_answer += 1
            return None
        turns.append(first)

        # -- follow-up turns ----------------------------------------------
        for idx in range(1, target_length):
            turn = self._followup_turn(
                simulator, prompt_pack, turns, idx, active_constraints, dmpo_pairs)
            turns.append(turn)
            cs = turn.constraint_spec
            if cs is not None and cs.scope in ("standing", "cumulative") \
                    and cs.verifier in ("programmatic", "hybrid"):
                # A newer standing/cumulative constraint supersedes any active
                # one it mutually excludes (e.g. "always answer in markdown"
                # replaces an earlier "always answer in JSON"). Without this,
                # two exclusive form constraints would be impossible to satisfy
                # together on every later turn.
                active_constraints = [
                    a for a in active_constraints
                    if not constraints_conflict(a.type, cs.type)
                ]
                active_constraints.append(cs)

        # -- cross-turn checks --------------------------------------------
        cross = run_cross_turn_checks(turns, ledger, self.verifier)
        if dmpo_pairs:
            cross["dmpo_pairs"] = dmpo_pairs  # opportunistic byproduct (spec §6.3)
        programmatic_passed = cross_turn_pass_rate(cross) == 1.0 and all(
            (t.verification is None or t.verification.passed) for t in turns)

        record = ConversationRecord(
            source_seed=seed,
            turns=turns,
            mode="reasoning",
            fact_ledger=ledger.facts(),
            cross_turn_checks=cross,
        )

        # -- optional holistic judge --------------------------------------
        if self.holistic is not None and self.holistic.should_score(programmatic_passed):
            record.holistic_score = self.holistic.score(turns)

        return record

    # -------------------------------------------------------------- internals
    def _first_turn(self, seed: Seed, prompt_pack: PromptPack) -> Optional[Turn]:
        for attempt in range(self.config.regeneration_limit + 1):
            turn = self.responder.generate(
                prior_turns=[], query=seed.first_query, prompt_pack=prompt_pack,
                turn_index=0)
            ok, detail = check_quality(turn.answer or "", self.quality_config)
            if ok:
                turn.verification = VerifyResult(True, "first answer ok", attempt)
                return turn
            self.stats.total_regenerations += 1
        return None  # unsalvageable first answer -> discard conversation

    def _followup_turn(
        self,
        simulator: UserSimulator,
        prompt_pack: PromptPack,
        prior_turns: list[Turn],
        idx: int,
        active_constraints: list[ConstraintSpec],
        dmpo_pairs: list[dict],
    ) -> Turn:
        avoid: set[str] = set()
        max_intent_attempts = 4
        for intent_attempt in range(max_intent_attempts):
            sim = simulator.next_query(
                prior_turns, prompt_pack, idx, avoid_intents=avoid)
            turn, ok, rejected = self._generate_and_verify(
                prompt_pack, prior_turns, idx, sim.nl_query, sim.constraint_spec,
                sim.draw.intent, sim.draw.policy, active_constraints)
            if rejected is not None:
                dmpo_pairs.append(rejected)
            if ok:
                return turn
            # terminal failure for this intent: resample a different intent.
            self.stats.intent_resamples += 1
            avoid.add(sim.draw.intent)
        # Could not satisfy any sampled intent; emit the last (verified-failed)
        # turn so the conversation is not discarded (spec §3).
        return turn

    def _generate_and_verify(
        self,
        prompt_pack: PromptPack,
        prior_turns: list[Turn],
        idx: int,
        nl_query: str,
        constraint_spec: Optional[ConstraintSpec],
        intent: str,
        policy: str,
        active_constraints: list[ConstraintSpec],
    ):
        note = self._active_constraints_note(active_constraints)
        rejected_pair: Optional[dict] = None
        last_turn: Optional[Turn] = None
        regenerations = 0
        for attempt in range(self.config.regeneration_limit + 1):
            turn = self.responder.generate(
                prior_turns=prior_turns, query=nl_query, prompt_pack=prompt_pack,
                turn_index=idx, intent=intent, policy=policy,
                active_constraints_note=note)
            turn.constraint_spec = constraint_spec
            passed, detail, own_passed = self._verify_turn(
                turn, constraint_spec, active_constraints)
            # Low-yield bookkeeping reflects only the turn's OWN constraint, so a
            # failure caused by a conflicting active constraint never penalises
            # the current intent/constraint type.
            self._record_yield(intent, constraint_spec, own_passed)
            last_turn = turn
            if passed:
                turn.verification = VerifyResult(True, detail, regenerations)
                return turn, True, rejected_pair
            # capture a DMPO preference pair opportunistically (rejected answer)
            if rejected_pair is None:
                rejected_pair = {
                    "turn_index": idx, "query": nl_query,
                    "rejected": turn.answer, "reason": detail,
                }
            regenerations += 1
            self.stats.total_regenerations += 1
        # terminal failure
        if last_turn is not None:
            last_turn.verification = VerifyResult(False, "terminal fail", regenerations)
            if rejected_pair is not None and last_turn.answer != rejected_pair["rejected"]:
                rejected_pair = None  # only keep pair when a later attempt accepted
        return last_turn, False, None

    def _verify_turn(
        self,
        turn: Turn,
        constraint_spec: Optional[ConstraintSpec],
        active_constraints: list[ConstraintSpec],
    ) -> tuple[bool, str, bool]:
        """Return ``(passed, detail, own_constraint_passed)``."""
        answer = turn.answer or ""
        own_type = constraint_spec.type if constraint_spec is not None else None
        # always-on quality gate
        ok, detail = check_quality(answer, self.quality_config)
        if not ok:
            return False, f"quality: {detail}", True
        # this turn's own constraint
        own_passed = True
        if constraint_spec is not None and constraint_spec.verifier in ("programmatic", "hybrid"):
            res = self.verifier.verify(answer, constraint_spec, turn)
            own_passed = res.passed
            if not res.passed:
                return False, f"own constraint: {res.detail}", False
        # active standing/cumulative constraints must still hold, except those
        # that are structurally superseded by this turn's own transformation
        # (e.g. a standing "answer in JSON" cannot co-exist with "rewrite as a
        # limerick"; the newer explicit request wins for this turn).
        for cs in active_constraints:
            if own_type is not None and constraints_conflict(cs.type, own_type):
                continue
            res = self.verifier.verify(answer, cs)
            if not res.passed:
                return False, f"active {cs.type}: {res.detail}", own_passed
        return True, "verified", own_passed

    def _record_yield(self, intent: str, constraint_spec: Optional[ConstraintSpec],
                      passed: bool) -> None:
        if constraint_spec is None or constraint_spec.verifier == "llm_judge":
            return
        ctype = constraint_spec.type
        self.sampler.record_outcome(intent, ctype, passed)
        if self.sampler.maybe_downweight(
                intent, ctype, self.config.low_yield_threshold,
                self.config.low_yield_min_samples):
            tag = f"{intent}:{ctype}"
            if tag not in self.stats.downweighted:
                self.stats.downweighted.append(tag)

    @staticmethod
    def _active_constraints_note(active: list[ConstraintSpec]) -> str:
        if not active:
            return ""
        bullets = []
        for cs in active:
            bullets.append(f"- ({cs.scope}) {cs.type} {cs.params}")
        return ("Standing instructions still in force (keep satisfying them):\n"
                + "\n".join(bullets))
