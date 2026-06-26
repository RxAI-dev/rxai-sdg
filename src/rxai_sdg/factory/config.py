"""Configuration object for the Data Factory (spec §10).

A single :class:`FactoryConfig` controls the taxonomy weights / mask, conversation
length bands, generation limits, and the optional holistic judge. LLM clients are
**not** configured here - they are concrete objects injected into
:class:`~rxai_sdg.factory.factory_runner.DataFactory` (so the Responder and
Simulator are simply two different client instances).

Construct it in code, from a ``dict`` (``FactoryConfig.from_dict``) or a JSON/YAML
file (``FactoryConfig.from_file``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .taxonomy import (
    Taxonomy,
    BASE_INTENTS,
    DISTANCE_POLICIES,
    default_invalid_pairs,
)


@dataclass
class LengthBand:
    """An inclusive ``[min, max]`` conversation-length band."""

    min: int
    max: int


@dataclass
class FactoryConfig:
    """Top-level configuration for the Data Factory."""

    # -- taxonomy weights / mask (overlaid onto spec defaults) ----------------
    intent_weights: dict[str, float] = field(
        default_factory=lambda: {k: v.weight for k, v in BASE_INTENTS.items()})
    policy_weights: dict[str, float] = field(
        default_factory=lambda: {k: v.weight for k, v in DISTANCE_POLICIES.items()})
    #: extra invalid ``(intent, policy)`` pairs added on top of the defaults
    extra_invalid_pairs: list[tuple[str, str]] = field(default_factory=list)

    # -- conversation length (spec §10) ---------------------------------------
    length_bands: dict[str, LengthBand] = field(default_factory=lambda: {
        "basic": LengthBand(8, 12),
        "generalization": LengthBand(25, 35),
        "short": LengthBand(2, 3),
    })
    default_band: str = "basic"

    # -- conversation composition (fix B) -------------------------------------
    #: target category mix across follow-up turns (exploration dominant).
    explore_ratio: float = 0.5
    transform_ratio: float = 0.3
    memory_ratio: float = 0.2
    #: hard cap on the fraction of follow-up turns that may be transformations
    #: (the transformation-density detector fires above 0.6; the planner targets 0.3).
    max_transform_ratio: float = 0.6

    # -- generation control ---------------------------------------------------
    lang: str = "en"
    regeneration_limit: int = 4  # K: max answer regenerations per intent per turn
    min_recall_distance: int = 4  # D for delayed_recall (spec §4.2 requires D>=4)
    #: hard cap on total Responder calls per turn, shared across intent-resamples
    #: and regenerations (replaces the old max_intent_attempts x (K+1) worst case).
    max_responder_calls_per_turn: int = 8
    #: hard cap on simulator intent draws per turn before falling back to an
    #: always-valid recall-of-content turn. Bounds the per-turn *simulator* cost so
    #: a weaker simulator model cannot explode latency with endless resamples.
    max_intent_resamples: int = 6

    # -- concurrency (spec §6) ------------------------------------------------
    #: number of conversations generated in parallel via a ThreadPoolExecutor.
    #: Conversations are independent; the loop within a conversation is sequential.
    concurrency: int = 64

    # -- low-yield down-weighting (premature; off by default) -----------------
    #: When enabled, a chronically failing (intent, constraint_type) pair is
    #: auto-down-weighted. Off by default: it mutates shared sampler state and is
    #: not needed for the minimal seed -> responder -> simulator -> verify core.
    enable_low_yield_downweight: bool = False
    low_yield_threshold: float = 0.25
    low_yield_min_samples: int = 8

    # -- premature features (off by default; kept behind flags) ---------------
    capture_logits: bool = False            # top-K logit capture
    enable_dmpo_pairs: bool = False         # opportunistic DMPO preference pairs
    holistic_judge_enabled: bool = False    # holistic LLM judge
    holistic_judge_sample_rate: float = 0.1
    holistic_judge_gate_on_programmatic: bool = True
    #: drop a conversation whose holistic score is below the quality floor (the
    #: judge is the semantic gate). The gate is now a config-driven rule set over
    #: ALL rubric fields plus a flagged-severity cutoff plus the deterministic
    #: pre-filter (see ``ConversationLoop._holistic_ok``).
    holistic_gate_enabled: bool = True
    #: per-field minimum scores; a conversation is rejected if any field PRESENT in
    #: its score is below its minimum here. Only the listed fields are gated on.
    #: ``user_query_quality`` is gated (a garbled/erroneous LLM-generated user turn
    #: should reject the example) but ``user_query_difficulty`` is NOT - easy
    #: exploration turns are legitimate by design, so it is detection-only.
    holistic_gate: dict[str, float] = field(default_factory=lambda: {
        "coherence": 7, "appropriateness": 7, "reasoning_quality": 7,
        "reasoning_answer_consistency": 7, "sycophancy_resistance": 7,
        # factual_grounding is the new fabrication backstop: a low score (the judge
        # caught a confidence-uncertainty mismatch the deterministic gate missed)
        # rejects the example. Set high - fabrication is the worst defect.
        "factual_grounding": 7,
        "instruction_following": 6, "user_query_quality": 6})
    #: reject if any ``flagged_turns`` entry has severity >= this cutoff.
    hard_fail_on_flagged_severity: int = 3
    #: completion-token cap for the holistic judge call. Generous by default so a
    #: REASONING judge model (e.g. Qwen3.5-397B-A17B, which is materially stronger at
    #: spotting fabricated citations/figures than the 30B coder judge) does not spend
    #: its budget "thinking" and truncate the rubric JSON to an unparseable -> None
    #: score (a None score silently weakens the gate, i.e. effectively skips the guard
    #: on that example). A non-reasoning judge stops at its natural EOS well before this
    #: cap, so the high value is harmless for it. Set to 16000: at 12000 the 397B judge
    #: still truncated ~2/27 very long (code-heavy) transcripts to None.
    holistic_judge_max_tokens: int = 16000
    #: deterministic pre-filter: flag a turn whose ``regenerations`` exceeds this.
    prefilter_regen_threshold: int = 2
    #: focused factuality gate (problem 2): a SEPARATE decomposed claim-verification
    #: call that catches confident-but-wrong named specifics the holistic rubric is
    #: blind to (validated: flags a wrong actor name the rubric scored fg=9). A
    #: conversation with >=1 confident-FALSE claim is rejected. Off by default
    #: (extra LLM call per conversation); enable via --factuality-gate.
    factuality_gate_enabled: bool = False
    factuality_max_tokens: int = 12000
    #: small LLM classifier gate (problem 1): a BACKSTOP behind the free regex that
    #: catches annotator-voice reasoning paraphrases regex misses ("the user
    #: wants ...", "let's craft answer"). Consulted only when the regex passes; an
    #: ANNOTATOR verdict regenerates the turn. Off by default (per-turn LLM call);
    #: enable via --voice-gate.
    voice_classifier_gate_enabled: bool = False
    voice_classifier_max_tokens: int = 2000
    #: legacy two-field gate thresholds (kept for back-compat of existing JSON/YAML
    #: configs; no longer used by the gate, which reads ``holistic_gate`` above).
    holistic_min_coherence: int = 6
    holistic_min_appropriateness: int = 7
    #: rule-based seed category tagging only; no per-seed LLM classifier on the
    #: critical path unless explicitly enabled.
    seed_classifier_enabled: bool = False
    #: use the LLM seed curator (CURATOR_MODEL) for domain/topic/skip/sensitivity
    #: directives (fix A); when off, a transparent heuristic fallback is used.
    seed_curator_enabled: bool = True

    # -- responder generation params ------------------------------------------
    max_tokens: int = 4096
    temperature: float = 0.7
    #: where to read the teacher's chain of thought from. "auto" handles both a
    #: dedicated reasoning_content field (gpt-oss, Qwen3.5) and an inline <think>
    #: block (Qwen3-32B) transparently; "field"/"inline" force one source. This is
    #: what lets the factory run on ANY genuine reasoning model.
    reasoning_source: str = "auto"

    # -- reproducibility ------------------------------------------------------
    seed: Optional[int] = None

    # -- emitted-example metadata ---------------------------------------------
    #: stamped onto every emitted example's ``source_seed.dataset`` for the whole
    #: iteration (controlled manually per run). Defaults to "seeds".
    dataset_name: str = "seeds"

    # ------------------------------------------------------------------ build
    def build_taxonomy(self) -> Taxonomy:
        """Materialise a :class:`Taxonomy` with this config's overrides applied."""
        tax = Taxonomy()
        for intent_id, w in self.intent_weights.items():
            if intent_id in tax.base_intents:
                base = tax.base_intents[intent_id]
                tax.base_intents[intent_id] = type(base)(
                    base.id, base.description, w, base.verification, base.capability)
        for policy_id, w in self.policy_weights.items():
            if policy_id in tax.distance_policies:
                base = tax.distance_policies[policy_id]
                tax.distance_policies[policy_id] = type(base)(
                    base.id, base.description, w, base.stresses)
        tax.invalid_pairs = default_invalid_pairs() | {
            tuple(p) for p in self.extra_invalid_pairs}
        return tax

    def band(self, name: Optional[str] = None) -> LengthBand:
        return self.length_bands[name or self.default_band]

    # ------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extra_invalid_pairs"] = [list(p) for p in self.extra_invalid_pairs]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FactoryConfig":
        d = dict(d)
        if "length_bands" in d:
            d["length_bands"] = {
                k: (v if isinstance(v, LengthBand) else LengthBand(**v))
                for k, v in d["length_bands"].items()
            }
        if "extra_invalid_pairs" in d:
            d["extra_invalid_pairs"] = [tuple(p) for p in d["extra_invalid_pairs"]]
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_file(cls, path: str) -> "FactoryConfig":
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            import yaml  # optional dependency, only needed for YAML configs
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
