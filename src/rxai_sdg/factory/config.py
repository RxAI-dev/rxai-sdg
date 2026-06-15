"""Configuration object for the Data Factory (spec §10).

A single :class:`FactoryConfig` controls intent/policy weights, the invalidity
mask, conversation-length bands, domain mix, regeneration limits, derived-variant
sets, logit capture, and the holistic judge. It can be constructed in code, from
a plain ``dict`` (``FactoryConfig.from_dict``) or a JSON/YAML file
(``FactoryConfig.from_file``), making the whole pipeline reproducible.
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

    def clamp(self, n: int) -> int:
        return max(self.min, min(self.max, n))


@dataclass
class ClientConfig:
    """Pluggable LLM client configuration (Responder or Simulator).

    The Responder and Simulator should be *different* models where possible to
    avoid self-collusion (spec §5.3). ``provider`` is advisory metadata; the
    actual client object is injected at construction time.
    """

    model_name: str = "gpt-4"
    provider: str = "openai"
    api_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    use_ollama: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    #: when importing short reasoning datasets, expand them to ~this length
    expand_short_to: int = 10

    # -- seeds / domains ------------------------------------------------------
    domain_mix: dict[str, float] = field(default_factory=lambda: {
        "writing": 1.0, "math": 1.5, "coding": 1.0, "extraction": 1.0,
        "stem": 1.0, "humanities": 1.0, "reasoning": 1.5, "roleplay": 1.0,
    })
    haystack_fraction: float = 0.15
    lang: str = "en"

    # -- generation control ---------------------------------------------------
    regeneration_limit: int = 4  # K
    #: a (intent_type, constraint_type) pair is auto-down-weighted once its pass
    #: rate over at least ``low_yield_min_samples`` attempts drops below this.
    low_yield_threshold: float = 0.25
    low_yield_min_samples: int = 8
    #: minimum distance D for delayed_recall (spec §4.2 requires D>=4)
    min_recall_distance: int = 4

    # -- HuggingFace output dataset (spec §10) --------------------------------
    hf_dataset_id: Optional[str] = None
    hf_config_name: Optional[str] = None
    hf_split: str = "train"

    # -- derived variants / reasoning post-processing (spec §8) ---------------
    # NOTE: deriving instruct/mixed variants is a SEPARATE post-processing step
    # (see rxai_sdg.factory.variants), not part of the generation pipeline. These
    # settings configure that optional step when it is run independently later.
    derived_variants: list[str] = field(default_factory=lambda: ["reasoning", "instruct", "mixed"])
    mixed_mode_keep_ratio: float = 0.5  # fraction of turns that keep reasoning in "mixed"

    # -- distillation / holistic judge ----------------------------------------
    capture_logits: bool = False
    holistic_judge_enabled: bool = False
    holistic_judge_sample_rate: float = 0.1  # only score this fraction
    holistic_judge_gate_on_programmatic: bool = True  # only score conversations that pass

    # -- clients (separate models) --------------------------------------------
    responder: ClientConfig = field(default_factory=lambda: ClientConfig(model_name="gpt-4"))
    simulator: ClientConfig = field(
        default_factory=lambda: ClientConfig(model_name="gpt-4o-mini"))

    # -- reproducibility ------------------------------------------------------
    seed: Optional[int] = None

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
        # asdict turns LengthBand into nested dicts already; keep tuples as lists
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
        for key in ("responder", "simulator"):
            if key in d and not isinstance(d[key], ClientConfig):
                d[key] = ClientConfig(**d[key])
        if "extra_invalid_pairs" in d:
            d["extra_invalid_pairs"] = [tuple(p) for p in d["extra_invalid_pairs"]]
        # ignore unknown keys defensively
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
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
