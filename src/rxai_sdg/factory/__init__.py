"""rxai_sdg.factory - the Data Factory.

A stateless synthetic multi-turn conversation generator for stateful Reactive
Transformer (RxT / rc-RxT) models. It orchestrates two stateless LLMs (a
Responder/Teacher and a User-Simulator) in an alternating loop seeded from
existing conversational datasets, deliberately covering a typed taxonomy of
follow-up intents x memory-distance policies, verifying machine-checkable
constraints per response and across turns, and emitting a segment-typed output
schema suitable for staged RxT training.

The module is **stateless**: it produces text. No RxT / STM / memory machinery
lives here (spec non-goals).

Quick start
-----------
>>> from rxai_sdg.factory import DataFactory, FactoryConfig, MockLLMClient
>>> cfg = FactoryConfig(seed=0)
>>> records = DataFactory(cfg, MockLLMClient(default="<think>x</think> Answer.")) \
...     .generate(["Explain entropy."])

The package's core (schemas, taxonomy, sampler, verifiers) imports with only the
Python standard library, so it can be unit-tested without ``openai`` /
``datasets`` installed. Provider clients are imported lazily.
"""

from __future__ import annotations

# schemas
from .schemas import (
    Seed,
    Segment,
    Turn,
    ConstraintSpec,
    VerifyResult,
    Fact,
    ConversationRecord,
    validate_record,
    SchemaError,
)

# taxonomy / config
from .taxonomy import (
    Taxonomy,
    BaseIntent,
    DistancePolicy,
    BASE_INTENTS,
    DISTANCE_POLICIES,
    TRANSFORMATION_INTENTS,
    FACT_INTENTS,
    POLICY_TO_SCOPE,
    default_invalid_pairs,
)
from .taxonomy import (
    COMPOSITION_CATEGORIES,
    INTENT_TO_CATEGORY,
    TRANSFORM_CATEGORY_INTENTS,
    SENSITIVE_ALLOWED_INTENTS,
)
from .config import FactoryConfig, LengthBand
from .planner import plan_conversation, TurnPlan, CompositionRatios

# clients
from .clients import LLMClient, LLMResponse, MockLLMClient, OpenAILLMClient

# components
from .verifiers import ConstraintVerifier, register_language_stubs, registered_types
from .sampler import IntentPolicySampler, SamplerDraw
from .ledger import FactLedger, NeedlePlanner
from .seed_curator import SeedCurator, SeedDirective, CuratedSeed, EVAL_CATEGORIES
from .prompts import PromptPack, get_prompt_pack
from .responder import (
    Responder,
    ResponderOutput,
    ParsedResponse,
    parse_response,
    split_reasoning_answer,
    is_memory_disclaimer,
    has_cot_leak,
    has_harness_leak,
    has_turn_index_leak,
    has_trailing_artifact,
    sanitize_reasoning,
    sanitize_generated_text,
    format_transcript,
    format_transcript_for_judge,
)
from .user_simulator import UserSimulator, SimulatorResult, GROUNDING_KINDS
from .cross_turn import run_cross_turn_checks, cross_turn_pass_rate
from .holistic import (
    HolisticJudge,
    RUBRIC_AXES,
    PrefilterResult,
    deterministic_prefilter,
)
from .factuality import FactChecker, FactCheckResult
from .reasoning_voice import ReasoningVoiceClassifier
from .reasoning_rewrite import ReasoningRewriter
from .writer import SegmentWriter, flag_dangling_references
from .variants import derive_variants
from .dataset import (
    FactoryDatasetPostprocessor,
    record_to_row,
    row_to_record,
)
from .quality import QualityConfig, check_quality

# orchestration
from .loop import ConversationLoop, LoopStats
from .factory_runner import DataFactory, FactoryRunStats

__all__ = [
    # schemas
    "Seed", "Segment", "Turn", "ConstraintSpec", "VerifyResult", "Fact",
    "ConversationRecord", "validate_record", "SchemaError",
    # taxonomy / config
    "Taxonomy", "BaseIntent", "DistancePolicy", "BASE_INTENTS", "DISTANCE_POLICIES",
    "TRANSFORMATION_INTENTS", "FACT_INTENTS", "POLICY_TO_SCOPE", "default_invalid_pairs",
    "COMPOSITION_CATEGORIES", "INTENT_TO_CATEGORY", "TRANSFORM_CATEGORY_INTENTS",
    "SENSITIVE_ALLOWED_INTENTS",
    "FactoryConfig", "LengthBand",
    "plan_conversation", "TurnPlan", "CompositionRatios",
    # clients
    "LLMClient", "LLMResponse", "MockLLMClient", "OpenAILLMClient",
    # components
    "ConstraintVerifier", "register_language_stubs", "registered_types",
    "IntentPolicySampler", "SamplerDraw",
    "FactLedger", "NeedlePlanner",
    "SeedCurator", "SeedDirective", "CuratedSeed", "EVAL_CATEGORIES",
    "PromptPack", "get_prompt_pack",
    "Responder", "ResponderOutput", "ParsedResponse", "parse_response",
    "split_reasoning_answer", "is_memory_disclaimer", "has_cot_leak",
    "has_harness_leak", "has_turn_index_leak", "has_trailing_artifact",
    "sanitize_reasoning", "sanitize_generated_text",
    "format_transcript", "format_transcript_for_judge",
    "UserSimulator", "SimulatorResult", "GROUNDING_KINDS",
    "run_cross_turn_checks", "cross_turn_pass_rate",
    "HolisticJudge", "RUBRIC_AXES", "PrefilterResult", "deterministic_prefilter",
    "FactChecker", "FactCheckResult", "ReasoningVoiceClassifier", "ReasoningRewriter",
    "SegmentWriter", "flag_dangling_references", "derive_variants",
    "FactoryDatasetPostprocessor", "record_to_row", "row_to_record",
    "QualityConfig", "check_quality",
    # orchestration
    "ConversationLoop", "LoopStats", "DataFactory", "FactoryRunStats",
]
