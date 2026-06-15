"""Core data schemas for the Data Factory.

These dataclasses define the structured types that flow through the
Data Factory pipeline: ``Seed`` -> ``Turn``/``Segment`` -> ``ConversationRecord``.

The module is intentionally dependency-light (standard library only) so that
schemas, verifiers and samplers can be imported and unit-tested without the
heavy LLM / dataset dependencies used by the rest of ``rxai_sdg``.

All dataclasses provide ``to_dict``/``from_dict`` helpers that produce the exact
JSON shapes described in the implementation spec (§4.4 and §7), so a record can
be serialised straight to a JSONL training file.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Literal type aliases (kept as plain strings so records are JSON-friendly)
# ---------------------------------------------------------------------------

SegmentType = Literal["query", "reasoning", "answer"]
VerifierKind = Literal["programmatic", "hybrid", "llm_judge"]
Scope = Literal["current_turn", "cumulative", "standing", "delayed_recall"]
Mode = Literal["reasoning", "instruct", "mixed"]
MatchKind = Literal["exact", "fuzzy"]


# ---------------------------------------------------------------------------
# Segment / Turn
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A typed slice of a single turn.

    ``segment_type`` maps to the model's segment-type embeddings during
    training. Reasoning segments are *not* masked.
    """

    segment_type: SegmentType
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"segment_type": self.segment_type, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(segment_type=d["segment_type"], text=d["text"])


@dataclass
class ConstraintSpec:
    """Machine-checkable specification attached to a verifiable follow-up.

    This single object drives the verifier and is stored verbatim in the output
    record. ``lang`` is a first-class field from day one (default ``"en"``).
    """

    intent: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    lang: str = "en"
    verifier: VerifierKind = "programmatic"
    scope: Scope = "current_turn"
    applies_from_turn: Optional[int] = None
    planted_turn: Optional[int] = None
    fact_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConstraintSpec":
        return cls(
            intent=d["intent"],
            type=d["type"],
            params=dict(d.get("params", {})),
            lang=d.get("lang", "en"),
            verifier=d.get("verifier", "programmatic"),
            scope=d.get("scope", "current_turn"),
            applies_from_turn=d.get("applies_from_turn"),
            planted_turn=d.get("planted_turn"),
            fact_id=d.get("fact_id"),
        )


@dataclass
class VerifyResult:
    """Outcome of a per-response verification."""

    passed: bool
    detail: str = ""
    regenerations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "detail": self.detail,
            "regenerations": self.regenerations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerifyResult":
        return cls(
            passed=d["passed"],
            detail=d.get("detail", ""),
            regenerations=d.get("regenerations", 0),
        )


@dataclass
class Turn:
    """A single conversational turn (user query + teacher reasoning/answer)."""

    turn_index: int
    segments: list[Segment] = field(default_factory=list)
    intent: Optional[str] = None
    policy: Optional[str] = None
    constraint_spec: Optional[ConstraintSpec] = None
    verification: Optional[VerifyResult] = None
    reasoning_flag: bool = True
    topk_logits_ref: Optional[str] = None

    # -- convenience accessors -------------------------------------------------
    def segment_text(self, segment_type: SegmentType) -> Optional[str]:
        for seg in self.segments:
            if seg.segment_type == segment_type:
                return seg.text
        return None

    @property
    def query(self) -> Optional[str]:
        return self.segment_text("query")

    @property
    def reasoning(self) -> Optional[str]:
        return self.segment_text("reasoning")

    @property
    def answer(self) -> Optional[str]:
        return self.segment_text("answer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "segments": [s.to_dict() for s in self.segments],
            "intent": self.intent,
            "policy": self.policy,
            "constraint_spec": self.constraint_spec.to_dict() if self.constraint_spec else None,
            "verification": self.verification.to_dict() if self.verification else None,
            "reasoning_flag": self.reasoning_flag,
            "topk_logits_ref": self.topk_logits_ref,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        cs = d.get("constraint_spec")
        ver = d.get("verification")
        return cls(
            turn_index=d["turn_index"],
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            intent=d.get("intent"),
            policy=d.get("policy"),
            constraint_spec=ConstraintSpec.from_dict(cs) if cs else None,
            verification=VerifyResult.from_dict(ver) if ver else None,
            reasoning_flag=d.get("reasoning_flag", True),
            topk_logits_ref=d.get("topk_logits_ref"),
        )


# ---------------------------------------------------------------------------
# Seed / Fact
# ---------------------------------------------------------------------------

@dataclass
class Seed:
    """A curated starting point for a conversation."""

    dataset: str
    first_query: str
    category: str = "general"
    domain: str = "general"
    lang: str = "en"
    is_haystack: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Seed":
        return cls(
            dataset=d.get("dataset", "unknown"),
            first_query=d["first_query"],
            category=d.get("category", "general"),
            domain=d.get("domain", "general"),
            lang=d.get("lang", "en"),
            is_haystack=d.get("is_haystack", False),
        )


@dataclass
class Fact:
    """A salient fact tracked by the :class:`FactLedger`.

    ``value`` always reflects the *latest* value; ``value_history`` records the
    ordered list of ``(turn, value)`` entries (including the initial plant) so
    update/overwrite sequences are auditable.
    """

    fact_id: str
    value: Any
    planted_turn: int
    fact_type: str = "generic"
    value_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "value": self.value,
            "planted_turn": self.planted_turn,
            "fact_type": self.fact_type,
            "value_history": list(self.value_history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fact":
        return cls(
            fact_id=d["fact_id"],
            value=d["value"],
            planted_turn=d["planted_turn"],
            fact_type=d.get("fact_type", "generic"),
            value_history=list(d.get("value_history", [])),
        )


# ---------------------------------------------------------------------------
# Conversation record (the emitted training example)
# ---------------------------------------------------------------------------

@dataclass
class ConversationRecord:
    """One derived training example (see §7 of the spec)."""

    source_seed: Seed
    turns: list[Turn] = field(default_factory=list)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mode: Mode = "reasoning"
    fact_ledger: list[Fact] = field(default_factory=list)
    cross_turn_checks: dict[str, Any] = field(default_factory=dict)
    holistic_score: Optional[dict[str, Any]] = None

    @property
    def length(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "source_seed": self.source_seed.to_dict(),
            "mode": self.mode,
            "length": self.length,
            "turns": [t.to_dict() for t in self.turns],
            "fact_ledger": [f.to_dict() for f in self.fact_ledger],
            "cross_turn_checks": self.cross_turn_checks,
            "holistic_score": self.holistic_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConversationRecord":
        return cls(
            conversation_id=d.get("conversation_id", str(uuid.uuid4())),
            source_seed=Seed.from_dict(d["source_seed"]),
            mode=d.get("mode", "reasoning"),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
            fact_ledger=[Fact.from_dict(f) for f in d.get("fact_ledger", [])],
            cross_turn_checks=d.get("cross_turn_checks", {}),
            holistic_score=d.get("holistic_score"),
        )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class SchemaError(ValueError):
    """Raised when a record does not conform to the output schema."""


def validate_record(record: dict[str, Any]) -> None:
    """Validate a serialised :class:`ConversationRecord` against the schema.

    Raises :class:`SchemaError` with a descriptive message on the first problem
    found. Used by the smoke test to assert every emitted record is well-formed.
    """

    required_top = {
        "conversation_id", "source_seed", "mode", "length",
        "turns", "fact_ledger", "cross_turn_checks", "holistic_score",
    }
    missing = required_top - set(record)
    if missing:
        raise SchemaError(f"record missing top-level keys: {sorted(missing)}")

    if record["mode"] not in ("reasoning", "instruct", "mixed"):
        raise SchemaError(f"invalid mode: {record['mode']!r}")

    seed = record["source_seed"]
    for k in ("dataset", "first_query", "category", "domain", "lang", "is_haystack"):
        if k not in seed:
            raise SchemaError(f"source_seed missing key: {k!r}")

    if record["length"] != len(record["turns"]):
        raise SchemaError(
            f"length {record['length']} != number of turns {len(record['turns'])}"
        )

    valid_segment_types = {"query", "reasoning", "answer"}
    for i, turn in enumerate(record["turns"]):
        for k in ("turn_index", "segments", "reasoning_flag"):
            if k not in turn:
                raise SchemaError(f"turn {i} missing key: {k!r}")
        seg_types = [s.get("segment_type") for s in turn["segments"]]
        for st in seg_types:
            if st not in valid_segment_types:
                raise SchemaError(f"turn {i} has invalid segment_type: {st!r}")
        if "query" not in seg_types:
            raise SchemaError(f"turn {i} has no query segment")
        if "answer" not in seg_types:
            raise SchemaError(f"turn {i} has no answer segment")
        # Mode/segment consistency: instruct records must not carry reasoning.
        if record["mode"] == "instruct" and "reasoning" in seg_types:
            raise SchemaError(f"turn {i} carries reasoning in instruct-mode record")
        cs = turn.get("constraint_spec")
        if cs is not None:
            for k in ("intent", "type", "verifier", "scope", "lang"):
                if k not in cs:
                    raise SchemaError(f"turn {i} constraint_spec missing key: {k!r}")
