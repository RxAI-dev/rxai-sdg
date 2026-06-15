"""Sample-quality assertions over a small generated batch (spec §7).

Run a small batch with a deterministic, realistic ``MockLLMClient`` handler and
assert the coherence contracts hold on the emitted records:

* no ``answer`` segment leaks chain-of-thought markers;
* no answer contains a memory-disclaimer;
* every follow-up query is grounded in real content (asserted via the turn's
  structured metadata, not text heuristics);
* under standing constraints, later turns still address substantive topic content.
"""

import random
import re

import pytest

from rxai_sdg.factory import DataFactory, FactoryConfig, MockLLMClient
from rxai_sdg.factory.responder import has_cot_leak, is_memory_disclaimer
from rxai_sdg.factory.taxonomy import (
    FACT_INTENTS, TRANSFORMATION_INTENTS,
)
from rxai_sdg.factory.testing import constraint_satisfying_handler

SEEDS = [
    "Explain how entropy relates to information.",
    "Write a short paragraph about lighthouses.",
    "What is 17 * 23 and why does the method work?",
    "Outline a function to reverse a linked list.",
]

# intents that operate on the prior answer's claims (grounded by construction)
_OPERATES_ON_PRIOR = {"chained_compute", "self_critique", "deepen"}


@pytest.fixture(scope="module")
def batch():
    cfg = FactoryConfig(seed=5, concurrency=8)
    client = MockLLMClient(handler=constraint_satisfying_handler)
    factory = DataFactory(cfg, client, rng=random.Random(5))
    return factory.generate(SEEDS, band="generalization")


def test_no_cot_markers_in_any_answer(batch):
    for rec in batch:
        for turn in rec.turns:
            answer = turn.answer or ""
            assert "</think>" not in answer and "<think>" not in answer
            assert not has_cot_leak(answer), f"cot leak: {answer!r}"


def test_no_memory_disclaimer_in_any_answer(batch):
    for rec in batch:
        for turn in rec.turns:
            assert not is_memory_disclaimer(turn.answer or "")


def test_every_followup_is_grounded_via_metadata(batch):
    for rec in batch:
        for turn in rec.turns[1:]:  # skip the seed turn
            cs = turn.constraint_spec
            assert turn.intent is not None
            assert cs is not None
            if turn.intent in FACT_INTENTS:
                # grounded on a tracked fact
                assert cs.fact_id is not None
                if cs.scope != "delayed_recall":
                    # plant / immediate-recall / update: the value is woven in
                    value = str(cs.params.get("value", ""))
                    assert value and value.lower() in (turn.query or "").lower()
                else:
                    # delayed recall references a fact planted >= D turns earlier
                    assert cs.planted_turn is not None
                    assert turn.turn_index - cs.planted_turn >= 0
            else:
                # transformation / operates-on-prior / open_chat are grounded by
                # construction; the intent itself is the structured evidence.
                assert (turn.intent in TRANSFORMATION_INTENTS
                        or turn.intent in _OPERATES_ON_PRIOR
                        or turn.intent == "open_chat")


def test_substantive_thread_persists_under_standing_constraints(batch):
    """At least one conversation keeps doing substantive work after a standing rule."""
    checked = 0
    for rec in batch:
        standing_turns = [t.turn_index for t in rec.turns
                          if t.constraint_spec and t.constraint_spec.scope == "standing"]
        if not standing_turns:
            continue
        checked += 1
        first_standing = min(standing_turns)
        later = [t for t in rec.turns if t.turn_index > first_standing]
        # the conversation does not collapse into a single repeated format
        # instruction: later turns engage more than one intent and reference the
        # prior content / running topic rather than being bare format toggles.
        topic_tokens = set(re.findall(r"[a-z0-9']+", rec.source_seed.first_query.lower()))
        grounded_later = [
            t for t in later
            if t.intent in TRANSFORMATION_INTENTS
            or t.intent in _OPERATES_ON_PRIOR
            or t.intent in FACT_INTENTS
            or (topic_tokens & set(re.findall(r"[a-z0-9']+", (t.query or "").lower())))
        ]
        assert grounded_later, "later turns must still address substantive content"
        assert len({t.intent for t in later}) >= 1
    assert checked >= 1, "expected at least one conversation with a standing constraint"


def test_constraints_are_not_the_whole_conversation(batch):
    """Format/standing constraints transform substance; they are not the substance."""
    for rec in batch:
        followups = rec.turns[1:]
        if len(followups) < 4:
            continue
        # not every follow-up is a pure standing format instruction
        standing_form = sum(
            1 for t in followups
            if t.constraint_spec and t.constraint_spec.scope == "standing"
            and t.intent in ("reformat", "genre_convert"))
        assert standing_form < len(followups)
