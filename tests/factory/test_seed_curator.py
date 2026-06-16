"""SeedCurator tests (spec §5.1, fix A)."""

import json

from rxai_sdg.factory.clients import MockLLMClient, LLMResponse
from rxai_sdg.factory.seed_curator import (
    SeedCurator, CuratedSeed, SeedDirective, HAYSTACK_MIN_CHARS,
)


def test_accepts_strings_and_dicts_and_returns_curated_seeds():
    out = SeedCurator().curate([
        "What is 2+2?",
        {"prompt": "Write a poem about the sea."},
        {"messages": [{"role": "user", "content": "Explain recursion."}]},
    ])
    assert all(isinstance(c, CuratedSeed) for c in out)
    assert len(out) == 3
    assert out[0].seed.first_query == "What is 2+2?"
    assert "recursion" in out[2].seed.first_query


def test_near_duplicate_dedup():
    out = SeedCurator(dedup_threshold=0.8).curate([
        "Explain how entropy relates to information",
        "Explain how entropy relates to information!",  # near dup
        "Write a haiku about rain",
    ])
    assert len(out) == 2


def test_heuristic_skips_contentless_greetings():
    out = SeedCurator().curate(["hi", "What's up doc?", "Explain entropy clearly."])
    assert len(out) == 1
    assert "entropy" in out[0].seed.first_query


def test_heuristic_flags_sensitive_and_restricts_intents():
    out = SeedCurator().curate(
        ["I feel hopeless and don't want to be here anymore."])
    assert len(out) == 1
    d = out[0].directive
    assert d.sensitivity == "sensitive"
    assert d.action == "keep"
    # restricted to the safe supportive subset
    assert set(d.allowed_intents) == {
        "deepen", "expand", "compress", "open_chat", "self_critique"}


def test_non_sensitive_seed_has_no_intent_restriction():
    out = SeedCurator().curate(["Explain how a TCP handshake works."])
    assert out[0].directive.sensitivity == "none"
    assert out[0].directive.allowed_intents is None


def test_explicit_category_on_dict_wins():
    out = SeedCurator().curate([{"query": "anything substantive here", "category": "humanities"}])
    assert out[0].seed.category == "humanities"


def test_haystack_flag_on_long_query():
    long_q = "word " * (HAYSTACK_MIN_CHARS // 2)  # well over the char threshold
    out = SeedCurator().curate([long_q])
    assert out[0].seed.is_haystack is True


def test_llm_curator_parses_json_directive():
    payload = json.dumps({
        "domain": "humanities", "topic": "meaning of life",
        "action": "keep", "sensitivity": "none"})
    client = MockLLMClient(default=LLMResponse(text=payload))
    curator = SeedCurator(client=client)
    out = curator.curate(["Tell me your thoughts on the meaning of it all."])
    assert client.calls, "curator LLM should have been invoked"
    assert out[0].directive.domain == "humanities"
    assert out[0].directive.topic == "meaning of life"


def test_llm_curator_skip_drops_seed():
    payload = json.dumps({"domain": "general", "topic": "greeting",
                          "action": "skip", "sensitivity": "none"})
    client = MockLLMClient(default=LLMResponse(text=payload))
    out = SeedCurator(client=client).curate(["yo"])
    assert out == []


def test_llm_curator_invalid_response_falls_back_to_heuristic():
    client = MockLLMClient(default=LLMResponse(text="not json at all"))
    out = SeedCurator(client=client).curate(["Write a python function to sort a list."])
    assert len(out) == 1
    # heuristic fallback still produces a usable directive
    assert out[0].directive.action == "keep"
