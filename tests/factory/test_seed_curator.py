"""SeedCurator tests (spec §5.1, §11.4)."""

from rxai_sdg.factory.clients import MockLLMClient
from rxai_sdg.factory.seed_curator import SeedCurator, HAYSTACK_MIN_CHARS


def test_accepts_strings_and_dicts_and_extracts_query():
    seeds = SeedCurator().curate([
        "What is 2+2?",
        {"prompt": "Write a poem about the sea."},
        {"messages": [{"role": "user", "content": "Explain recursion."}]},
    ])
    assert len(seeds) == 3
    assert seeds[0].first_query == "What is 2+2?"
    assert "recursion" in seeds[2].first_query


def test_near_duplicate_dedup():
    seeds = SeedCurator(dedup_threshold=0.8).curate([
        "Explain how entropy relates to information",
        "Explain how entropy relates to information!",  # near dup
        "Write a haiku about rain",
    ])
    assert len(seeds) == 2


def test_domain_inference():
    seeds = SeedCurator().curate([
        "Calculate the probability of two heads in a row.",
        "Write a python function to sort a list.",
        "Pretend you are a pirate and greet me.",
    ])
    domains = [s.domain for s in seeds]
    assert "math" in domains
    assert "coding" in domains
    assert "roleplay" in domains


def test_explicit_category_on_dict_wins():
    seeds = SeedCurator().curate([{"query": "anything", "category": "humanities"}])
    assert seeds[0].category == "humanities"


def test_haystack_flag_on_long_query():
    long_q = "word " * (HAYSTACK_MIN_CHARS // 2)  # well over the char threshold
    seeds = SeedCurator().curate([long_q])
    assert seeds[0].is_haystack is True


def test_llm_classifier_fallback_when_heuristic_inconclusive():
    client = MockLLMClient(default="humanities")
    curator = SeedCurator(classifier_client=client)
    seeds = curator.curate(["Tell me your thoughts on the meaning of it all."])
    assert seeds[0].category == "humanities"
    assert client.calls, "classifier should have been invoked"


def test_llm_classifier_not_called_when_heuristic_succeeds():
    client = MockLLMClient(default="humanities")
    curator = SeedCurator(classifier_client=client)
    seeds = curator.curate(["Write a python function to sort a list."])  # clearly coding
    assert seeds[0].category == "coding"
    assert not client.calls  # heuristic was sufficient


def test_llm_classifier_invalid_response_defaults_to_general():
    client = MockLLMClient(default="banana")  # not a valid category
    curator = SeedCurator(classifier_client=client)
    seeds = curator.curate(["Some vague musing without keywords."])
    assert seeds[0].category == "general"
