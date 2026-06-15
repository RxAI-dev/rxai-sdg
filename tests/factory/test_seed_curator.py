"""SeedCurator tests (spec §5.1, §11.4)."""

import random

from rxai_sdg.factory.seed_curator import SeedCurator, DatasetSpec


def test_load_and_extract_query_variants():
    recs = [
        {"query": "What is 2+2?"},
        {"prompt": "Write a poem about the sea."},
        {"messages": [{"role": "user", "content": "Explain recursion."}]},
    ]
    seeds = SeedCurator().load_seeds(DatasetSpec(records=recs))
    assert len(seeds) == 3
    assert seeds[0].first_query == "What is 2+2?"
    assert "recursion" in seeds[2].first_query


def test_near_duplicate_dedup():
    recs = [
        {"query": "Explain how entropy relates to information"},
        {"query": "Explain how entropy relates to information!"},  # near dup
        {"query": "Write a haiku about rain"},
    ]
    seeds = SeedCurator(dedup_threshold=0.8).load_seeds(DatasetSpec(records=recs))
    assert len(seeds) == 2


def test_domain_inference():
    recs = [
        {"query": "Calculate the probability of two heads in a row."},
        {"query": "Write a python function to sort a list."},
        {"query": "Pretend you are a pirate and greet me."},
    ]
    seeds = SeedCurator().load_seeds(DatasetSpec(records=recs))
    domains = [s.domain for s in seeds]
    assert "math" in domains
    assert "coding" in domains
    assert "roleplay" in domains


def test_explicit_category_field_wins():
    recs = [{"query": "anything", "category": "humanities"}]
    seeds = SeedCurator().load_seeds(DatasetSpec(records=recs, category_field="category"))
    assert seeds[0].category == "humanities"


def test_haystack_flag_on_long_query():
    long_q = "word " * 400
    seeds = SeedCurator().load_seeds(
        DatasetSpec(records=[{"query": long_q}], haystack_fraction=1.0, haystack_min_chars=100))
    assert seeds[0].is_haystack is True


def test_balance_domains_respects_present_domains():
    recs = (
        [{"query": "Calculate 3*4", "category": "math"}] * 2
        + [{"query": "Write a story", "category": "writing"}] * 8
    )
    curator = SeedCurator(rng=random.Random(0))
    seeds = curator.load_seeds(DatasetSpec(records=recs, category_field="category"))
    balanced = curator.balance_domains(seeds, {"math": 3.0, "writing": 1.0},
                                       rng=random.Random(0))
    math_n = sum(1 for s in balanced if s.domain == "math")
    writing_n = sum(1 for s in balanced if s.domain == "writing")
    assert math_n > writing_n  # math up-weighted despite being rarer in source
