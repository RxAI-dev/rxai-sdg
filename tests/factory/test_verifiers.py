"""Fixture suite of known pass/fail cases per constraint type (spec §11.2, §12)."""

import pytest

from rxai_sdg.factory.schemas import ConstraintSpec
from rxai_sdg.factory.verifiers import (
    ConstraintVerifier, resolve_checker, has_checker, registered_types,
    register_language_stubs, UnknownCheckerError, WILDCARD_LANG,
)

V = ConstraintVerifier()


def _spec(ctype, params=None, lang="en", verifier="programmatic"):
    return ConstraintSpec(intent="t", type=ctype, params=params or {}, lang=lang,
                          verifier=verifier)


# ---- universal -------------------------------------------------------------

@pytest.mark.parametrize("text,ok", [
    ('{"a": 1}', True),
    ('```json\n{"a": 1}\n```', True),
    ("not json", False),
    ("[1, 2, 3]", True),
])
def test_json_valid(text, ok):
    assert V.verify(text, _spec("json_valid")).passed is ok


def test_json_top_type_object():
    assert V.verify("[1,2]", _spec("json_valid", {"top_type": "object"})).passed is False
    assert V.verify('{"a":1}', _spec("json_valid", {"top_type": "object"})).passed is True


@pytest.mark.parametrize("text,ok", [
    ("key: value\nother: 2", True),
    ("just a sentence", False),  # bare scalar rejected
])
def test_yaml_valid(text, ok):
    res = V.verify(text, _spec("yaml_valid"))
    # yaml may be unavailable in some envs; only assert when it ran
    if "not available" not in res.detail:
        assert res.passed is ok


def test_markdown_table():
    good = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert V.verify(good, _spec("markdown_table")).passed is True
    assert V.verify("no table here", _spec("markdown_table")).passed is False


def test_length_tokens():
    assert V.verify("one two three", _spec("length_tokens", {"max_words": 5})).passed is True
    assert V.verify("one two three four five six", _spec("length_tokens", {"max_words": 5})).passed is False
    assert V.verify("one", _spec("length_tokens", {"min_words": 2})).passed is False


def test_n_bullets():
    txt = "- a\n- b\n- c"
    assert V.verify(txt, _spec("n_bullets", {"n": 3})).passed is True
    assert V.verify(txt, _spec("n_bullets", {"n": 2})).passed is False


def test_fact_recall_exact_and_fuzzy():
    assert V.verify("Your color is teal.", _spec("fact_recall", {"value": "teal"})).passed is True
    assert V.verify("Your color is blue.", _spec("fact_recall", {"value": "teal"})).passed is False
    fuzzy = _spec("fact_recall", {"value": "blue meridian", "match": "fuzzy"})
    assert V.verify("the meridian is blue today", fuzzy).passed is True


def test_fact_update_rejects_stale():
    spec = _spec("fact_update", {"value": "crimson", "stale_values": ["teal"]})
    assert V.verify("It is now crimson.", spec).passed is True
    assert V.verify("It is teal, now crimson.", spec).passed is False  # stale present
    assert V.verify("It is teal.", spec).passed is False  # current missing


# ---- english ---------------------------------------------------------------

def test_first_letter():
    good = "Apples are red. Avocados are green. Always fresh."
    assert V.verify(good, _spec("first_letter", {"letter": "A"})).passed is True
    bad = "Apples are red. Bananas are yellow."
    assert V.verify(bad, _spec("first_letter", {"letter": "A"})).passed is False


def test_first_letter_prose_only_ignores_latex_and_code():
    # The rule applies to PROSE: a clean answer with intact LaTeX/code passes (the
    # model must NOT prefix the target letter onto formula/code lines to satisfy it).
    ans = ("Such codes are doubly-even. \\[ G=\\begin{pmatrix} 1&1&0&0 \\end{pmatrix} \\] "
           "So every codeword has even weight. ```python\nx = compute()\n``` "
           "Stacking the rows shows the pattern.")
    assert V.verify(ans, _spec("first_letter", {"letter": "S"})).passed is True


def test_max_words_per_sentence():
    assert V.verify("Short one. Tiny two.", _spec("max_words_per_sentence", {"max_words": 3})).passed is True
    assert V.verify("This sentence has clearly more than three words.",
                    _spec("max_words_per_sentence", {"max_words": 3})).passed is False


def test_forbidden_token():
    assert V.verify("a clean rewrite", _spec("forbidden_token", {"token": "very"})).passed is True
    assert V.verify("a very clean rewrite", _spec("forbidden_token", {"token": "very"})).passed is False


def test_no_gendered_pronouns():
    assert V.verify("The team shipped it.", _spec("no_gendered_pronouns")).passed is True
    assert V.verify("He shipped it.", _spec("no_gendered_pronouns")).passed is False


def test_alphabetical_sentence_starts():
    assert V.verify("Apple first. Banana next. Cat last.",
                    _spec("alphabetical_sentence_starts")).passed is True
    assert V.verify("Banana first. Apple next.",
                    _spec("alphabetical_sentence_starts")).passed is False


def test_limerick_structure():
    five = "a\nb\nc\nd\ne"
    assert V.verify(five, _spec("limerick_structure")).passed is True
    assert V.verify("a\nb\nc", _spec("limerick_structure")).passed is False


# ---- registry / verifier behaviour -----------------------------------------

def test_llm_judge_skipped_by_verifier():
    res = V.verify("anything", _spec("style", {"style": "x"}, verifier="llm_judge"))
    assert res.passed is True
    assert "skipped" in res.detail


def test_hybrid_runs_programmatic_part():
    res = V.verify("one two three four five six", _spec("length_tokens", {"max_words": 3}, verifier="hybrid"))
    assert res.passed is False  # programmatic length part still enforced


def test_unknown_checker_is_unverifiable():
    res = V.verify("x", _spec("does_not_exist"))
    assert res.passed is False
    assert "unverifiable" in res.detail


def test_universal_checker_resolves_for_any_lang():
    assert has_checker("json_valid", "de")  # wildcard fallback
    assert resolve_checker("json_valid", "fr") is resolve_checker("json_valid", "en")


def test_language_specific_not_available_for_other_lang_until_stub():
    # english-only checker has no german impl by default -> falls back to nothing
    assert not has_checker("no_gendered_pronouns", "de")
    register_language_stubs("de", ["no_gendered_pronouns"])
    assert has_checker("no_gendered_pronouns", "de")
    res = V.verify("Er ist da.", _spec("no_gendered_pronouns", lang="de"))
    assert res.passed is False
    assert "not implemented" in res.detail


def test_registered_types_lists_en_and_universal():
    types = registered_types("en")
    assert "first_letter" in types and "json_valid" in types
