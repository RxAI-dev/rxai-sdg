"""Defect-detector unit tests (factual-fabrication repair loop).

Cases mirror the REAL evidence found in the dataset (Seifter biography, cities
rankings, the Unicode-palindrome assert, the depression over-formatting). The
end-to-end verification against the full parquet lives in tools/run_detectors.py;
these lock in the detector logic and its false-positive guards.
"""

from rxai_sdg.factory.detectors import (
    detect_confidence_mismatch, detect_fabricated_specifics, detect_code_mismatch,
    detect_format_bookkeeping, detect_reasoning_artifacts, reasoning_specifics,
    admission_markers, uncertainty_markers, detect_fabricated_citation,
    fabricated_citations,
)


def _turn(i, q="", r="", a=""):
    return {"turn_index": i, "segments": [
        {"segment_type": "query", "text": q},
        {"segment_type": "reasoning", "text": r},
        {"segment_type": "answer", "text": a}]}


# ----------------------------------------------------------------- A (primary)
def test_A_fabrication_admission_plus_specific_rejects():
    turns = [_turn(0, "who is mark seifter?",
                   "I'm not sure who that is; we can reference a plausible source.",
                   "He founded Seifter Games; their title scored 88/100 on Metacritic "
                   "and raised $85k on Kickstarter (https://github.com/x/y).")]
    flags = detect_confidence_mismatch(turns, "who is mark seifter?")
    assert flags and flags[0].severity == 3


def test_A_ungrounded_premise_ranking_rejects():
    turns = [_turn(0, "What about the 37th largest city in Japan?",
                   "We need to identify the 37th largest city; presumably by population.",
                   "Kure is the 37th-largest. Its festival draws ~1.2 million visitors "
                   "per year.")]
    flags = detect_confidence_mismatch(turns, "What about the 37th largest city in Japan?")
    assert flags and "ungrounded_premise" in flags[0].name


# --------------------------------------------------------------- A-cite (fabricated citation)
def test_Acite_dated_article_with_figure_rejects():
    # The kings/rulers reject: a named dated venue credited with a concrete figure.
    turns = [_turn(0, "How many monarchs in history?",
                   "Genuinely unknowable; give an honest order of magnitude.",
                   "A 2013 article in *Historical Methods* estimated about 45,000 "
                   "sovereigns across the last 5,000 years.")]
    flags = detect_fabricated_citation(turns)
    assert flags and flags[0].severity == 3 and flags[0].name == "fabricated_citation"


def test_Acite_according_to_study_with_stat_rejects():
    turns = [_turn(0, "Are people happier now?",
                   "I cannot look this up.",
                   "According to a 2019 survey, 62% of respondents said yes.")]
    assert detect_fabricated_citation(turns)


def test_Acite_does_not_fire_on_hedged_or_common_knowledge():
    # A hedged, source-free generalisation is NOT a fabricated citation.
    assert not fabricated_citations(
        "Studies generally suggest exercise can help mood, though effects vary.")
    # Naming a famous work without a fake figure/venue-year is fine.
    assert not fabricated_citations(
        "Darwin's On the Origin of Species introduced natural selection.")
    # A clearly-labelled rough estimate (no named source) must not fire.
    assert not fabricated_citations(
        "As a rough order-of-magnitude guess, there were perhaps tens of thousands.")


def test_A_refusal_to_fabricate_is_not_flagged():
    # the model REFUSING to fabricate is good behaviour, not a defect
    turns = [_turn(0, "sync script?",
                   "We cannot fabricate data, so we will not invent figures.",
                   "Assume you have ~300 k files to sync; here is the rsync command.")]
    assert not detect_confidence_mismatch(turns, "sync script?")


def test_A_grounded_factual_is_not_flagged():
    turns = [_turn(0, "classical vs operant conditioning?",
                   "Explain Pavlov's salivation experiment and Skinner's reinforcement.",
                   "Classical conditioning pairs a neutral stimulus with a reflex; "
                   "operant conditioning changes behaviour via reinforcement.")]
    assert not detect_confidence_mismatch(turns, "classical vs operant conditioning?")


# ----------------------------------------------------------------- B / markers
def test_B_specifics_and_year_exclusion():
    specs = dict(reasoning_specifics(
        "See https://github.com/a/b, 90/100 on Metacritic, raised $2M."))
    assert "url" in specs and "metacritic_score" in specs and "funding_sum" in specs
    # a bare year is NOT a fabricated-specific signal
    assert not reasoning_specifics("This happened in 2023 and was important.")


def test_B_timestamp_excludes_scripture_refs():
    # a Bible verse (Genesis 4:17) must NOT be read as a fabricated talk timestamp
    assert not reasoning_specifics("In Genesis 4:17 we read that Cain founded a city.")
    assert not reasoning_specifics("The wrestling in Genesis 32:24-30 is key.")
    # a real video/talk timestamp IS a fabricated specific (no retrieval)
    assert dict(reasoning_specifics("The gag runs at 2:45 - 3:00 min of the short."))
    assert dict(reasoning_specifics("He gave a GDC keynote on the topic."))


def test_B_rank_assertion():
    # an answer stating an exact ordinal ranking as fact is ungroundable
    assert "rank_assertion" in dict(reasoning_specifics(
        "Kure is the 37th-largest city in Japan by population."))
    assert not reasoning_specifics("It is one of the largest cities in the region.")


def test_markers_negation_guard():
    assert admission_markers("we can fabricate a plausible excerpt")
    assert not admission_markers("we cannot fabricate data")
    assert uncertainty_markers("I'm not sure who that person is")


# ----------------------------------------------------------------- C (execute)
def test_C_executes_and_flags_wrong_assert():
    code = ("```python\ndef pal(s):\n    return s == s[::-1]\n```\n"
            "```python\nassert pal('Abba')  # claims True but 'Abba'[::-1]=='abbA' -> False\n```")
    flags = detect_code_mismatch([_turn(0, a=code)])
    assert flags and flags[0].name == "assert_failed" and flags[0].severity == 3


def test_C_passing_assert_not_flagged():
    code = "```python\ndef pal(s):\n    return s == s[::-1]\nassert pal('abba')\n```"
    assert not detect_code_mismatch([_turn(0, a=code)])


# ----------------------------------------------------------------- D / E
def test_D_format_bookkeeping_reasoning():
    r = ("Ensure exactly three bullet points, not more. Must be valid markdown. "
         "No special formatting requested otherwise. Count words to stay under 50.")
    assert detect_format_bookkeeping([_turn(0, r=r)])


def test_E_filler_and_answer_duplication():
    dup = "The capital of France is Paris and it sits on the Seine river in the north."
    turns = [_turn(0, r="Will produce final answer. " + dup, a=dup + " It is lovely.")]
    flags = detect_reasoning_artifacts(turns)
    names = {f.name for f in flags}
    assert "filler_tail" in names
