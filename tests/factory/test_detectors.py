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
    fabricated_citations, detect_constraint_corruption, detect_phantom_constraint,
    detect_value_drift, detect_count_violation, detect_scale_mismatch,
)
from rxai_sdg.factory.binary_gate import check_encoded_artifact_metadata, png_dimensions


def _turn(i, q="", r="", a="", constraint_spec=None):
    t = {"turn_index": i, "segments": [
        {"segment_type": "query", "text": q},
        {"segment_type": "reasoning", "text": r},
        {"segment_type": "answer", "text": a}]}
    if constraint_spec is not None:
        t["constraint_spec"] = constraint_spec
    return t


def test_phantom_constraint_rejects_ungrounded_genre():
    # user only ever asked for ELI5; spec claims genre="code comment" -> phantom.
    turns = [
        _turn(0, "What are the risks of AI-written code?", "", "..."),
        _turn(1, "From now on, always explain like I'm five.", "", "...",
              constraint_spec={"params": {"genre": "code comment"}, "scope": "standing",
                               "applies_from_turn": 1}),
    ]
    flags = detect_phantom_constraint(turns)
    assert flags and flags[0].name == "phantom_constraint"


def test_scale_mismatch_reasoning_vs_answer():
    bad = [_turn(0, "", "I'll use a scale from 1 to 5 where 1 is very bad.",
                 "Rate your mood on a scale from 0 to 10 each morning.")]
    f = detect_scale_mismatch(bad)
    assert f and f[0].name == "reasoning_answer_scale_mismatch"
    ok = [_turn(0, "", "I'll use a scale from 1 to 5.", "Use a scale from 1 to 5.")]
    assert detect_scale_mismatch(ok) == []


def test_encoded_artifact_png_dimension_mismatch():
    import base64
    import struct
    import zlib

    def make_png(w, h):
        def chunk(typ, data):
            c = typ + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        raw = b"".join(b"\x00" + b"\x00\x00\x00" * w for _ in range(h))
        return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

    b64 = base64.b64encode(make_png(64, 64)).decode()
    assert png_dimensions(make_png(64, 64)) == (64, 64)
    # the report's defect: text asserts 64x128 but the PNG decodes to 64x64
    bad = "Image size: (64, 128). Base64 output:\n" + b64[:200] + "..."
    assert check_encoded_artifact_metadata(bad)
    # matching dimensions, and no-base64 text, are both clean
    assert check_encoded_artifact_metadata("A 64x64 image. Base64: " + b64[:200]) == []
    assert check_encoded_artifact_metadata("The image is 64x128 pixels.") == []


def test_count_violation_exact_bullets_and_word_ceiling():
    bad = [_turn(0, "give me exactly five bullet points", "", "- a\n- b\n- c\n- d")]
    f = detect_count_violation(bad)
    assert f and f[0].name == "count_violation"
    over = [_turn(0, "summarize in under 10 words", "",
                  "This summary deliberately runs well past the requested ten word ceiling here.")]
    assert detect_count_violation(over)
    # honored constraints and ceiling-satisfying answers are clean
    assert detect_count_violation([_turn(0, "exactly three points", "", "- a\n- b\n- c")]) == []
    assert detect_count_violation([_turn(0, "in under 50 words", "", "Just six words here, fine.")]) == []


def test_value_drift_rejects_conflicting_facility_amount():
    turns = [
        _turn(0, "", "", "We set up a total credit facility of $150,000 to fund the work."),
        _turn(1, "", "", "Drawing $80,000 against the credit facility leaves room."),
        _turn(2, "", "", "The total credit facility of $200,000 gives you flexibility."),
    ]
    flags = detect_value_drift(turns)
    assert flags and flags[0].name == "cross_turn_value_drift"


def test_value_drift_clean_on_intended_change_and_lineitems():
    # an explicit change-verb licenses the new value -> not drift
    ok_change = [
        _turn(0, "", "", "The credit facility is $150,000."),
        _turn(1, "", "", "We later increased the facility to $200,000 for expansion."),
    ]
    assert detect_value_drift(ok_change) == []
    # different budget line items legitimately differ; only singular instruments
    # (facility/loan) are checked, so this must not fire.
    lineitems = [_turn(0, "", "", "Materials budget $150,000; labor budget $200,000.")]
    assert detect_value_drift(lineitems) == []


def test_phantom_constraint_grounded_is_clean():
    # the user DID ask for code comments -> grounded, no flag. And a normalised
    # style label ("ELI5 tone") the user expressed as "like I'm five" must NOT fire.
    turns = [
        _turn(0, "Please phrase replies as code comments from now on.", "", "...",
              constraint_spec={"params": {"genre": "code comment"}, "scope": "standing",
                               "applies_from_turn": 0}),
        _turn(1, "Also explain like I'm five.", "", "...",
              constraint_spec={"params": {"style": "a friendly ELI5 tone"},
                               "scope": "standing", "applies_from_turn": 1}),
    ]
    assert detect_phantom_constraint(turns) == []


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


# --------------------------------------------------------------- G (constraint corruption)
def test_G_constraint_corruption_latex_rejects():
    # first-letter-'S' spliced into every formula line garbles the matrix.
    corrupt = ("S The code has length four. S\\;G=\\begin{pmatrix} S\\;1&1&0&0\\\\ "
               "S\\;1&0&1&0\\\\ S\\;1&0&0&1\\\\ S\\;0&1&1&0 \\end{pmatrix}. S Done.")
    flags = detect_constraint_corruption([_turn(0, "q", "clean reasoning", corrupt)])
    assert flags and flags[0].severity == 3 and flags[0].name == "constraint_corruption"


def test_G_does_not_fire_on_clean_dense_latex():
    # Legitimate dense LaTeX uses \; and \, after symbols, but never one specific
    # letter glued repeatedly.
    clean = ("The garden algebra \\{Q_I,Q_J\\}=2\\delta_{IJ}\\,\\partial_t requires "
             "each \\(\\Gamma_I\\) to link bosons \\(\\phi_a\\;\\) to fermions. "
             "\\[ G=\\begin{pmatrix} 1&1&0&0\\\\ 1&0&1&0 \\end{pmatrix} \\]")
    assert not detect_constraint_corruption([_turn(0, "q", "r", clean)])


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


# --------------------------------------------------------------- A-disclaim (Tier 2)
def test_disclaimer_then_finding_rejects():
    from rxai_sdg.factory.detectors import detect_disclaimer_then_finding
    turns = [_turn(9, "estimate the burrow volume",
                   "We can't fabricate a specific documented case here, but we can approximate.",
                   "Field studies of Marmota monax burrows in the northeastern US, using "
                   "measuring tapes, found volumes of roughly 30-40 ft³.")]
    flags = detect_disclaimer_then_finding(turns)
    assert flags and flags[0].name == "disclaimer_then_finding" and flags[0].severity == 3


def test_disclaimer_then_finding_no_fp_on_recommendation_or_real_event():
    from rxai_sdg.factory.detectors import detect_disclaimer_then_finding
    # a RECOMMENDATION to run a study is not a claimed finding
    assert not detect_disclaimer_then_finding([_turn(0, "q",
        "We can't fabricate a documented study.",
        "You could run a small survey of purchasers, which found patterns vary by 10%.")])
    # a real, knowable documented event is fine
    assert not detect_disclaimer_then_finding([_turn(0, "q",
        "We cannot fabricate current info.",
        "Using the well-documented 2020-2021 transition as an example.")])


# --------------------------------------------------------------- H-safety (Tier 6)
def test_harmful_coping_pain_techniques_reject():
    from rxai_sdg.factory.detectors import detect_harmful_coping
    cases = [
        ("I keep having panic attacks, what helps?",
         "When a panic attack hits, snap the rubber band hard against your wrist — the "
         "sharp sting will pull you out of the spiral."),
        ("How do I stop intrusive thoughts?",
         "Hold an ice cube in your palm until it hurts; the pain interrupts the rumination."),
        ("I get overwhelmed and anxious a lot.",
         "When the anxiety spikes, pinch your arm hard to interrupt the feeling."),
    ]
    for q, a in cases:
        flags = detect_harmful_coping([_turn(3, q, "", a)])
        assert flags and flags[0].name == "harmful_coping_technique" and flags[0].severity == 3


def test_harmful_coping_no_fp_on_benign_grounding():
    from rxai_sdg.factory.detectors import detect_harmful_coping
    benign = [
        # sensory ANCHOR: focus on texture, not pain
        ("How can I ground myself during anxiety?",
         "Carry a small object — a smooth stone or a rubber band — and when the anxiety "
         "spikes, focus on its texture, temperature, and weight."),
        # a band worn as a visual REMINDER cue
        ("Any grounding tips?",
         "Some people keep a tiny reminder, like a rubber band on the wrist, that signals "
         "'time to ground.' When you see it, run your routine."),
        # evidence-based DBT cold reset with NO pain escalation
        ("DBT skills for distress?",
         "The TIPP skill: hold an ice cube in your hand or splash cold water on your face "
         "to reset your nervous system — no need to cause any pain."),
        # the answer explicitly DISCOURAGES the technique
        ("panic coping?",
         "When you feel a panic urge, don't snap a rubber band against your wrist — that's "
         "self-punishment; instead try paced breathing."),
        # unrelated business "shock"
        ("lumber budget?",
         "A six-month lumber-price shock raises material cost 35%, so pad the contract."),
    ]
    for q, a in benign:
        assert not detect_harmful_coping([_turn(0, q, "", a)]), a[:50]
