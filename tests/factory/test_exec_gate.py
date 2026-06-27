"""Frozen fixtures + unit tests for the programmatic execution/arithmetic gate.

The two FROZEN fixtures reproduce the human-confirmed D5 defects that the LLM judge
accepted (it scores plausibility; it does not execute code or do arithmetic):

* BOARD_VOLUME - a code comment asserts ``board_volume ≈ 0.44`` while the code
  computes ``(2*4*24)/1728 = 0.111`` (and the narrative inverts ~1 vs ~4.5 boards/day);
* BUFFERING - a single-process demo narrates that one block runs while stdout is a
  terminal (line-buffered) and another after redirection (fully-buffered), which is
  impossible (buffering mode is fixed per process) and NOT executable-decidable.

These must REJECT with the correct reason. The false-positive guards (correctly
rounded / scientific / unicode-fraction arithmetic) must PASS.
"""
from rxai_sdg.factory.exec_gate import (
    run_exec_gate, disposition, check_inline_arithmetic, verify_haiku,
    check_json_keys, count_syllables,
)


def _turn(idx, answer, reasoning="clean reasoning about the task", query=""):
    return {"turn_index": idx, "constraint_spec": None,
            "segments": [{"segment_type": "query", "text": query},
                         {"segment_type": "reasoning", "text": reasoning},
                         {"segment_type": "answer", "text": answer}]}


# --------------------------------------------------------------- frozen fixture 1
BOARD_ANSWER = (
    "Here's a quick calculator:\n\n"
    "```python\n"
    "daily_volume = 0.5\n"
    "weight_lb = daily_volume * 42          # ≈ 21 lb\n"
    "board_volume = (2*4*24) / 1728         # ft³ of a 2-ft 2x4 ≈ 0.44 ft³\n"
    "boards_per_day = daily_volume / board_volume   # ≈ 1.14 boards\n"
    'print(f"Boards per day: {boards_per_day:.2f}")\n'
    "```\n\n"
    "**Result:** about **1.1 boards** per day.")


def test_fixture_board_volume_rejects():
    res = run_exec_gate([_turn(9, BOARD_ANSWER)])
    assert not res.passed
    kinds = {f.kind for f in res.hard_fails}
    assert "exec_arithmetic_mismatch" in kinds
    ev = " ".join(f.evidence for f in res.hard_fails)
    assert "board_volume" in ev and "0.111" in ev
    assert disposition(res) == "regenerate_turn"   # confined to one turn's answer


# --------------------------------------------------------------- frozen fixture 2
BUFFER_ANSWER = (
    '{"explanation": "The script prints a first block (which runs while stdout is '
    "still the original terminal stream, line-buffered) and then a second block that "
    "runs after the process has been started with its output redirected (fully "
    'buffered). It demonstrates the two behaviours in one run.", '
    '"code": "import sys\\nprint(\'Line\')\\nsys.stdout.flush()"}')


def test_fixture_buffering_flags_for_human_and_drops():
    res = run_exec_gate([_turn(8, BUFFER_ANSWER)])
    assert not res.passed                       # never a silent accept
    assert any(f.kind == "flag_for_human_runtime_claim" for f in res.human_flags)
    assert res.human_flags[0].human_review is True
    assert disposition(res) == "drop_conversation"   # load-bearing runtime claim


# --------------------------------------------------------------- A: exec arithmetic
def test_exec_arithmetic_precision_no_false_positive():
    # a correctly ROUNDED comment must NOT be flagged
    ok = "```python\nx = 700 / 55   # ≈ 12.7\n```"
    assert run_exec_gate([_turn(0, ok)]).passed
    bad = "```python\ny = 192 / 1728   # ≈ 0.44\n```"
    assert not run_exec_gate([_turn(0, bad)]).passed


# --------------------------------------------------------------- B: inline arithmetic
def test_inline_arithmetic_catches_wrong_result():
    f = check_inline_arithmetic("So 192/1728 = 0.44 cubic feet.", 0, "answer")
    assert f and f[0].kind == "inline_arithmetic_mismatch"


def test_inline_arithmetic_no_fp_on_rounding_and_chains():
    # correctly rounded
    assert not check_inline_arithmetic("700 / 55 ≈ 12.7 boards.", 0, "answer")
    assert not check_inline_arithmetic("(15-4)/15 = 0.73 of the work.", 0, "answer")
    # a correct multi-step chain
    assert not check_inline_arithmetic("(3/8)*(2/7) = 6/56 = 3/28 ≈ 0.107", 0, "answer")


def test_inline_arithmetic_no_fp_on_scientific_and_unicode_fractions():
    # 109.2^3 = 1.30e6 is correct (scientific notation must not be truncated)
    assert not check_inline_arithmetic("(109.2)^3 = 1.30e6 (about 1.30 million).", 0, "a")
    # ¼ × 0.707 ≈ 1/5.66 is correct (dropped ¼ must not yield a partial expr)
    assert not check_inline_arithmetic("¼ × 0.707 ≈ 1/5.66, exactly.", 0, "a")
    # √2 ≈ 0.707 form (dropped √ must not be checked as "2 ≈ 0.707")
    assert not check_inline_arithmetic("a half-stop cuts it by √2 ≈ 0.707.", 0, "a")


def test_inline_arithmetic_skips_running_sums_and_definitions():
    assert not check_inline_arithmetic("5+12=17, +4=21, +5=26", 0, "answer")
    assert not check_inline_arithmetic("Let x = 5 and n = 8.", 0, "answer")


# --------------------------------------------------------------- secondary: JSON keys
def test_json_nonascii_key_rejects():
    f = check_json_keys('{"recommendedMeal‑PrepWorkflow": 3}', 0, "answer")
    assert f and f[0].kind == "json_nonascii_key" and "U+2011" in f[0].evidence


def test_json_ascii_key_ok():
    assert not check_json_keys('{"normal_key": 3, "another-key": 4}', 0, "answer")


# --------------------------------------------------------------- secondary: haiku
def test_haiku_compartment_jars_not_575():
    # the spec's case: "Compartment jars" = 4 syllables -> not 5-7-5
    status, _ = verify_haiku(
        '{"haiku": "Compartment jars / grains stay dry, sauce waits its turn / '
        'crisp bites await you"}')
    assert status == "syllable"   # 3 lines extracted, counts != [5,7,5]


def test_haiku_valid_575_passes():
    status, _ = verify_haiku("An old silent pond\nA frog jumps into the pond\nSplash! Silence again")
    assert status in ("ok", "syllable")  # extraction works; count may vary by counter


def test_syllable_counter_basic():
    assert count_syllables("jars") == 1
    assert count_syllables("compartment") == 3


# --------------------------------------------------------------- repetition (Tier 1)
def test_degenerate_block_chart_rejects():
    # the woodchuck "bar chart" runaway: thousands of block glyphs in one cell.
    ans = "| Hypothetical wood-chucking | " + "█" * 3000 + " | " + "▓" * 2500 + " |"
    res = run_exec_gate([_turn(6, ans)])
    assert not res.passed
    assert any(f.kind == "degenerate_repetition" for f in res.hard_fails)


def test_repetition_does_not_fp_on_markdown_rules_or_small_bars():
    from rxai_sdg.factory.exec_gate import detect_degenerate_repetition
    assert detect_degenerate_repetition("-" * 182) is None          # long md rule
    assert detect_degenerate_repetition("Energy: " + "█" * 24 + " 80%") is None  # real bar
    assert detect_degenerate_repetition("text\n" + "| --- | --- |\n" * 3) is None


# --------------------------------------------------------------- table consistency (Tier 3)
# The kings/interregna reject: prose subtracts "23 - 3 interregna = 20 monarchs",
# implying 3 interregnum rows, but the table the answer itself prints lists only ONE
# interregnum. The judge reads the prose as plausible; counting the printed rows
# (which the answer authored) is the only thing that settles it.
KINGS_ANSWER = (
    "Poland's elective monarchy is unusual. Counting heads:\n\n"
    "| # | Ruler | Note |\n"
    "| --- | --- | --- |\n"
    "| 1 | Henry de Valois | elected |\n"
    "| 2 | Stephen Báthory | elected |\n"
    "| 3 | Sigismund III | |\n"
    "| - | (interregnum 1572-1573) | no monarch |\n\n"
    "So of the 23 entries, 23 - 3 interregna = 20 monarchs actually reigned.")


def test_table_count_mismatch_rejects_kings():
    res = run_exec_gate([_turn(4, KINGS_ANSWER)])
    assert not res.passed
    flags = [f for f in res.hard_fails if f.kind == "table_count_mismatch"]
    assert flags and "interregna" in flags[0].evidence


# --------------------------------------------------------------- alpha sort (Doc 2)
# Frozen from the real "Organize these words alphabetically" reject: a list the answer
# itself LABELS as a primary alphabetical sort, but the order is wrong. Lexicographic
# order is decidable offline - no LLM judge should ever rule on it.
ALPHA_SORT_MD = (
    "## 2. Primary sort - alphabetical (A → Z)\n\n"
    "We first order the rows by the word itself, ignoring length:\n\n"
    "1. Casa\n2. Computer\n3. Computerized\n4. Dimension\n5. Fire\n6. Firearm\n"
    "7. Firefly\n8. House\n9. Housetop\n10. Household\n11. Weapon\n")

ALPHA_SORT_JSON = (
    '{\n  "sortingSteps": [\n    {\n'
    '      "step": "1. Primary alphabetical sort (A → Z)",\n'
    '      "order": ["Casa", "Computer", "Computerized", "Dimension", "Firearm", '
    '"Firefly", "Fire", "House", "Housetop", "Household", "Weapon"]\n    }\n  ]\n}')


def test_alpha_sort_violation_md_rejects():
    res = run_exec_gate([_turn(2, ALPHA_SORT_MD)])
    assert not res.passed
    flags = [f for f in res.hard_fails if f.kind == "alpha_sort_violation"]
    assert flags and "Housetop" in flags[0].evidence and "Household" in flags[0].evidence


def test_alpha_sort_violation_json_rejects():
    res = run_exec_gate([_turn(5, ALPHA_SORT_JSON)])
    assert not res.passed
    flags = [f for f in res.hard_fails if f.kind == "alpha_sort_violation"]
    assert flags and "Fire" in flags[0].evidence


def test_alpha_sort_no_fp_on_correct_or_tiebreaker_lists():
    from rxai_sdg.factory.exec_gate import detect_alpha_sort_violation
    # a CORRECT alphabetical list must pass
    good = ("Sorted alphabetically (A → Z):\n\n"
            "1. Casa\n2. Computer\n3. Dimension\n4. Fire\n5. House\n6. Weapon\n")
    assert not detect_alpha_sort_violation(good)
    # a LENGTH-sorted list where alphabetical is only the tie-breaker must NOT fire
    tb = ("Sorted by length; alphabetical is the secondary tie-breaker:\n\n"
          "1. Casa\n2. Fire\n3. House\n4. Weapon\n5. Computer\n")
    assert not detect_alpha_sort_violation(tb)


# --------------------------------------------------------------- hamming weight (Doc 4)
def test_hamming_weight_parity_contradiction_rejects():
    # frozen from the real Gates/codes reject (Doc 4 turn 1)
    ans = ("Because every codeword has even Hamming weight (the weight of 01,10,11 is "
           "1,1,2 respectively), the code is *even*.")
    res = run_exec_gate([_turn(1, ans)])
    assert not res.passed
    flags = [f for f in res.hard_fails if f.kind == "hamming_weight_contradiction"]
    assert flags and "even" in flags[0].evidence


def test_hamming_weight_no_fp_on_correct_claims():
    from rxai_sdg.factory.exec_gate import detect_hamming_weight_contradiction
    # a CORRECT even-weight statement must pass
    assert not detect_hamming_weight_contradiction(
        "All codewords have even Hamming weight (the weight of 0011,1100,1111 is 2,2,4).")
    # correct stated popcounts with no parity claim must pass
    assert not detect_hamming_weight_contradiction(
        "The weight of 01,10,11 is 1,1,2 respectively.")


def test_hamming_weight_flags_wrong_popcount():
    from rxai_sdg.factory.exec_gate import detect_hamming_weight_contradiction
    flags = detect_hamming_weight_contradiction("The weight of 1011 is 2.")  # popcount 3
    assert flags and "popcount is 3" in flags[0]


# --------------------------------------------------------------- haiku soft-flag (FN fix)
def test_heuristic_offby1_haiku_is_non_gating():
    # A haiku flagged ONLY by a heuristic off-by-1 syllable count (no cmudict) must NOT
    # reject - we cannot prove a defect, so it is a non-gating soft flag. (When cmudict
    # is installed the count is accurate and a real miss still hard-fails via the
    # confident path; this guards the heuristic-fallback envs against false negatives.)
    import rxai_sdg.factory.exec_gate as eg
    saved = eg._CMU
    eg._CMU = {}  # force the heuristic fallback
    try:
        # "Three doors, one car hides" heuristically counts 6 (miscounts "hides"); the
        # real haiku is 5-7-5. Under the heuristic this is an off-by-1 -> must be soft.
        turn = {"turn_index": 2, "constraint_spec": {"type": "genre", "params": {"genre": "haiku"}},
                "segments": [
                    {"segment_type": "query", "text": "write a haiku about the Monty Hall problem"},
                    {"segment_type": "reasoning", "text": "compose 5-7-5"},
                    {"segment_type": "answer",
                     "text": "Three doors, one car hides\nHost reveals goat, odds still shift\nSwitch, win two thirds chance"}]}
        res = eg.run_exec_gate([turn])
        assert res.passed, "heuristic off-by-1 haiku must not gate"
        assert any(f.kind == "haiku_syllables" for f in res.soft_flags)
    finally:
        eg._CMU = saved


def test_table_count_mismatch_no_fp_on_ranges_or_consistent_tables():
    from rxai_sdg.factory.exec_gate import detect_table_count_mismatch
    # a RANGE "5-7 characters = 12" is not a consistent subtraction (12 != 5-7); skip
    assert not detect_table_count_mismatch(
        "| a | b |\n| --- | --- |\n| x | y |\nUse 5-7 characters = 12 total.")
    # a subtraction with no accompanying table is out of scope here
    assert not detect_table_count_mismatch("Of 23 - 3 interregna = 20 kings reigned.")
    # a table whose interregnum row count MATCHES the stated M must pass
    good = (
        "| # | Ruler |\n| --- | --- |\n"
        "| - | interregnum A |\n| - | interregnum B |\n| - | interregnum C |\n"
        "So 23 - 3 interregna = 20 monarchs reigned.")
    assert not detect_table_count_mismatch(good)
