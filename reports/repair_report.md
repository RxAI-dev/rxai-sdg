# Dataset repair report — factual-fabrication loop

**Input:** `data/_immutable/original.parquet` (read-only backup of the supplied
dataset; 62 conversations that the *previous* judge+gate ACCEPTED — the 62/100
pass set). **Output:** `data/repaired/filtered.parquet` (NEW path; original never
mutated). Detectors + prevalence: `tools/run_detectors.py`,
`reports/prevalence_pre.json`.

## What was broken
The judge had **no factuality / grounding axis** and was blind to an entire defect
class: **factual fabrication and confidence–uncertainty mismatch**. The evidence is
in the reasoning trace itself — the model writes *"I'm not sure who…"*, *"we can
reference…"*, *"constructed illustration"* and the answer then asserts named
studios, games, Metacritic scores, Kickstarter sums, GitHub URLs, exact city
rankings, poll numbers and attendance figures with full confidence. Because the
generation pipeline has **no retrieval**, any such checkable specific is fabricated
by construction.

## Prevalence (full dataset, 62 conversations)
| det | defect class | convs | % | gate |
|---|---|---|---|---|
| **A** | confidence–uncertainty mismatch / ungrounded-premise fabrication | **2** | 3.2% | **HARD reject** |
| B | fabricated checkable specifics (URLs, scores, funding, attendance…) | 27 | 43.5% | penalty (hard only with A) |
| **C** | code whose own `assert` is provably wrong (executed) | **1** | 1.6% | **HARD reject** |
| D | format-bookkeeping-dominated reasoning | 17 | 27.4% | penalty |
| E | reasoning artifacts (filler tails, answer duplicated in reasoning) | 26 | 41.9% | penalty |
| F | phantom standing constraints (one-shot request labeled `scope=standing`) | 50 | 80.6% | penalty / relabel |

The 5 manually-reviewed examples were **representative, not outliers** — A fired on
exactly the two seed REJECTs (Seifter bio, cities rankings); C reproduced the
Unicode-palindrome bug by *executing* it (`assert is_palindrome("ÅbbaÅ")` →
`AssertionError`). F is pervasive: 96/100 standing constraints have no licensing
phrase ("from now on" / "always") in the query that introduced them.

## What was changed (additive only — Guardrails 1 & 2 honored)
1. **`detectors.py`** — A–F deterministic detectors, grounded in and verified on
   the real traces (negation-guarded so *refusing* to fabricate is not flagged;
   bare years excluded; A requires same-turn pairing or an ungrounded premise or an
   explicit fabrication admission, so it does not fire on grounded factual content).
2. **Pre-filter hard gates** — `deterministic_prefilter` now hard-fails A
   (`confidence_uncertainty_mismatch`, `ungrounded_premise_fabrication`,
   `fabrication_admission`) and C (`code_assert_failed`). These were the defects
   the LLM judge could not catch.
3. **LLM-judge backstop** — added a `factual_grounding` rubric axis (gate ≥ 7) for
   subtler mismatches the deterministic gate misses. No existing threshold lowered.
4. **Frozen fixtures** — added `gt6_fabrication_bio` (Seifter) and
   `gt7_fabrication_stats` (cities) as REJECT anchors and `gt_clean_factual`
   (conditioning explainer) as an ACCEPT anchor so the new gates cannot become
   over-aggressive on grounded content.
5. **Source fix (Responder prompt)** — the teacher must match answer confidence to
   reasoning confidence; never invent URLs/citations/scores/dates/stats/funding;
   hedge or decline on ungroundable premises (obscure bios, exact rankings); never
   pre-write the answer in the reasoning.

## Per-class action (this pass)
| class | convs | action taken | why |
|---|---|---|---|
| A | 2 | **DROPPED** | fabricated premise — cannot be patched honestly (Guardrail 4) |
| C | 1 | **DROPPED** | shipped a provably-wrong `assert`; needs regenerate+re-exec |
| B | 25 (in kept set) | flagged for answer-regeneration | drop ungrounded specifics / reseed if premise ungrounded |
| D/E | 17 / 25 | flagged for reasoning-regeneration | genuine cognition, no filler/duplication |
| F | 47 | flagged for `constraint_spec` relabel | scope=standing → one-shot unless query licenses standing |

## Yield
- **Old judge+gate:** 62/100 accepted (the supplied file).
- **Strengthened gate (A+C added):** **59/62 kept**, 3 rejected
  (`fabrication_admission`, `ungrounded_premise_fabrication`, `code_assert_failed`).
- The drop is **desirable** (fabrication/bug removal), not over-rejection: A fired
  on 0 grounded-factual conversations; the clean-factual ACCEPT anchor and the
  43%/B-flagged-but-grounded conversations (single legitimate URL/score) are kept.

## Fixture anti-regression — GREEN
All 7 reject anchors (5 original + Seifter + cities) reject; both accept anchors
(vaccine control + conditioning) pass. Verified after every change; no change that
broke a fixture was kept.

## Residual / not yet fixed (honest)
- **B/D/E/F are not yet regenerated at source.** They are penalties, not hard
  rejects, and regenerating reasoning/answers requires live generation. The OVH
  endpoint was intermittently unavailable during this work, so the Step-4
  regenerate/reseed loop was **not** run; the strengthened gates + prompt mean
  these defects are caught/avoided on the *next* generation. **No bar was lowered to
  make them disappear.**
- **F (phantom standing) is a generation-side bug**, not just data: the factory
  assigns `scope=standing` to one-shot transformation requests. The real fix is in
  the constraint/scope assignment (user-simulator/constraints), recommended next.
- **Residual unfixable in this pass:** the 2 A-conversations (Seifter, cities) are
  fabricated-premise and were dropped — they cannot be patched into grounded
  content; honest options are reseed with a groundable prompt or drop.
