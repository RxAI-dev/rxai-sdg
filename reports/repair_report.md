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

## Validation on FRESH generation (gpt-oss-120b, real endpoint)
Re-generated the fabrication-prone real seeds with the strengthened pipeline. The
fix works end-to-end:
- **Responder now hedges instead of fabricating** — "I'm not finding any reliable,
  publicly-available record of a game-designer named Mark Seifter"; "we simply
  don't have a reliable single figure for the total number of rulers"; "I don't
  have confirmed data on who's in office now (June 2026)". Groundable hedged
  answers (president, kings/rulers, robot-vacuum height) PASS.
- **The gate catches residual fabrication** — the "37th-largest-city" ranking and
  the Seifter later-turn invention were hard-failed
  (`ungrounded_premise_fabrication`, `confidence_uncertainty_mismatch`).
- `factual_grounding` judge axis is live (batch mean 9.82); gate pass-rate 0.706
  (in band); 0 malformed; 0 API errors.
- Precision: validation surfaced and fixed one over-rejection — the broad "who is
  X?" premise wrongly flagged the well-hedged "who is the president" answer; the
  ungrounded-premise gate is now scoped to exact rankings only (bios are caught by
  the A1/A3 uncertainty/admission gates). President now ACCEPTs; Seifter + cities
  still REJECT; all 9 fixtures green.

## All-fixes batch (seeds50, 26 conversations) — ACCEPTANCE PASS
A full run with every fix (factuality gates + hedging prompt + F source-fix):
gate pass-rate **0.769** (in band), `factual_grounding` mean **10**, reasoning_quality
9.62, max regen 2, 0 discarded, 0 code-mismatch. Detector prevalence collapsed vs
the original corpus: **F 81% → 0%** (all 23 standing constraints now licensed),
**D 27% → 8%**, A/C caught by the gate. E (filler "Proceed." tails / answer-
duplication) remains a ~50% **soft** penalty (severity 1, no reject) — discouraged
by the prompt and reflected in reasoning_quality, not blocking.

## 100-seed full verification (concurrency=32, all fixes)
A real reasoning-dataset slice of 100 prompts (fabrication-prone, sensitive/refusal,
code, math, low-content). 92 conversations emitted; **0 malformed, 0
reasoning_missing, 0 API errors** at concurrency 32; gate pass-rate **0.739** (in
band); `factual_grounding` mean **9.9**. Detector prevalence: A 3%, C 0%, D 10%,
**F 2%** (was 81%), E 45% (soft). Read across categories confirmed:
- Fabrication caught: 37th-largest-city ranking; a Bugs-Bunny "interview" that
  invented a minute-by-minute cartoon timeline with SFX timings (reasoning admits
  "don't have exact source").
- Grounded facts PASS (no over-rejection): Sun parameters cite the real IAU 2015
  Resolution B3; tallest skyscraper = Burj Khalifa 828 m; the Cain-and-Abel Jungian
  analysis cites real verses (Genesis 4:17, 32:24).
- Hedging on ungroundable: president, number-of-rulers, MSCI/Nordea live metrics
  ("figures up to end-2024, may have moved since").
- Sensitive handled: explicit-story and AI-doctor-roleplay seeds were declined /
  filtered (not emitted); the bomb/slur dilemma got a measured ethical discussion.
- Math correct: age-puzzle = 21; Alice probability = 3/28.

**New issue found & fixed during verification:** the `talk_timestamp` specific
(`\d{1,2}:\d{2}`) matched scripture references (Genesis 4:17), wrongly rejecting the
grounded Jungian analysis. Scoped it to real talk/video context and added a
`rank_assertion` specific so the cities anchor still rejects. All 9 fixtures green.

## F (phantom standing) — FIXED AT SOURCE
Root cause: when the sampler draws `policy=standing`, the constraint is scoped
standing but the simulator phrased the request as a one-shot ("reformat this as
JSON"), so later unrelated turns were force-formatted. Fixed in the user-simulator:
a standing-policy transformation now instructs the simulator to phrase the request
as an explicit PERSISTENT instruction ("from now on, always …"), so the query
licenses the standing scope and the data is internally consistent. The standing
scope itself (a deliberate memory-test feature) is unchanged.

## Residual / not yet done (honest)
- **B/D/E reasoning/answer regeneration was not run** as a bulk pass — they are
  penalties, not hard rejects, and the strengthened gates + hedging prompt prevent
  them on new generation, so the right move is to **regenerate the corpus with the
  fixed factory** rather than patch the old file. **No bar was lowered.**
- **The 2 A-conversations (Seifter, cities) are fabricated-premise and were
  dropped** — they cannot be patched into grounded content honestly.
