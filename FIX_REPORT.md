# Data Factory generation-core repair — FIX REPORT

Scope: `src/rxai_sdg/factory/`. The generation core was rewritten to enforce hard,
testable **coherence contracts** while preserving the verifier/constraint suite,
schema, taxonomy and existing tests. The diagnosed root cause was a missing
coherence contract plus unspecified second-pass embellishment (`naturalize`).

---

## 1. Rewritten vs. preserved

### Rewritten
| File | What changed |
|------|--------------|
| `responder.py` | Robust `<think>…</think>` parser (`parse_response`), `malformed_outputs` flag, memory-disclaimer + CoT-leak detectors, returns `ResponderOutput`. Generation prompt no longer carries the internal QA checklist. |
| `prompts.py` | Responder reframed as a **memory-enabled** assistant that never disclaims memory; QA-checklist / self-containment notes removed; simulator prompt is grounding-only. |
| `user_simulator.py` | `naturalize` and the second LLM pass **deleted**. Single-pass **grounded** query generation; temporal-policy validity gate; fact plant→inject→recall/update lifecycle with verbatim-string assertion; structured `grounding` metadata. |
| `loop.py` | Thread-safe (`run(…, rng)` returns a per-conversation `LoopStats`); single **bounded per-turn responder budget**; **coherence gate** in `_verify_turn` (disclaimer + CoT leak); substantive thread carried via grounded queries; opportunistic non-sequitur fact plant + DMPO removed from the hot path. |
| `ledger.py` | Injection tracking (`mark_injected`/`is_injected`/`injected_facts`), `recallable_fact(require_injected=…)`, `updatable_fact(...)` — recall/update only ever target an injected fact (kills the "Meridian" desync). |
| `factory_runner.py` | `ThreadPoolExecutor` concurrency; per-conversation `Random(seed+index)`; lock-guarded stats merge; seed-ordered, reproducible output; rule-based seed tagging by default. |
| `config.py` | New knobs (`concurrency`, `max_responder_calls_per_turn`); premature features defaulted **off** behind flags (`capture_logits`, `enable_dmpo_pairs`, `holistic_judge_enabled`, `enable_low_yield_downweight`, `seed_classifier_enabled`). |
| `sampler.py` | `sample(lang, rng=…)` accepts a per-conversation RNG (thread-safe, reproducible); no behavioural change otherwise. |

### Preserved (untouched logic; tests only extended)
`verifiers/` (universal + english checkers), `constraints.py` (constraint-spec
building), `schemas.py`, `taxonomy.py` (intent/policy tables + invalidity mask),
`cross_turn.py`, `quality.py`, `writer.py`, `variants.py`, `dataset.py`,
`seed_curator.py`, `holistic.py`.

---

## 2. Contracts now enforced (maps to the 9 observed failures)

1. **Follow-ups grounded in real content.** The simulator receives the prior
   conversation and emits a query that transforms the prior answer, operates on
   its claims, or continues the running topic. Grounding is surfaced as structured
   metadata (`SimulatorResult.grounding`). *(fixes #1)*
2. **`naturalize` removed entirely.** One generation produces the grounded query;
   no second-pass rewrite can corrupt content or invent facts. *(fixes #2)*
3. **Memory-enabled teacher.** Prompt frames persistent conversational memory and
   forbids disclaimer phrasing; any disclaimer answer fails the coherence gate and
   is regenerated. *(fixes #3)*
4. **Strict reasoning/answer segmentation.** Exactly one well-formed block ⇒
   `reasoning`+`answer`, `reasoning_flag=True`; otherwise whole tag-stripped output
   is the `answer`, `reasoning_flag=False`, `malformed_outputs++`. `</think>` never
   survives in an `answer`. *(fixes #4)*
5. **QA checklist out of the prompt.** Self-containment / "no reference to
   reasoning" notes are post-checks (`variants.flag_dangling_references`), not
   generation instructions. *(fixes #5)*
6. **Ledger is the single source of truth.** A planted/updated value is asserted
   present in the emitted query, then the fact is marked *injected*; recalls/updates
   only target injected facts. No scheduled-but-uninjected facts. *(fixes #6)*
7. **Constraints transform substance, they aren't the substance.** Standing/
   cumulative constraints modify answers *about the seed topic*; each follow-up
   still operates on the prior answer / topic, so the conversation never collapses
   into a stack of bare format instructions. *(fixes #7)*
8. **Temporal-policy validity.** `cumulative` requires ≥1 prior active constraint;
   `standing` may bootstrap on a follow-up turn (it *is* the first such rule — a
   strictly-literal "standing needs a prior" reading is self-contradictory, so
   standing bootstraps and cumulative accumulates); a fact `delayed_recall` needs an
   injected fact planted ≥ D turns earlier. Invalid draws resample. No more
   `cumulative` at turn 1. *(fixes #8)*
9. **Coherence gate makes `passed` meaningful.** Beyond the constraint + quality
   gate, a turn fails if it disclaims memory or leaks chain-of-thought, so literal-
   but-incoherent turns are not marked `passed`. *(fixes #9)*

Plus: **multithreaded generation** (`ThreadPoolExecutor`, configurable
`concurrency`, default 64) with thread-safe per-conversation RNG/stats and a
**bounded per-turn responder budget** (default ≤ 8 calls, replacing the old
`max_intent_attempts × (K+1) ≈ 24` worst case).

---

## 3. Tests added (`tests/factory/`)

| File | Coverage |
|------|----------|
| `test_responder_parsing.py` | Parser contract (well-formed / malformed / empty-reasoning / stray-tag / multi-block); `</think>` never in any answer; disclaimer + CoT-leak detectors; malformed counter; disclaimer answer fails & regenerates. |
| `test_user_simulator_grounding.py` | Every query grounded (metadata); open_chat continues topic; transformation targets prior answer; `cumulative` never without prior active; `delayed_recall` fact requires distant **injected** fact; recall value matches ledger; planted value present & injected; recall only scheduled against injected fact; update value present & injected. |
| `test_loop_coherence.py` | Per-turn responder budget bounded (== cap in worst case); coherence gate fails disclaimer/CoT turns even when the literal constraint holds. |
| `test_concurrency.py` | Parallel `generate` == serial `generate` (record-identical given per-conversation seeds); seed-order preserved; stats totals identical across concurrency; high-concurrency records validate. |
| `test_sample_quality.py` | No CoT markers / `</think>` in any answer; no memory disclaimers; every follow-up grounded via structured metadata; substantive thread persists under standing constraints; constraints are not the whole conversation. |

All existing tests under `tests/factory/` are kept; the two responder tests in
`test_misc_components.py` were updated only for the new `ResponderOutput` return
type.

**Result:** `python -m pytest tests/factory -q` → **127 passed, 1 skipped** (the
skip is the optional `datasets`-backed Hub test).

---

## 4. Measured behaviour on a sample batch

Deterministic `MockLLMClient` (realistic `constraint_satisfying_handler`),
6 seeds × `generalization` band (25–35 turns), `concurrency=16`:

- conversations = 6, total turns = 192, follow-ups = 186
- **memory-disclaimer answers = 0**, **CoT / `</think>` in answers = 0**
- follow-up turns `passed` = **184/186 (98.9%)**
- malformed outputs = 0; coherence failures = 0
- all 4 distance policies and all 12 intents covered
- **no** `cumulative` at turn 1; **no** `cumulative`/`standing` without a prior
  active constraint; **no** fact `delayed_recall` against a too-close fact
- every planted fact's exact string appears in its plant turn
- cross-turn: `update_overwrite` 8/8, `delayed_recall` 3/4, `standing` 145/171

The lone `delayed_recall` miss is a *delayed fact update* turn whose value is later
overwritten again; `cross_turn.py` (preserved) re-checks against the final ledger
value, a conservative metric artifact — the turn itself recalls the correct value
and is marked `passed`, and every value did appear in the text (no desync).
Parallel and serial runs produce byte-identical records and identical stats.
