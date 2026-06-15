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

---

# PASS 2 — User-Simulator rewrite + responder reasoning capture

Two concentrated problems from the previous pass remained: the User-Simulator was
still effectively templated (it emitted canned query strings, was denied the
conversation history, capped queries at 192 tokens, and planted facts as
same-turn non-sequiturs), and responder `reasoning` segments were empty. This
pass rewrites the simulator to be genuinely LLM-driven and fixes the reasoning
capture path.

## 1. User-Simulator — deleted vs rewritten

### Deleted (the bug)
- **All hardcoded query strings in `constraints.py`.** `BuildResult.nl_query` is
  gone; every `build_*` now returns only the machine-checkable `constraint_spec`.
  `constraints.py` emits **no user-facing query text**.
- `UserSimulator._open_chat_query` and `_chained_compute_query` (the second
  hardcoded overwrite in `_finalize`).
- The `max_tokens=192` cap in the LLM phrasing call (it biased toward short
  queries). The simulator passes **no** `max_tokens`; the sampled **verbosity
  target** controls length.
- The deterministic fact templates ("By the way, my X is V. What is my X?") and
  the value-verbatim-by-construction approach. Facts are now woven in **by the
  LLM**.
- The old `_phrase_with_llm` design that passed **only `prior_turns[-1].answer`**.
- The whole `fact_recall + immediate` plant-and-recall-in-one-turn path — it
  tested no memory.

### Rewritten (`user_simulator.py`, query side of `constraints.py`)
The simulator is now a **genuine, LLM-driven user that sees the full transcript**.
Per turn it:
1. samples a temporally-valid `(intent, policy)` (`_temporally_valid` kept as-is);
2. builds the exact `constraint_spec` **programmatically** (params chosen in code:
   `first_letter='A'`, `format='json'`, `forbidden_token='important'`,
   `max_words=30`, …) — the spec stays machine-checkable;
3. samples a **persona** (curious / skeptical / frustrated / enthusiastic /
   terse-expert / casual, no immediate repeat) and a **verbosity** target
   (short ↔ long);
4. drives the instruct LLM with the **entire conversation** + a steer that, for
   verifiable constraints, instructs it to *explicitly and naturally request that
   exact constraint*; for `llm_judge` intents it writes a natural follow-up of
   that type grounded in the prior answer/topic;
5. runs a **post-generation coherence check** (verifiable: query mentions the
   format/letter/forbidden word/length; fact: value present for plant/update,
   absent for recall, fact named) and **regenerates once, then resamples** the
   intent (reusing the existing resample budget). There is **no second pass** and
   **no canned-string fallback** (the fallback is itself an LLM call).

The simulator **system prompt** now enforces the user role ("never answer your
own question, never ask the assistant to pose a question, never speak as the
assistant"), grounding in real content, intent realisation, and persona/length
diversity.

### Fact lifecycle (across turns, never same-turn)
- **Plant** (`fact_recall + immediate`): the LLM weaves the exact value into a
  natural, topical turn; the value is asserted present and the fact marked
  `injected`. The plant turn carries an `llm_judge` spec (it is *not* answer-gated
  — the assistant just acknowledges).
- **Recall** (`fact_recall + delayed_recall`, ≥ `min_recall_distance` later, a
  different turn): the LLM asks about the fact **without restating the value**;
  the recall only fires against an already-`injected` fact (no desync).
- **Update** (`fact_update`): the LLM states a new value; a subsequent recall
  returns the latest. `NeedlePlanner.update_value_for` now picks a value distinct
  from the **entire history** (not just the current value), so an "update" never
  coincides with a stale value (which the `fact_update` checker would flag).

### Fact diversity
The plant pool grew from 6 to 13 varied kinds (names, places, numbers,
preferences, dates, codenames, tiers, seats, …) and is sampled **without
immediate repetition** across a batch.

## 2. Responder reasoning capture (`responder.py`, `clients.py`)

Root cause: the OpenAI-compatible endpoint returns reasoning in a **separate
`message.reasoning_content` field**, but the text-only helper
(`BaseDatasetGenerator.generate_items`) returned `content` only and dropped it.

Fix — **both paths implemented, the field preferred**:
- `OpenAILLMClient.generate` now accesses the **raw** chat-completion. It sets
  `LLMResponse.reasoning = message.reasoning_content` when present; the Ollama
  path is left to inline parsing. It **logs the first raw response**
  (`reasoning_content present=…; inline_think=…`) so the live path can be
  verified against the endpoint.
- `Responder._segment_response` prefers the `reasoning_content` field (stripping
  any `<think>` block/tag from the answer); when absent it falls back to inline
  `<think>…</think>` parsing. `reasoning_flag = bool(reasoning.strip())`.
- New `LoopStats.reasoning_missing` counter (incremented when a reasoning-mode
  responder turn yields empty reasoning) makes endpoint misconfiguration visible.
- The simulator client stays on the **instruct** model — no reasoning captured
  for it.

## 3. Tests (`tests/factory/`)

| File | Coverage added/updated |
|------|------------------------|
| `test_user_simulator_grounding.py` | Rewritten for the LLM-driven contract (scripted Mock realising the STEER): full transcript reaches the client; **no** `max_tokens` cap; verifiable turns mention the exact constraint; plant/recall are **separate** turns and recall never restates the value; update states the new value; persona + length + query-length diversity; user role never inverted; **grep test** that no hardcoded query strings remain in `constraints.py` / `user_simulator.py`. |
| `test_responder_parsing.py` | `reasoning_content` field → non-empty `reasoning`, `reasoning_flag=True`, think stripped from answer; inline `<think>` fallback; `reasoning_missing` set when expected-but-absent; `reasoning_missing` increments in `LoopStats`. |
| `test_constraints.py` | `fact_recall + immediate` is a plant-only `llm_judge` spec; no `nl_query`. |
| `test_integration.py`, `test_sample_quality.py`, `test_concurrency.py`, `test_loop_coherence.py` | Now pass a dedicated simulator Mock (`simulator_user_turn_handler`). |

`python -m pytest tests -q` → **140 passed**.

## 4. Measured behaviour (deterministic Mock, 6 seeds × generalization band 25–35)

- conversations = 6, turns = 192; **reasoning segment present on 192/192 turns**;
  `reasoning_missing = 0`, `malformed = 0`.
- query length (words): min 9, p50 20, mean 29, max 66 — genuinely variable, no
  192-token cap.
- persona spread over 59 turns: curious 7 / skeptical 10 / frustrated 10 /
  enthusiastic 11 / terse-expert 9 / casual 12; verbosity short 17 / medium 18 /
  long 24.
- all 12 intents and all 4 distance policies covered.
- cross-turn: `update_overwrite` 12/12, `standing` 265/279, `delayed_recall` 6/10
  — the `delayed_recall` misses are the same preserved-`cross_turn.py` artifact
  (a recall that was correct at its turn, re-checked against a value the fact was
  later updated to); those recall turns themselves pass at turn level.
- reasoning-capture path: implemented field-first with inline fallback; the
  `OpenAILLMClient` one-shot raw log confirms which path fires on the live
  endpoint (offline here, so verified via unit tests for both branches).
