# `rxai_sdg.factory` — Synthetic Multi-Turn Conversation Generator (Data Factory)

A **stateless** synthetic-data generator that produces high-quality, multi-turn
conversational training data for stateful **Reactive Transformer (RxT / rc-RxT)**
models. It orchestrates two stateless LLMs — a **Responder/Teacher** and a
**User-Simulator** — in an alternating loop seeded from existing conversational
datasets, deliberately covering a typed taxonomy of follow-up intents weighted
toward the capabilities current models fail on (instruction-following under
transformation, constraint accumulation, instruction retention, long-range
recall).

> The Data Factory produces **text only**. It contains no RxT / STM / memory
> machinery — the training framework (RxLM) recomputes memory from the emitted
> conversations.

## Why a separate module?

Our models carry a fixed-size Short-Term Memory (STM) *between* interactions
instead of re-reading the whole history. Training data must therefore stress
*memory persistence across turns*, not just single-turn quality. The Data
Factory's centerpiece is a **cross-product taxonomy**: *what* a follow-up asks
for (`base_intent`) is orthogonal to *the memory distance* at which it operates
(`distance_policy`). Memory stress comes from the distance.

## Architecture

```
Seed Curator → [ Responder → Verifier → User-Simulator → Fact-Ledger/Needle-Planner ]* → Cross-turn checks (+ optional holistic judge) → Segment Writer
```

| Component | Module | Role |
|-----------|--------|------|
| `SeedCurator` | `seed_curator.py` | Load/dedup/tag seeds (rule-based), flag haystacks, load prompt packs |
| `Responder` | `responder.py` | Strong **memory-enabled** teacher; reasoning-mode answers; robust `<think>` parsing |
| `ConstraintVerifier` | `verifiers/` | Per-response, language-aware programmatic checkers |
| `UserSimulator` | `user_simulator.py` | Temporally-valid `(intent × policy)` draw + **grounded** follow-up + `constraint_spec` |
| `FactLedger` / `NeedlePlanner` | `ledger.py` | Plant / recall / update facts; schedule delayed recalls |
| `SegmentWriter` | `writer.py` | Typed segments + JSONL writing |
| `FactoryDatasetPostprocessor` | `dataset.py` | Build / append a HuggingFace dataset |
| `derive_variants` | `variants.py` | **Separate** reasoning→instruct/mixed post-processing |
| `ConversationLoop` | `loop.py` | Orchestrates the alternating loop with regen / resample / discard rules |
| `DataFactory` | `factory_runner.py` | High-level entry point |

## The taxonomy (spec §4)

* **Axis 1 — base intents** (`taxonomy.BASE_INTENTS`): `reformat`,
  `lexical_constraint`, `restyle`, `compress`, `expand`, `genre_convert`,
  `fact_recall`, `fact_update`, `chained_compute`, `self_critique`, `deepen`,
  `open_chat`.
* **Axis 2 — memory-distance policies** (`taxonomy.DISTANCE_POLICIES`):
  `immediate` (30), `cumulative` (25), `standing` (20), `delayed_recall` (25).
  The three memory-stressing policies sum to 70% by design; `immediate` at 30%
  preserves the depth-1 recall capability we already have.
* **Invalidity mask** (`taxonomy.default_invalid_pairs`) is encoded as data, not
  hardcoded `if`s. The sampler draws the two axes independently and resamples on
  an invalid pair.

## Typical workflow (one of many parallel sessions)

In practice you run 10-20 notebook sessions of this factory in parallel, each
generating conversations and appending them to a shared HuggingFace dataset:

```python
import random
from rxai_sdg.factory import DataFactory, FactoryConfig, OpenAILLMClient

# 1. initialise factory + clients (Responder and Simulator are different models)
cfg = FactoryConfig(seed=0)
responder = OpenAILLMClient(model_name="gpt-4", api_key="sk-...")        # strong teacher
simulator = OpenAILLMClient(model_name="gpt-4o-mini", api_key="sk-...")  # different model
factory = DataFactory(cfg, responder, simulator_client=simulator)        # simulator also
                                                                          # serves as the
                                                                          # category classifier

# 2. provide seeds as a list of strings or dicts with a 'query' field
seeds = ["Explain how entropy relates to information.",
         {"query": "Outline a function to reverse a linked list."}]

# 3. run the pipeline over all seeds -> ONE reasoning-mode record per conversation
records = factory.generate(seeds, band="generalization")

# 4. save to a HuggingFace dataset: append to an existing one or create a new one
factory.save_to_hub("org/rxt-factory", config_name="default", split="train",
                    token="hf_...", append=True)
```

The category/domain of each seed is **inferred** (a keyword heuristic, with an
optional LLM classifier fallback when the heuristic is inconclusive) — you do not
need to tag seeds.

### Deterministic, no-network example

```python
import random
from rxai_sdg.factory import DataFactory, FactoryConfig, MockLLMClient, validate_record
from rxai_sdg.factory.testing import constraint_satisfying_handler

factory = DataFactory(FactoryConfig(seed=0),
                      MockLLMClient(handler=constraint_satisfying_handler),
                      rng=random.Random(0))
records = factory.generate(["Explain how entropy relates to information."], band="basic")
for rec in records:
    validate_record(rec.to_dict())
factory.write_jsonl("out.jsonl")
```

### HuggingFace output

Records are written to a stable, append-safe row schema: scalar/seed fields are
native columns; the variable-keyed nested parts (`turns`, `fact_ledger`,
`cross_turn_checks`, `holistic_score`) are stored as JSON strings so that many
independent sessions can append to the same dataset without Arrow schema
conflicts. Use `FactoryDatasetPostprocessor` directly for more control, and
`record_to_row` / `row_to_record` to convert.

```python
from rxai_sdg.factory import FactoryDatasetPostprocessor
post = FactoryDatasetPostprocessor(records, dataset_id="org/rxt-factory", token="hf_...")
post.push_to_hf_hub(append=True)   # load existing + concatenate, or create new
```

## Derived training variants are a separate step (spec §8)

Generation emits exactly **one** reasoning-mode record per conversation. Deriving
the `instruct` (reasoning stripped) and `mixed` variants is a deterministic
post-processing step you run independently later, over an already-generated
dataset:

```python
from rxai_sdg.factory.variants import derive_variants
reasoning, instruct, mixed = derive_variants(record, ["reasoning", "instruct", "mixed"])
```

## CLI / batch runner

```bash
# Deterministic smoke test (no network)
python -m rxai_sdg.factory.cli --smoke -n 5 --band basic --out out.jsonl

# Real run from a JSONL seed file
python -m rxai_sdg.factory.cli --seeds seeds.jsonl --n 100 --config factory.json \
    --responder-model gpt-4 --simulator-model gpt-4o-mini --api-key $OPENAI_API_KEY \
    --out conversations.jsonl
```

## Reasoning → derived training examples (spec §8)

Every conversation is generated in **reasoning mode** and the factory emits one
such record per conversation. The derived `instruct` (all `<think>` stripped) and
`mixed` (a sampled subset kept) variants are produced later by the **separate**
`rxai_sdg.factory.variants.derive_variants` post-processing step. That step runs a
regex self-containment pass flagging answers with dangling references to removed
reasoning ("as computed above", "from step 2").

## Multilingual surface (spec §9)

`lang` is a first-class field on `Seed`, `constraint_spec`, and the verifier
registry key (default `"en"`). Universal checkers (`json_valid`,
`length_tokens`, `n_bullets`, …) are language-agnostic; language-specific
checkers (`first_letter`, `no_gendered_pronouns`, …) are keyed by `lang`. Only
English + universal checkers are implemented; other locales register explicit
`NotImplementedError` stubs via `register_language_stubs`. Constrained
transformation tasks are **never** blindly translated — multilingual generation
must be native.

## Coherence contracts (grounded by construction)

The generation core enforces hard, testable invariants so the data is coherent,
not merely structurally valid:

* **Memory-enabled teacher.** The Responder is prompted as an assistant with
  persistent memory of the whole conversation; memory-disclaimer answers ("I
  don't retain information between conversations") fail verification and are
  regenerated. The internal QA checklist is **never** put in the generation
  prompt.
* **Strict reasoning/answer segmentation.** `parse_response` splits on the
  `<think>…</think>` delimiters: exactly one well-formed block ⇒ `reasoning` +
  `answer` (`reasoning_flag=True`); otherwise the whole tag-stripped output is the
  `answer` (`reasoning_flag=False`, `malformed_outputs` incremented). A `</think>`
  tag is never left inside an `answer`.
* **Grounded, single-pass follow-ups.** The User-Simulator produces one grounded
  query (no second-pass `naturalize`): transformations target the prior answer,
  `deepen`/`self_critique`/`chained_compute` operate on its claims, `open_chat`
  continues the running topic. The machine-checkable `constraint_spec` is always
  built programmatically.
* **Fact lifecycle.** The `FactLedger` is the single source of truth: a planted
  value is asserted present in the emitted query and marked *injected*; a recall /
  update is only scheduled against an already-injected, sufficiently-distant fact.
* **Temporal-policy validity.** `cumulative` needs ≥1 prior active constraint;
  `standing` may bootstrap the stack on a follow-up; a fact `delayed_recall` needs
  an injected fact planted ≥ `min_recall_distance` turns earlier.

## Verification levels (spec §6)

1. **Per-response** (`ConversationLoop._verify_turn`) — a general quality gate
   (refusal/length/repetition), a **coherence gate** (no memory-disclaimer, no
   chain-of-thought leakage), and the machine-checkable constraint
   (`programmatic`/`hybrid`). `passed` therefore reflects coherence, not just
   literal constraint satisfaction.
2. **Cross-turn** (`run_cross_turn_checks`) — programmatic relational checks:
   delayed-recall fidelity, standing-instruction adherence, update-overwrite
   correctness.
3. **Optional holistic judge** (`HolisticJudge`) — **off by default**; gated/sampled;
   emits a structured rubric. Keep its model/prompt separate from the eval judge.

## Throughput & concurrency (spec §6)

`DataFactory.generate` runs conversations **concurrently** with a
`ThreadPoolExecutor` (`config.concurrency`, default 64). Conversations are
independent; the loop within a conversation stays sequential. Each conversation
gets its own `Random(config.seed + index)` so output is **reproducible**
regardless of thread scheduling (parallel `generate` produces the same records as
serial), and per-conversation stats are merged under a lock. The blocking client
is I/O-bound, so threads suffice.

> Throughput is bounded by the **saturated inference replica**, not the session
> count: one process at high `concurrency` should keep the endpoint busy. Run more
> sessions only to drive more replicas, not to speed up a single saturated one. A
> bounded **per-turn responder budget** (`max_responder_calls_per_turn`, default 8)
> caps the worst-case calls per turn across intent-resamples and regenerations.

## Configuration (spec §10)

`FactoryConfig` controls intent/policy weights, the invalidity mask, length bands
(`basic` 8–12, `generalization` 25–35, `short` 2–3), `lang`, regeneration limit
`K`, `max_responder_calls_per_turn` (per-turn budget), `concurrency`,
`min_recall_distance`, responder `max_tokens` / `temperature`. Premature features
are **off by default** and gated behind flags: `capture_logits`,
`enable_dmpo_pairs`, `holistic_judge_enabled`, `enable_low_yield_downweight`,
`seed_classifier_enabled` (rule-based seed tagging is the default — no per-seed LLM
call on the critical path). LLM clients are injected objects, not config. Load from
JSON or YAML via `FactoryConfig.from_file`.

## Tests

```bash
python -m pytest tests/factory -q
```

Covers: per-constraint pass/fail fixtures, sampler distribution vs. configured
weights, ledger/needle-planner logic, writer variant derivation + self-
containment, and an end-to-end integration batch (schema validity, ledger
correctness, reasoning-stripping self-consistency, intent/policy coverage).
