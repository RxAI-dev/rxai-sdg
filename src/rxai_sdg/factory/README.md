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
| `SeedCurator` | `seed_curator.py` | Load/dedup/tag seeds, flag haystacks, load prompt packs |
| `Responder` | `responder.py` | Strong teacher; reasoning-mode answers; optional logit capture |
| `ConstraintVerifier` | `verifiers/` | Per-response, language-aware programmatic checkers |
| `UserSimulator` | `user_simulator.py` | `(intent × policy)` sampler + structured `constraint_spec` emission |
| `FactLedger` / `NeedlePlanner` | `ledger.py` | Plant / recall / update facts; schedule delayed recalls |
| `SegmentWriter` | `writer.py` | Typed segments + derived reasoning/instruct/mixed variants |
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

## Quick start (deterministic, no network)

```python
import random
from rxai_sdg.factory import (
    DataFactory, FactoryConfig, MockLLMClient, DatasetSpec, validate_record,
)
from rxai_sdg.factory.testing import constraint_satisfying_handler

cfg = FactoryConfig(seed=0)
client = MockLLMClient(handler=constraint_satisfying_handler)
factory = DataFactory(cfg, client, rng=random.Random(0))

records = factory.generate(
    DatasetSpec(records=[{"query": "Explain how entropy relates to information.",
                          "category": "stem"}]),
    n_conversations=4, band="basic",
)
for rec in records:
    validate_record(rec.to_dict())
factory.write_jsonl(records, "out.jsonl")
```

## Real run (two separate models)

```python
from rxai_sdg.factory import DataFactory, FactoryConfig, OpenAILLMClient, DatasetSpec

cfg = FactoryConfig(seed=0)
responder = OpenAILLMClient(model_name="gpt-4", api_key="sk-...")          # strong teacher
simulator = OpenAILLMClient(model_name="gpt-4o-mini", api_key="sk-...")    # different model
factory = DataFactory(cfg, responder, simulator_client=simulator)
records = factory.generate(DatasetSpec(records=my_records), n_conversations=500,
                           band="generalization")
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

Every conversation is generated in **reasoning mode**; the writer then derives
multiple self-consistent training records by deterministic post-processing:
`reasoning` (all `<think>` kept), `instruct` (all stripped), `mixed` (a sampled
subset kept). A regex self-containment pass flags answers with dangling
references to removed reasoning ("as computed above", "from step 2").

## Multilingual surface (spec §9)

`lang` is a first-class field on `Seed`, `constraint_spec`, and the verifier
registry key (default `"en"`). Universal checkers (`json_valid`,
`length_tokens`, `n_bullets`, …) are language-agnostic; language-specific
checkers (`first_letter`, `no_gendered_pronouns`, …) are keyed by `lang`. Only
English + universal checkers are implemented; other locales register explicit
`NotImplementedError` stubs via `register_language_stubs`. Constrained
transformation tasks are **never** blindly translated — multilingual generation
must be native.

## Verification levels (spec §6)

1. **Per-response** (`ConstraintVerifier`) — fires only for machine-checkable
   `constraint_spec`s (`programmatic`/`hybrid`); a general quality gate
   (refusal/length/repetition) always applies.
2. **Cross-turn** (`run_cross_turn_checks`) — programmatic relational checks:
   delayed-recall fidelity, standing-instruction adherence, update-overwrite
   correctness.
3. **Optional holistic judge** (`HolisticJudge`) — off by default; gated/sampled;
   emits a structured rubric. Keep its model/prompt separate from the eval judge.

## Configuration (spec §10)

`FactoryConfig` controls intent/policy weights, the invalidity mask,
length bands (`basic` 8–12, `generalization` 25–35, `short` 2–3), domain mix,
haystack fraction, regeneration limit `K`, low-yield down-weighting, derived
variants + mixed ratio, `capture_logits`, and the holistic judge. Load from JSON
or YAML via `FactoryConfig.from_file`.

## Tests

```bash
python -m pytest tests/factory -q
```

Covers: per-constraint pass/fail fixtures, sampler distribution vs. configured
weights, ledger/needle-planner logic, writer variant derivation + self-
containment, and an end-to-end integration batch (schema validity, ledger
correctness, reasoning-stripping self-consistency, intent/policy coverage).
