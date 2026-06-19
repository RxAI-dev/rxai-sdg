# Holistic-judge fix + autonomous generate→judge→fix loop

Scope: `src/rxai_sdg/factory/` (judge, pre-filter, gate, generation prompts,
sanitization) + `tools/` (regression fixtures, Phase-2 validation, the autonomous
loop harness) + `audit/` (the per-iteration, human-legible trail). Everything was
validated against the **real OVH endpoint**; no mocks gate "done".

Models (read from the environment, never committed):
`RESPONDER=Qwen3.5-397B-A17B`, `SIMULATOR=Qwen3-Coder-30B-A3B-Instruct`,
`CURATOR=Mistral-Small-3.2-24B-Instruct-2506`, **`JUDGE=Mistral-Small-3.2-24B-Instruct-2506`**.

> The judge was deliberately set to **Mistral-Small**, a different model family
> from BOTH the responder (Qwen3.5) and the simulator (Qwen3-Coder). The simulator
> writes the user queries the judge now grades, so using the Qwen judge would be
> self-evaluation (bias toward its own queries). See §5 for the empirical bias check.

---

## 0. Confirmed root cause (from the task; verified on the real endpoint)

`HolisticJudge.score()` built its prompt from `format_transcript(turns)`, which
renders only `query` + `answer` and **drops `reasoning`**. The judge had therefore
never seen a reasoning trace — which is why defective conversations scored 8–10. The
gate `_holistic_ok` only checked two hard-coded fields (`coherence`,
`appropriateness`).

Two facts confirmed by probing `Qwen3.5-397B-A17B` directly:
1. It returns its chain of thought in a **separate `reasoning` field** (not inline).
2. It **always** opens that reasoning with the literal scaffold
   `"Thinking Process:\n\n1. **Analyze the Request:** ..."` and **echoes any
   meta-instruction back into the reasoning** — so telling it "do not write
   'Thinking Process'" makes it write *"…No header like 'Thinking Process:'…"* in
   the reasoning. Negative prompt instructions therefore backfire; the scaffold must
   be removed by a deterministic sanitization pass, not by prompting.

---

## 1. PHASE 1 — the judge now sees reasoning + detects A–G (+ user-query quality)

**`format_transcript_for_judge(turns)`** (new, in `responder.py`; `format_transcript`
is untouched — every other caller keeps user+assistant-only, no turn numbers). It
renders each turn segment-delimited and labeled so the judge can cite a turn:

```
[Turn 0]
User: <query>
Reasoning: <reasoning or "(none)">
Assistant: <answer>
```

**Rewritten `_JUDGE_SYSTEM`** — grades TRAINING DATA for a stateful memory model
where the reasoning is a first-class, unmasked target and the model has no full
context / no turn numbers / no harness at inference. Rubric (all 1–10, anchored in
the prompt), `temperature=0.0`, strict JSON:

| kept (back-compat) | added |
|---|---|
| instruction_following, coherence, naturalness, role_consistency, recall_fidelity, appropriateness | **reasoning_quality** (A/D/E), **reasoning_answer_consistency** (F), **sycophancy_resistance** (G) |

Per the follow-up clarification, the **LLM-generated USER turns are also graded**
(they are not assumed good): **`user_query_quality`** (gated — garbled / erroneous /
self-answering queries reject) and **`user_query_difficulty`** (detection-only — easy
exploration turns are legitimate by design). The judge also returns
`flagged_turns`: `{turn_index, dimension, severity 1–3, evidence}`.

**Deterministic pre-filter** (`holistic.deterministic_prefilter`, runs BEFORE the
LLM judge — these defects are objective):
- `\bTurn\s+\d+\b` / `reference_turn_N` / cue+`turn N` in any **answer** → **hard fail** (B; FATAL, corrupts the target). Lowercase `turn N` needs a positional cue so physical "turn 90 degrees" is not flagged.
- any harness phrase (A) in any **reasoning** → **hard fail** (persistent memory, "drawing on the whole conversation above", "write only the final answer", "never deny having memory", "make each inferential step explicit and checkable", a `Thinking Process:` header, agonizing about system instructions).
- trailing artifact on any segment (`(?:\bcw|\bcltr)\s*$` and a glued `word.xx` form, with a file-extension/TLD allow-list) → **hard fail** (C).
- `verification.regenerations` > threshold (default 2) → **soft flag**.
- degenerate-loop reasoning (>40 % duplicated line/sentence units) → **soft flag** (D).

**Config-driven gate** (`ConversationLoop._holistic_ok`, replacing the two-field
check):

```python
holistic_gate = {"coherence": 7, "appropriateness": 7, "reasoning_quality": 7,
                 "reasoning_answer_consistency": 7, "sycophancy_resistance": 7,
                 "instruction_following": 6, "user_query_quality": 6}
hard_fail_on_flagged_severity = 3
prefilter_regen_threshold = 2
```

Reject if the pre-filter hard-failed, **or** any *present* rubric field is below its
min, **or** any `flagged_turns` entry ≥ the severity cutoff. "No rubric → do not gate
on the judge" and the `should_score`/sampling logic are preserved.

## 2. PHASE 3 — model-independent generation fixes (applied up front)

- **A/B (harness & turn-index):** the responder system prompt and context were
  stripped of every harness phrase the native-reasoning model was echoing
  (`_RESPONDER_BASE` rewritten; the `"…drawing on the whole conversation above.
  Write only the final answer."` trailer removed; the `reasoning` category flavor
  "Make each inferential step explicit and checkable" reworded). Behaviour is steered
  positively (recall earlier content, apply constraints, hold a justified position).
  The transcript fed to the generator has **no turn numbers**; the simulator is told
  never to refer to a turn by index.
- **C (artifact) + residual A:** a deterministic **sanitization pass**
  (`sanitize_reasoning` / `sanitize_generated_text`) strips the leading
  `Thinking Process:` scaffold and glued trailing artifacts (`.cw`/`.cltr`) from
  generated reasoning/answers and from the simulator's user query. This is the hard
  guard; the prompt rewrite is the primary fix (the scaffold cannot be prompted away).

## 3. PHASE 2 — frozen-judge validation (real endpoint)

Five known-bad fixtures covering A–G + a sixth covering the new user-query mode (H) +
one hand-written CLEAN control (`tools/judge_fixtures.py`). Full transcript in
`audit/phase2_judge_validation.txt`. Summary (judge = Mistral-Small):

| fixture | covers | OLD judge (q+a only) | NEW: how it's caught | gate |
|---|---|---|---|---|
| bad1 | A + B | coherence 7, appr 5 | pre-filter hard-fail (harness + turn-index); judge `reasoning_quality=2`, sev-3 flags | **REJECT** |
| bad2 | B + F | coherence 3 | pre-filter hard-fail (turn-index); judge `reasoning_answer_consistency=1`, sev-3 | **REJECT** |
| bad3 | C + E | **coherence 10, appr 10 → PASS** | pre-filter hard-fail (artifact); judge `reasoning_quality=2` | **REJECT** |
| bad4 | D | **coherence 10 → PASS** | judge `reasoning_quality=4`, degenerate flag | **REJECT** |
| bad5 | G | coherence 2 | judge `sycophancy_resistance=1`, sev-3 | **REJECT** |
| bad6 | H (bad user query) | **coherence 10 → PASS** | judge `user_query_quality` low, `bad_user_query` flag | **REJECT** |
| clean | — | PASS | judge all 10 | **PASS** |

The old judge passed bad3/bad4/bad6 at 8–10; the new judge + pre-filter reject all
six bad fixtures and pass the clean one. **The judge, gate thresholds and pre-filter
are FROZEN after this point.**

## 4. PHASE 4 — autonomous generate → judge → analyze → fix loop

Each iteration = a real 10-seed batch (one greeting seed is curator-skipped → 9
conversations), concurrency 10, judge always-on. Audit JSON per iteration in
`audit/loop/iterN.json`; raw batches under `runs/loop/` (gitignored). The
pre-filter signals (objective) drove every fix; the judge means are secondary.

| iter | key change | harness | turn-idx (reason) | artifact | degenerate | regen>2 | mean rq | pass-rate | judge |
|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline (Phase-3 prompt) | 23 | 9 | 1 | 1 | 0 | 10* | 0.00 | Mistral |
| 2 | real chat-message history; robust JSON parse | 11 | 3 | 3 | 1 | 0 | 9.9* | 0.22 | Mistral |
| 3 | bare-identity prompt; turn-idx + artifact sanitize; mt 6000 | 3 | 0 | 0 | 0 | 0 | 9.8* | 0.67 | Mistral |
| 4 | aside guard v1; pre-filter precision fix | 5 | 0 | 0 | 1 | 0 | 9.8* | 0.67 | Mistral |
| 5 | steer-kwarg (no leak); broad aside guard; mt 5000 | **0** | **0** | **0** | 1 | 0 | 9.9* | 1.00* | Mistral |
| 6 | Qwen judge; freq_penalty 0.3; degenerate→hard-fail | **0** | **0** | 1 | **0** | 0 | 8.2 | 0.44 | **Qwen** |
| 7 | freq_penalty 0.1; markdown-glued artifact strip | **0** | **0** | **0** | 1 | 0 | 9.8 | **0.78** | Qwen |
| 8 | freq_penalty 0.2 | _(see final numbers below)_ | | | | | | | Qwen |

`*` = the Mistral judge scored almost exclusively 9–10 (never < 7), so pass-rate
1.0 in iter5 was the "gate too lax" signal — not a clean-data artefact. Re-judging
the **same iter5 batch** with Qwen3-Coder gave a real spread {8:4, 9:11, 10:84}
and pass-rate **0.89** (in band), which is why the judge was switched (§5).

### Per-iteration diagnosis → fix (the legible trail)

- **iter1 (harness 23, turn-idx 9):** the responder was fed a `User:/Assistant:`
  transcript blob + a rules-y system prompt. The native-reasoning model
  **re-numbered the transcript** ("History Check: Turn 1… Turn 2… Turn 3
  (Current)") and **quoted/agonized about the system prompt** ("despite the system
  instruction saying 'continuing an ongoing conversation'"). → **iter2:** pass
  prior turns as REAL chat messages (no transcript to number); drop the "ongoing
  conversation" framing (it created a turn-0 "contradiction" the model agonized
  over). A probe confirmed: 0 system-quoting, 0 turn-numbering — only the
  `Thinking Process:` header, which sanitization already strips.
- **iter2→3 (harness 11→3):** the model quoted *any* behavioural sentence in the
  prompt ("Wait, looking at the system instructions: 'Answer the most recent user
  message…'"). → reduce the responder prompt to a **bare warm identity** with no
  quotable imperatives. Also added meaning-preserving **turn-index de-numbering**
  and broader **artifact** stripping in the sanitization pass.
- **iter3→4 (harness 3):** 1 of the 3 was a **false positive** — the model
  correctly noting the *user's* request was contradictory. → narrowed the
  over-broad `contradictory…instructions` pre-filter pattern to require the SYSTEM
  context (precision fix; Phase 2 re-validated). The other 2 were a sporadic
  compliance-check tic → a targeted aside guard.
- **iter4→5 (harness 5→0):** two new causes — the simulator **leaked its
  `=== STEER ===` block** into the user query (the responder then reasoned about
  it) → steer now passed via a `steer=` kwarg, never in the prompt; and the model
  references its own prompt in **varied phrasings** (incl. hallucinated safety
  "system instructions" on crisis turns) → broadened the aside guard to strip any
  self-reference to "the/my system prompt/instructions".
- **iter5→6 (pass-rate 1.0 → judge swap):** Mistral was too lenient to let the gate
  discriminate. Switched to the discriminating **Qwen3-Coder** judge; promoted the
  objective **degenerate** detector to a deterministic **hard-fail** (Qwen
  under-penalizes a fluent loop); added a **`frequency_penalty`** decoding param to
  break the loop at source.
- **iter6→7 (pass-rate 0.44 → 0.78):** `frequency_penalty=0.3` over-penalized
  fluency (Qwen then rejected too many turns). Dialed to **0.1** (still crushes a
  27× loop — the penalty scales with count); fixed a markdown-glued artifact
  (`*Let's go.*cw`). 0.1 left one degenerate spiral on the mental-health support
  turn → **iter8** tunes `frequency_penalty=0.2`.

## 5. Judge↔simulator self-evaluation bias check & the judge choice

The user flagged that the judge (`Qwen3-Coder`) was the **same model as the
simulator** that writes the user queries — a self-evaluation bias risk. Two
findings:

1. **The bias does not materialize.** The iter1 bias probe scored the same batch
   with the frozen judge and with the simulator's own family: on
   `user_query_quality`, **Mistral-Small = 10.0 vs Qwen3-Coder = 9.89** — Qwen is
   *slightly stricter* on its own simulator's queries, the opposite of inflation.
2. **Mistral-Small is too lenient to gate.** Across iter5 it scored only 9s and
   10s on every axis (never < 7) and missed a 13.8k-char degenerate spiral
   (`reasoning_quality=9`). With a judge that never scores below the gate floor,
   the gate can only ever reject on the deterministic pre-filter, so a clean batch
   pins pass-rate at 1.0 — exactly the "gate too lax" failure the spec warns about.

So the **frozen judge is `Qwen3-Coder-30B`** (the task's specified judge): more
discriminating (real {8,9,10} spread → pass-rate in band), parseable, and
empirically un-biased on its own queries. The deterministic pre-filter catches the
objective A/B/C/D defects judge-independently, so Qwen's softer touch on E
(reasoning-about-format) does not matter — bad3 is still gate-rejected by the
artifact pre-filter. Phase 2 re-validated green with Qwen
(`audit/phase2_judge_validation_qwen.txt`).

## 6. Files changed / new config surface

**Changed (`src/rxai_sdg/factory/`):** `holistic.py` (judge prompt + rubric +
pre-filter), `responder.py` (`format_transcript_for_judge`, leakage detectors,
sanitization, harness-free prompt), `prompts.py` (responder/simulator prompts),
`loop.py` (pre-filter wiring + config-driven gate), `config.py` (gate config),
`user_simulator.py` (sanitize query), `clients.py` (token-usage accounting),
`__init__.py` (exports).

**New config surface (`FactoryConfig`):** `holistic_gate: dict[str,float]`,
`hard_fail_on_flagged_severity: int = 3`, `prefilter_regen_threshold: int = 2`. The
legacy `holistic_min_coherence/appropriateness` are kept (no longer used by the gate).

**New tooling:** `tools/judge_fixtures.py`, `tools/validate_judge.py` (Phase 2),
`tools/factory_loop.py` (Phase 4). **Tests:** `tests/factory/test_judge_prefilter.py`
(17 deterministic tests) + updates to two existing tests; full suite green.
