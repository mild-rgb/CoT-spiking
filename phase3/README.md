# Phase 3 — stub

**Status: pipeline ported; first experiment done.** Run on `Qwen/Qwen3-8B` on a Colab A100
(2026-07-28/29). See *First run* for the port and *SAE-space GCG* for the result. The committed
`.ipynb` is the executed copy, outputs included.

Picks up from `../phase2-junk-tokens/` (read its README first — the open threads at the bottom
are the phase-3 agenda).

## The model question

Phase 2 ran on `Qwen/Qwen3-4B-Thinking-2507`, and its one attempt at SAE analysis died on an
unusable community checkpoint. So: **is there a ~4B thinking Qwen with official SAEs?**

**No.** Checked 2026-07-28. Qwen's official suite is **Qwen-Scope** — residual-stream TopK SAEs,
one per layer, 14 repos under the `Qwen` org:

| backbone | SAE repo prefix | width | L0 | thinking? |
|---|---|---|---|---|
| Qwen3-1.7B-Base | `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-` | 32K | 50, 100 | hybrid (post-trained sibling) |
| **Qwen3-8B-Base** | `Qwen/SAE-Res-Qwen3-8B-Base-W64K-` | 64K | 50, 100 | hybrid (post-trained sibling) |
| Qwen3.5-2B-Base | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-` | 32K | 50, 100 | hybrid, non-thinking by default |
| Qwen3.5-9B-Base | `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-` | 64K | 50, 100 | hybrid |
| Qwen3.5-27B | `Qwen/SAE-Res-Qwen3.5-27B-W80K-` | 80K | 50, 100 | hybrid — **the only instruct-trained SAEs** |
| Qwen3-30B-A3B-Base | `Qwen/SAE-Res-Qwen3-30B-A3B-Base-` | 32K/128K | 50, 100 | MoE |
| Qwen3.5-35B-A3B-Base | `Qwen/SAE-Res-Qwen3.5-35B-A3B-Base-` | 32K/128K | 50, 100 | MoE |

No 4B of any generation. `Qwen/Qwen3.5-4B` exists as a *model* but has no SAE, and every 4B SAE
on the Hub is a hobby repo with single-digit downloads — the same class of artifact as the one
that failed in phase 2.

**Default choice: `Qwen/Qwen3-8B` + `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`.**

- **36 layers, same as the phase-2 model.** Layer indices transfer directly, which matters
  because the phase-2 alignment result peaked at **layer 32** and the plan is to retarget the
  lens there.
- Genuine `<think>` block via `enable_thinking=True` (hybrid, not thinking-only like the
  `-Thinking-2507` line — the chat template differs, see below).
- SAEs at **every** layer, on the residual stream, which is where the phase-2 `DELTA_MID`
  steering-delta lives.
- Cost: 8B, not 4B. bf16 weights ~16 GB — **A100-class, not a T4**. If you need a T4, switch
  `WHICH = "qwen3-1.7b"` (28 layers, so layer indices need rescaling, not copying).

Two caveats to keep in view, both handled by the last notebook cell:

1. **The SAEs are trained on the Base checkpoint** and we run the post-trained sibling. The
   model card calls that "reasonable"; FVE on our own activations is what decides it.
2. **Hook point is the residual stream *after* layer *n***, i.e. `hidden_states[n+1]` in HF's
   `output_hidden_states` convention. An off-by-one here looks exactly like a broken SAE, so
   the gate scans a small neighbourhood rather than trusting one index.

## Notebook

`gcg_pipeline_stub.ipynb` — the stock GCG pipeline lifted from phase 2, **pictographs allowed**,
and nothing else:

- model switch (`WHICH`), load, `steer(cue)` scaffold
- weak-token pool with the structural/chat-control-token guard (sizes: *Pool vs candidate set*)
- ID-splice scaffold, `grad_logit` (stock GCG), `search()`
- A/B/C/D verification helpers, `setup_target` (translations + top-300 embedding neighbours)
- a 20-step smoke test on `wolf`
- **⁂ an aside** (clearly marked, nothing depends on it): decomposition of the trigger the smoke
  test happened to find — see *Anatomy of one trigger* below
- the SAE FVE gate

**Deliberately not carried over** (all in `../phase2-junk-tokens/junk_tokens_steering.clean.ipynb`):
the `grad_lens` scorer and `DELTA_MID`, the 8-animal sweep, the disjoint-letter test, the
quality-ladder alignment analysis.

Changes from the phase-2 code, all forced by the hybrid backbone:

- `_build` handles three chat-template shapes — pre-opened `<think>` (Thinking-2507), no think
  block (hybrid + `enable_thinking=True`), and a closed *empty* block (hybrid, thinking off,
  which gets stripped) — and asserts exactly one open/close pair.
- `PRE_TXT` / `SUF_TXT` are derived by splitting `_build()` on a sentinel instead of being
  hardcoded, so the splice can't drift from the scaffold.

### Pool vs candidate set — two different filters, don't conflate them

Every search here (`search`, `search_sae`, `search_T`) shares one funnel, and "the 4096" refers
to the *middle* stage only. In GCG's own terminology the **candidate set** is the per-slot top-k
substitutions — here **256**, not 4096.

| stage | size | what does the cutting |
|---|---|---|
| vocab | 151936 | — |
| usable | 148013 | structural guard: specials, added/control tokens, `<...>` shapes, Cc/Cs/Co, empty/whitespace |
| **POOL** | **4096** | per-target blocklist, then rank by weakness `1 − rank(‖E[i] − mean(E)‖)`, take top-k |
| candidate set | 256 / slot | `gr[:, ~pool_mask] = inf`, then `topk(-gr, n_top)` — the gradient's proposals, 8 slots → 2048 pairs |
| evaluated | 64 or 128 | random (slot, pick) draws, forward-passed for real |

Three things that are easy to get wrong:

- **4096 is a top-k, not a threshold.** There is no weakness cutoff — it takes the weakest 4096
  that survive the guard *and* the blocklist, so the pool's composition shifts per target.
- **The pool is rebuilt on every `setup_target` call.** §7 therefore runs against 14 different
  4096-token pools, plus a 15th for stage 1 (union blocklist, 2411 tokens blocked).
- **617 of 4096 are pictographs** (measured pre-blocklist) — ~15%, which is the "pictographs
  allowed" decision doing visible work rather than a nominal setting.

## First run (2026-07-28, Colab A100 40 GB, bf16)

**The pipeline ports cleanly.** Qwen3-8B loads in 71 s; `_build`'s assert passes (the hybrid
template ends at `<|im_start|>assistant\n` and does *not* pre-open `<think>`, so `_build` adds
its own); the structural guard keeps `<think>`/`</think>`/`<|im_start|>` out of the pool; the
splice is **75 prefix + 13 suffix tokens, identical to phase 2**; vocab is 151936, also
identical. Embeddings are *untied* here (phase-2's 4B were tied) — irrelevant to the splice.

**New neutral baseline** — different from phase 2, so targets must be re-picked:

| | dolphin | elephant | wolf | cat | tiger | eagle | dog | lion |
|---|---|---|---|---|---|---|---|---|
| p | 0.615 | 0.200 | **0.050** | 0.039 | 0.039 | 0.027 | 0.011 | 0.006 |

Dolphin is even more dominant than on the 4B (0.615 vs 0.278), and **wolf is no longer a clean
discriminator** — it was 0.0079 and absent from the top-10 on the phase-2 model, here it is 3rd
at 0.050. Phase 2's "pick targets absent from the top-10" rule needs re-applying from scratch.

**Smoke test (wolf, 20 steps, batch 64, 14 s):** p 0.0505 → **0.9906** (real-cue ceiling
0.9998). A=`wolf.` ✓, B=`wolf` ✓, **C=`dog` ✗**, D=32/32. Stronger than phase 2's wolf (0.7255
at 60 steps, batch 128) — but it fails the strict free-reasoning test C, and the trigger
`ᅬ쓩𝗪sPidроссийск具有战士החלטה𝔴` contains `𝗪` and `𝔴`, folding to `w`. **The spelling confound
reappears immediately on the new backbone.**

### ⁂ Anatomy of one trigger (an aside, not a phase-3 result)

The smoke test's trigger `ᅬ쓩𝗪sPidроссийск具有战士החלטה𝔴` decomposes into **three independent
routes stacked past saturation**, each confirmable with a plain-English cue (prior 0.050):

| route | plain cue | p(wolf) |
|---|---|---|
| orthographic | `' w'` / `' W'` — the `𝗪`/`𝔴` are disguised `w` | 0.727 / 0.809 |
| cultural | `' Russian'` / `' Russia'` / `российск` | 0.461 / 0.505 / 0.575 |
| category | `' warrior'` (lion 0.156, tiger 0.107) | 0.699 |

Two things worth carrying into phase 3:

- **No single slot is load-bearing.** Largest leave-one-out drop 0.27, most under 0.05. Knock
  out *routes*, not tokens: both w-glyphs → 0.744, both semantic → 0.817, all four → **0.126**.
  On a trigger at 0.99, single-token ablation tells you nothing.
- **Weak-normed tokens are not semantically empty.** `具有战士` ("possesses warrior") alone gives
  **tiger 0.826**, lion 0.112, wolf 0.015 — a generic big-predator token, which is why the same
  token appeared in phase-2's *lion* triggers on the 4B model.

The cultural route has a clean negative control: under every Russia cue **p(bear) stays at
prior** (0.007–0.015), so the association routes to Russian *wildlife* (wolf; tiger second;
free reasoning answers `snow leopard`) rather than the heraldic Russian bear. Nor is it generic
Russia-adjacency — `' Siberia'` → 0.018, below prior, and `' vodka'` → 0.062.

**SAE gate: passes the letter of the test, fails the spirit.**

The checkpoint is genuine — tensors exactly as documented, and FVE over all token positions is
**0.840**. That is a real, trained SAE, unlike phase 2's. But two things came out of the gate
that change how it can be used:

1. **Its error is ~4× the signal we care about.** At the answer position, across 14 cue
   prompts: mean ‖h‖ = 115.1, mean ‖h − recon‖ = **48.6**, but the across-cue spread — the
   cue-minus-neutral steering delta, i.e. the entire phase-2 quantity — is only **11.8**.
   Reading the delta through this SAE reads its reconstruction error, not the delta.
2. **The gate cell's answer-position FVE (−13.8) is a metric artifact, not a failure.** The FVE
   denominator is across-sample variance, which is tiny when all 14 prompts share their whole
   context; the same reconstruction scores +0.822 against total energy. Same trap as the
   raw-cosine floor in phase 2 — never score a single shared-context position by FVE.

Also: **the hook-point scan does not discriminate.** FVE is flat to 4 decimals across
`hidden_states[18/20/21/22]` (0.8401/0.8401/0.8400/0.8396), falling off only at the extremes
(L2 −2.9, L36 −4.4). The `hidden_states[n+1]` convention is therefore *unverified*, not
confirmed — an off-by-one is invisible to this test.

→ Before agenda item 5, either find a readout that measures error **on the delta** rather than
on `h`, or accept that this SAE only resolves coarse features.

## SAE-space GCG (§5) — the first real phase-3 result

Record what the real `' wolf'` cue looks like in SAE feature space at layers 15/20/25/30/35
(skipping the first 40%), then run GCG in which **both the gradient and the accept test are the
SAE objective** — cosine between the trigger's pre-activations on wolf's top-100 features and
wolf's own values. `p(' wolf')` is never optimised, only recorded, so the question is clean:
*does driving a trigger into wolf's SAE representation make the model say wolf?*

| run | SAE score | p(wolf) | D | model says |
|---|---|---|---|---|
| diff-L15 | +0.9612 | 0.0060 | 0/32 | elephant |
| diff-L20 | +0.9576 | 0.0000 | 0/32 | cow |
| diff-L25 | **+0.9639** | 0.0000 | 0/32 | mouse |
| diff-L30 | +0.9507 | **0.6868** | 28/32 | **wolf** |
| diff-L35 | +0.8485 | 0.2417 | 8/32 | tiger |
| diff-ALL | +0.8732 | 0.0157 | 0/32 | tiger |
| raw-ALL | +0.9618 | **0.6498** | 22/32 | **wolf** |

*(prior 0.0505 · real cue 0.9998 · logit-GCG reference 0.9906)*

1. **SAE match quality does not predict behaviour.** `corr(score, p) = −0.191` (n=5). The
   best-scoring run reaches cos 0.9639 with wolf's feature vector and puts p(wolf) at exactly
   **0.0000**. "I matched the SAE representation" is not evidence of having reproduced the thing.
2. **It is a depth story — a sharp inverted U peaking at layer 30 of 36.** Below 30 the
   objective is satisfiable by *any* animal, and the search says so out loud: it plants an
   animal emoji (🐄 → "cow", 🐁 → "mouse"). Mid-layer wolf-minus-neutral features encode
   *"a specific animal is being cued"*, not which one. Only ≈0.83 depth is wolf-specific enough
   to force the answer — consistent with phase-2's alignment peak at layer 32/36.
3. **Combining layers hurts.** diff-ALL scores 0.0157 and says tiger, against 0.6868 for L30
   alone. The compromise that satisfies every layer satisfies none behaviourally.
4. **A prediction of mine failed and is worth recording.** I argued the *raw* target (wolf's own
   top-100 features, no neutral subtraction) would be useless because a neutral cue already
   scores 0.9222 of a perfect 1.0 — a usable range of 0.078. It reached p=0.6498, 22/32, as good
   as the best differential run. A near-saturated objective can still carry the signal in its
   last 8%.
5. **Convergent evidence for §"Anatomy of one trigger".** The only two runs that produced wolf
   share exactly two tokens: **`𝗪`** (the disguised-`w` route) and **`שומר`** (Hebrew
   "guard/watchman"); diff-L30 also contains **`российск`**. An objective that never sees the
   output distribution rediscovered the same routes the logit-GCG trigger used.

Caveats: one seed per cell, 5 probe layers rather than all 21, and the `hidden_states[n+1]` hook
convention remains assumed rather than verified.

**Operational note.** Looping over 5 SAE downloads in one call OOM-killed the kernel and lost all
state (model included). §5.1 is now resumable and defaults to 2 layers per call, with
`HF_HUB_DISABLE_XET=1`. One gotcha left: `os.remove` on the snapshot path deletes the symlink,
not the blob — use `os.path.realpath` to actually reclaim the 2.15 GB.

## Why layer 20 accepted a cow (§6)

The layer-20 search planted 🐄 and still scored 0.9576 against **wolf's** target. That is not the
optimiser gaming an adversarial point — at that depth the objective simply cannot tell animals
apart. Three views, all agreeing:

**1. Real cues, scored on wolf's objective** (1.000 = the real `' wolf'` cue, 0.000 = neutral):

| | L15 | L20 | L25 | L30 | L35 |
|---|---|---|---|---|---|
| mean of 13 other animals | +0.9889 | **+0.9937** | +0.9824 | +0.8736 | +0.8955 |
| best other | +0.9968 | +0.9981 | +0.9936 | +0.9571 (fox) | +0.9603 |

A real `' cow'` cue scores **0.9916** on wolf's layer-20 objective — *higher than the searched cow
trigger managed*. Even at L30 the objective is only graded, not exact: fox (0.9571) still
outscores the trigger the L30 search found (0.9507).

**2. Feature-set overlap** — each animal's *own* top-100 differential features, compared as sets:

| | mean over all 91 pairs | wolf ∩ cow | wolf vs others |
|---|---|---|---|
| SAE layer 20 | 63.1 / 100 | 53 | mean 66.2 (min 41, max 79) |
| SAE layer 30 | 24.5 / 100 | 19 | mean 29.1 (min 12, max 47) |

At L20 any two animals share two-thirds of their features. At L30 a quarter — and the remainder
is a genuine taxonomy: wolf's nearest are fox 47, tiger 45, lion 40; tiger–lion 60; cow–horse 49;
whale is isolated (29 with dolphin, 12 with wolf).

**3. The cross-target delta floor** — `cos(h(a)−h(neutral), h(b)−h(neutral))`, no SAE involved.
**This closes the caveat phase 2 flagged and never measured.**

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 30 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean off-diag | 0.654 | 0.790 | 0.750 | 0.913 | 0.931 | 0.927 | 0.803 | 0.652 | **0.522** | 0.733 |

So mid-network, *every* animal's steering delta points nearly the same way (0.93), and the floor
bottoms out at **layer 32** — exactly where phase 2 found its alignment peak. That validates
phase 2's choice of layer 32 rather than undermining it: 32 is where animals are maximally
distinguishable. It also explains why `grad_lens` at **layer 18** was mediocre — it was reading
where the floor is ≈0.91, so almost any cue looks like any other.

**The synthesis:** mid-layers encode *"a specific animal is being named"* — the slot, not the
filler. The filler only becomes linearly legible around layer 30–32. An objective built on
mid-layer representations is therefore underdetermined by construction, which is exactly what
§5's depth curve showed.

## Warm start from the L20 base (§7) — no effect, and the prior is what matters

Stage 1 finds one trigger matching the *mean* animal delta at layer 20 (`🐁ᅴהחלט𝓭꞊풂🐁ﲢ`,
score 0.7811 → 0.9693). Stage 2 starts from it and refines against each animal's own **layer-30**
representation, 100 steps, 14 animals, against a cold-start control at identical seed and budget.
35 min total.

**The warm start does nothing.** Targets answered: warm **9/14**, cold **9/14**. Mean p 0.5385 vs
0.5235, mean D 18.4 vs 18.6 / 32.

**Success is governed by the prior:** `corr(best p, log10 prior) = **+0.855**`. All 9 animals with
prior ≥ 0.0077 succeed; 4 of the 5 below 0.0025 fail outright (fox is a partial at 9/32).

| animal | prior | warm p | cold p | | animal | prior | warm p | cold p |
|---|---|---|---|---|---|---|---|---|
| dolphin | 0.3730 | 0.9424 | 0.9685 | | dog | 0.0113 | 0.9857 | 0.8174 |
| elephant | 0.3730 | 0.6414 | 0.8038 | | whale | 0.0077 | 0.9713 | 0.4943 |
| tiger | 0.0734 | 0.9716 | 0.8111 | | fox | 0.0025 | 0.0154 | 0.2985 |
| wolf | 0.0505 | 0.9066 | 0.2819 | | bear | 0.0002 | 0.0126 | 0.0333 |
| cat | 0.0505 | 0.6257 | 0.8665 | | horse | 0.0002 | 0.0306 | 0.0788 |
| eagle | 0.0238 | 0.6758 | 0.9815 | | cow | 0.0000 | 0.0079 | 0.2463 |
| lion | 0.0128 | 0.7422 | 0.6400 | | mouse | 0.0000 | 0.0099 | 0.0063 |

**This is the sharpest contrast with phase 2 so far.** Logit-GCG had
`corr(final p, log10 prior) = −0.024` and drove **crab from a 0.0000 prior to 0.92**. The SAE-space
objective cannot do that: it amplifies an animal the model already half-believes and fails to
install one from nothing.

**A pattern that did not survive the sweep.** At 10 of 14 animals the warm/cold split fell exactly
on carnivore vs non-carnivore, 6/6 consistent (p ≈ 0.07 under random assignment). The final four
broke it: carnivores where warm wins are **5/10**, and the three largest warm *losses* in the table
(eagle −0.306, fox −0.283, cat −0.241) are all carnivores. `corr(Δ, is-carnivore) = +0.300`.
Recorded because the partial pattern was genuinely convincing — a uniform prefix says nothing
about the tail.

The mechanistic explanation fails too: `corr(Δ, base-trigger score on that animal) = **−0.283**`,
i.e. the base being *closer* to an animal's target does not help and may hurt. Note mouse has the
highest base score (0.9589) — the stage-1 trigger contains 🐁 twice, because the union blocklist
covered animal *names* but not emoji. That confound is worth fixing before re-running §7.

## Before trusting anything on the new backbone

Everything in phase 2 is model- and scaffold-specific. Re-establish, in this order:

1. **The neutral baseline.** Phase 2's was dolphin 27.8% / panda 14.9% / octopus 14.9%. Unknown
   here. Targets must be picked *absent* from this top-10, or steering to them measures nothing.
2. **Single-token screening.** `' <word>'` must be one token. `' penguin'` was not, on the
   phase-2 tokenizer.
3. **The `' **'` problem.** Confirm the no-markdown system message is still doing its job — if
   the top post-`</think>` token is formatting, the answer slot isn't reading preference.
4. **A phase-2-comparable result** (the smoke test), before building anything on top.

## Agenda

Inherited from phase 2, in rough priority order:

1. **Spelling vs semantics.** Disjoint-letter triggers capped at ~0.5–0.6 while spelling-based
   ones hit 0.99. Real ceiling, or just a weaker search? More steps, larger batch, several seeds.
2. **Retarget the lens to layer 32** (alignment peak) instead of 18.
3. **Measure the cross-target `DELTA_MID` floor** — every target's delta shares a
   generic→specific component, which puts an unmeasured floor under the +0.699 partial r.
4. **Transfer.** Does any trigger survive a reworded prompt, a different lead-in, or phase-1's
   guard system prompt?
5. **Now actually possible: SAE the winning triggers** — but only past the FVE gate.
