# CoT-spiking

Interpretability experiments on `Qwen/Qwen3-4B-Thinking-2507` — planting text in a
reasoning model's `<think>` block and studying what it does to the answer.

By [@mild-rgb](https://github.com/mild-rgb). Work done for EleutherAI's SOAR program.

## Layout

Each phase is self-contained, with its own notebook.

| path | contents |
|---|---|
| `phase1-cot-injection-steering/` | **complete.** CoT-injection steering & system-prompt defence (findings #1–12). Start at its `OVERVIEW.md`; `RECIPE.md` is the full reproduction protocol. Notebook: `qwen3_favourite_animal_logits.ipynb` (stages 0–3). |
| `phase2-junk-tokens/` | **complete.** Covert triggers found by GCG over an undertrained-token pool. Start at its `README.md`. Notebook: `junk_tokens_steering.clean.ipynb` (runs top-to-bottom). |
| `phase3/` | **in progress.** Stock GCG ported to a backbone with official SAEs (`Qwen/Qwen3-8B` + Qwen-Scope), then GCG run *in SAE feature space* (§5–§7). Notebook: `gcg_pipeline_stub.ipynb` (executed, outputs included). |
| `phase4/` | **in progress.** The same stock-GCG spine against a **non-thinking** model — `Qwen3-8B` with `enable_thinking=False`, trigger spliced into the user turn at either end. Notebook: `gcg_nonthinking_stub.ipynb` (executed, outputs included). |
| `embedding-space-football-metaphor.md` | standalone note: intuition for the embedding-space picture used throughout. |

## Phases

- **Phase 1 — CoT-injection steering (done).** A single sentence planted in the
  `<think>` block steers the final answer (100/100 under prefill); a guard system prompt
  is a near-total defence via immediate self-reassertion, not injection detection. Full
  writeup and 8-stage recipe live in `phase1-cot-injection-steering/`.
- **Phase 2 — junk tokens (done, 2026-07-28).** Successor to phase-1 finding #6
  ("SolidGoldMagikarp glitch tokens make useless triggers"). A GCG-style search over an
  8-slot trigger drawn from the 4096 weakest-norm tokens, with the target blocklisted in
  ~40 languages plus its top-300 embedding neighbours. Headlines:
  - **Finding #6 does not generalise to combinations.** Single glitch tokens don't steer;
    optimised 8-token strings do — panda 0.9991, crab 0.9236, elephant 0.7898 against priors
    of 0.13 / 0.0000 / 0.04, verified under sampling and under free reasoning with no forced
    `</think>`.
  - **Success does not track the prior** (`corr = −0.024`); it tracks **letter overlap**
    (`corr = +0.596`). Much of the effect is the target word spelled in exotic Unicode.
  - **Forbidding every token that shares a letter with the target collapses 3 of 8 runs** —
    so the effect is neither purely spelling nor purely semantic. Non-spelling triggers exist
    but cap around 0.5–0.6 instead of 0.99.
  - **What the working triggers do** is reconstruct the real cue's residual-stream steering
    direction: within-animal `corr(p, alignment)` peaks at **+0.86 at layer 32**, partial
    **r = +0.699** after residualising on degradation level, positive in 8/8 animals.
  - A mid-layer "twist lens" gradient is a **better proposal scorer** than the standard GCG
    logit gradient (predicted-vs-realised corr +0.348 vs −0.192, n=8).
  - Negative result kept on the record: the one public SAE for this model/layer is unusable
    (reconstructions ~1000× too large, encoder/decoder anti-aligned).
- **Phase 3 — in progress (2026-07-28/29).** Pipeline ported to `Qwen/Qwen3-8B`; smoke test
  0.050 → 0.9906 in 20 steps, and the spelling confound reappeared immediately. Then three
  results, all in `phase3/README.md`: **SAE match quality does not predict behaviour** and the
  SAE-space objective is a *depth* story peaking at layer 30/36; **the cross-target delta floor
  bottoms out at layer 32** (0.522), which is why mid-layer objectives can't tell animals apart
  — layer 20 accepted a 🐄 against wolf's target; and a warm start from the shared L20 base
  **does nothing**, with success governed by the prior (`corr = +0.855`) unlike logit-GCG.
  Original agenda: push the spelling-vs-semantics split, retarget the
  lens to layer 32, measure the cross-target `DELTA_MID` floor, test whether any trigger
  transfers off the scaffold it was optimised on — and finally read the winning triggers through
  an SAE that actually reconstructs our activations.
  **There is no ~4B thinking Qwen with official SAEs** (checked 2026-07-28): Qwen-Scope covers
  1.7B, 8B, Qwen3.5-2B/9B/27B and two MoEs, and `Qwen3.5-4B` exists as a model but has no SAE.
  Phase 3 therefore moves to **`Qwen/Qwen3-8B` + `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`** —
  same 36-layer depth as the phase-2 model, so layer indices carry over. See `phase3/README.md`
  for the full table and the caveats (Base-trained SAEs on a post-trained model; hook point is
  `hidden_states[n+1]`).
- **Phase 4 — in progress (2026-07-29).** Stock GCG for the same animal question against a
  **non-thinking** model: `Qwen/Qwen3-8B` with `enable_thinking=False`, i.e. phase 3's exact
  weights with only the channel changed. With no `<think>` block to spike, the 8-slot trigger
  moves into the **user turn**, and the identical search runs at *both* ends of it. Headlines:
  - **Steering does not need the CoT channel.** 20 steps, 12 s, both ends beat phase 3's
    thinking-scaffold smoke test: suffix 0.0333 → **0.9951**, prefix 0.0597 → **0.9991**, each
    surviving rung B (no prefilled answer slot) and 32/32 under sampling.
  - `российск` and `具有战士` reappear — the same routes phase 3 found in the `<think>` channel
    and in SAE space. Three independent objectives, same components.
  - **Then a 40-animal sweep (2 positions, 50 steps, seed 1) overturned three of the four
    patterns the n=1 smoke test suggested.** 35/40 targets pass 0.5 and 29/40 pass 0.9 — but
    **position has no effect** (prefix wins 18/40, sign test p=0.636), **transfer is poor in both
    directions and not asymmetric** (42% vs 49% retention, against 27%/84% at n=1), and **prior
    dependence is real**: `corr(best p, log10 prior) = +0.560`, between phase 2's −0.024 and
    phase 3's +0.855. Spelling is *not* the driver here (+0.119 vs phase 2's +0.596) — but
    **79 of 80 triggers carry a pictograph** from a pool only ~15% pictograph, so a no-emoji
    control is owed before any of this counts as *covert* steering.
  - **A rung-C control broke the "reasoning defends" reading too.** Splice the *real word* into
    the same scaffold and it survives only 36/80; and 54/80 runs never closed `<think>` inside
    the 320-token budget, so those were censored observations scored as failures. Conditional on
    the block closing: real cue 64%, trigger 42% — a genuine covert-steering gap, but not the
    wall it looked like. Sharpest unexplained effect: a plain cue at the *head* of the user turn
    is ignored once the model reasons (32%) while the same cue at the *end* survives **90%**.
    See `phase4/README.md`.
  - ⁂ **Aside: ActAdd replicated with a bridge vector.** Turner et al.'s wedding demo on this
    backbone — two forward passes instead of 50 GCG steps. It works at L4–L6 after two
    backbone-specific fixes: skip position 0 (Qwen's attention sink runs ‖h‖ ~10⁴, and the
    canonical `" bridge"`−`" "` pair wrecks it into `"said said said"`), and use a pair whose
    *final* token is the topic — the classic negation pair carries the assertion rather than the
    subject, with the model saying *"I love discussing the importance of the topic"* and never
    naming it.
