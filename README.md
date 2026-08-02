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
| `phase3/` | **experiments done, agenda open.** Stock GCG ported to a backbone with official SAEs (`Qwen/Qwen3-8B` + Qwen-Scope), then GCG run *in SAE feature space* (§5–§7). Three results in; the inherited agenda at the bottom of its README is not exhausted. Notebook: `gcg_pipeline_stub.ipynb` (executed, outputs included). |
| `phase4/` | **complete.** The same stock-GCG spine against a **non-thinking** model — `Qwen3-8B` with `enable_thinking=False`, trigger spliced into the user turn at either end. Three experiments; the owed pictograph control was honoured in phase 5. ⚠ Its §6 ActAdd layer curve is **superseded by phase 5** (extraction artifact). Notebook: `gcg_nonthinking_stub.ipynb` (executed, outputs included). |
| `phase5-steering-vectors/` | **§1–§7 run (2026-07-30).** Isolate a *bridge* steering vector, then GCG-search for a token trigger whose activations match the steered ones. The search wins the objective and gets 0.0% of the behaviour; a hand-written phrase in the same slots gets 100%. `RESEARCH.md` is the extraction literature note, `RECIPE.md` the protocol as run. Notebook: `steering_vectors_stub.ipynb` (executed, outputs included). |
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
- **Phase 3 — experiments done, agenda open (2026-07-28/29).** Pipeline ported to `Qwen/Qwen3-8B`; smoke test
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
- **Phase 4 — done (2026-07-29).** Stock GCG for the same animal question against a
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
    backbone — two forward passes instead of 50 GCG steps. Two backbone-specific fixes were
    needed: skip position 0 (Qwen's attention sink runs ‖h‖ ~10⁴, and the
    canonical `" bridge"`−`" "` pair wrecks it into `"said said said"`), and use a pair whose
    *final* token is the topic — the classic negation pair carries the assertion rather than the
    subject, with the model saying *"I love discussing the importance of the topic"* and never
    naming it. ⚠ **Its layer curve is wrong** — the reported L4–L6 peak was an extraction
    artifact of the same sink; corrected in phase 5, the peak is L16–L20.
- **Phase 5 — §1–§7 run (2026-07-30).** The direction reverses. Phases 1–4 searched for an *input*
  that moves the answer; phase 5 takes an **activation edit as ground truth** and asks what input
  reproduces it. Two steps: isolate a steering vector that makes the model really enjoy talking
  about **bridges** (phase 4 §6's ActAdd work, consolidated and finally measured properly), then
  run a **GCG search whose objective is matching the steered activations**. Phase 4's GCG spine is
  kept intact with the scorer and gradient passed in; the favourite-animal apparatus — the probe
  scaffold, the one-token answer slot, the A/B/C/D ladder, the animal blocklists — is gone, since
  the readout is now free generation scored against a topic word list.
  - **Two published results shape it.** *Steered LLM Activations are Non-Surjective* finds steered
    activations sit off the manifold of natural-prompt activations, with the SIPIT inversion
    algorithm failing at the very first token — which bounds *exact* matching but not the
    experiment, since our pool is junk tokens rather than natural text. And *Activation-Guided
    GCG* already swaps GCG's log-likelihood loss for residual-stream projection losses and reports
    better success per step, so the objective has precedent; its single-layer / layer-wide /
    token-wide / global variants are a real axis, against phase 3's prior that combining layers
    hurt.
  - Phase 4's **owed pictograph control** is honoured here as a one-flag default
    (`BLOCK_PICTOGRAPHS = True`). See `phase5-steering-vectors/RESEARCH.md` for the extraction
    literature — method taxonomy, the pair-design and layer/normalisation rules, Tan et al.'s
    reliability caveats, and ActAdd's evaluation metric.
  - ⚠ **First result: phase 4's ActAdd layer curve was an extraction artifact.** `' bridge'` and
    `' cat'` are single tokens, so a bare pair puts the topic token on **position 0 — the attention
    sink**. From **L7** the two prompts are the same vector to five decimal places
    (cos = 0.99995), 99.9% of the energy sits in three dimensions, and ‖v‖ is pinned at 1310.5 all
    the way to L24: a difference of two massive-activation spikes, carrying no semantics. Phase 4
    sampled {4, 6, 8, 12, 16}, never saw that the break is at 7, and read a dead vector as dead
    steering — so **"peaks at L4–L6, dies by L8, mirroring ActAdd's own layer curve" is wrong, and
    the resemblance to ActAdd's curve was coincidence.** Fixed with a shared context prefix
    (`"The word is bridge"` − `"The word is cat"`): the two forms agree at L2–L6 (cos 0.76–0.88),
    go orthogonal from L7 (0.037), and the corrected curve puts coherent on-topic steering at
    **L16–L20**, with L2–L6 producing only degenerate repetition. That agrees with the literature's
    mid-to-late finding and against the ~12%-depth story inherited from GPT-2-XL.
  - **CAA over 8 negatives strictly dominates a single prompt pair** — 99.0% vs 94.8% on-topic,
    4.2% vs 11.5% degenerate, perplexity 11.6 vs 14.5, for 8 extra forward passes. The eight
    `bridge − X` vectors agree only ~0.42 pairwise, so a single-pair vector is about a third
    idiosyncratic. Controls hold: reversal gives 0.0% on-topic, and a **matched-norm random
    direction** gives 0.0% at perplexity 7.8 against an unsteered 5.4 — the effect is the
    direction, not the magnitude.
  - **A steering vector is roughly twice as potent as the best possible token splice** — 99.0%
    against 52.1% for the real word `' bridge'` in the same 8 slots. A behavioural measure of what
    the non-surjectivity result asserts geometrically.
  - ⚠ **Activation match quality does not predict behaviour — phase 3's result, replicated on a
    different objective, target and behaviour.** GCG matching the steering vector's downstream
    delta beat the real word 2:1 on the objective (+0.8497 vs +0.4204) and produced **0.0%**
    bridges, the same as random junk. Swapping the gameable projection for magnitude-free cosine
    changes nothing: the cosine-optimised trigger aligns *better than the genuine near-synonym*
    `viaduct` and still gives 0.0%. Its pieces decode as **disease, fear, borrow money, infection,
    lose, war** — a coherent cluster that is not bridges, so the direction mixes topic with
    something else. And a mid-layer version of the objective is worse than useless: at L17–L29
    random junk *outscores* the real word, because it measures slot occupancy rather than content
    — phase 3's depth story again.
  - **The prefill-only ablation says why, and it is not what the nearest prior work thinks.**
    [rain-1/emergent-misalignment-steering-with-tokens](https://github.com/rain-1/emergent-misalignment-steering-with-tokens)
    independently reports the same negative (GCG margin trigger 0/24 and 0/30 against an activation
    vector at 93–96%, vector-objective GCG plateauing at α-equivalent 0.51 where behaviour needs
    2.5) and attributes it to the vector being *re-added at every generated position* so its push
    compounds. Tested here, that is wrong: dropping re-injection during generation costs 99.0% →
    **95.8%** and *improves* fluency. The advantage is **spatial extent, not temporal compounding**
    — confine the same vector to a trigger's 8 slots and it gives **3.1%**, with 16× strength still
    giving 0.0% and perplexity unmoved. Yet a **real word in those same 8 slots gets 52.1%**, 17×
    the vector. So tokens and activation edits are not interchangeable interventions even at
    identical sites. The trade-off is the real finding: **the behaviourally potent target is
    unmatchable and the matchable target is inert** — the reference §3.3 selected for matchability
    corresponds to a 3.1% behaviour, so a perfect match would have scored ~3%.
  - **Their trigger derivation differs from ours in pool and channel — both now ruled out.** They
    search plain English words and splice mid-reasoning; we inherited phase 2's weakest-norm junk
    pool and appended to a bare user turn. A 2×2 over both, behavioural objective, scored on free
    generation: **0.0% in all four cells.** Junk is in fact the *better* optimisation substrate
    (log p −3.88 vs −5.48 for words) and still converts to nothing.
  - ⁂ **And the ceiling is 100%.** A fluent five-token phrase in those same 8 slots —
    `' the user really loves bridges'` — reaches **100%**, above the steering vector's 99.0%, for
    two fewer forward passes. `' bridge'` alone gets 48.6%, so **fluency is the active ingredient**
    and the channel was never the limit: GCG had a hundred points of headroom and took none of it.
    The uncomfortable implication for phases 2–4: `p(' wolf') = 0.9991` measures **next-token
    control at a prefilled answer slot**, which GCG is excellent at, while phase 5 measures
    **sustained topical behaviour in free generation** and gets 0.0% across 6 searches, 2
    objectives, 2 pools and 2 channels. Phase 4's rung B hinted at the gap (42/80 failures at
    p ≈ 1.0). Only the first claim is established by phases 2–4.
  - ⁂ **The final control says it is a proxy problem, not a search problem.** Adding the two fixes
    the adversarial-prompt literature already has — multi-prompt optimisation (Zou et al.) and a
    fluency penalty (AutoDAN / COLD-Attack / FLRT) — drives trigger perplexity from 3×10⁷ down to
    284, producing grammatical English (`' Did this cat and the duck live together'`), and still
    yields **0.0%**. Scoring the *working* hand-written phrases on that same objective settles it:
    **every GCG trigger beats every working phrase**, and the rank correlation between proxy score
    and free-generation behaviour is **−0.55** across all nine interventions and **−0.60** among
    the five human phrases alone. GCG won at what it was told to maximise. Every cheap
    differentiable proxy tried — activation match (projection and cosine) and teacher-forced target
    NLL — orders interventions backwards relative to the behaviour.
  - **And it is not a keyhole.** Profiling the match at *every* layer downstream of the injection
    kills the natural explanation: GCG's triggers track the steering vector across the whole depth
    profile, with the same shape as the real word's, while the intervention that reaches 100%
    matches **worst** of the non-random candidates at every depth (mean +0.155, near the random
    floor of +0.149). Activation match is **uninformative** (corr +0.05); teacher-forced NLL is
    **inverted** (−0.55). The steering vector's activation signature simply is not what produces
    the behaviour — a fluent statement of preference and an activation edit are two different
    internal routes to the same output.
