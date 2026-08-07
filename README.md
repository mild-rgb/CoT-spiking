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
| `phase6/` | **§1–§6 run (2026-08-02).** A continuous, differentiable topic metric — `Σ_v p(v)·cos(e_v, e_bridge)` over free generation — and a tuned GCG search that maximises it. The metric separates real bridge queries from controls in all four embedding spaces; GCG raises only the space it was optimised on and produces no bridges. Notebook: `phase6_stub.ipynb` (executed, outputs included); raw record in `phase6_results.json`. |
| `phase7/` | **§1–§5 run (2026-08-03/04).** A gradient target in activation space: stop the backward pass at layer L, form `h_L + η·grad`, then search for tokens that land there. The edit is written in directly first and produces nothing (max gain +0.0070 against the +0.0313 needed); the token search cannot reach it for geometric reasons (needs `cos > 0.329`, tokens deliver 0.0206). ⁂ A comma scores 0.0990 under phase 6's metric, above `tell me about bridges`. Notebook: `phase7_gradient_target_stub.ipynb`; raw record in `phase7_gradient_target.json`. |
| `phase8/` | **complete (2026-08-04/05).** Phase 2's "twist lens" transplanted onto phase 6's bridge problem, plus the two controls NEXT-STEPS asked for. ⁂ The metric ranks all five *working* phrases in or above the bridge-query band in all four spaces (the objective was never the problem); ⁂ equal-compute **random search matches GCG** (0.0622 vs 0.0623); ⁂ the lens objective is reached at 2.8× the working phrase's value by genuine alignment and yields no behaviour. Start at its `README.md`; `RECIPE.md` is the protocol. Notebook: `twist_lens_bridge_stub.ipynb`; raw record in `phase8_twist_lens.json`. |
| `phase9/` | **complete (2026-08-05).** The bridge objective dropped for **decoherence** — make the model emit slop rather than an answer. ⁂ No trigger found: 42 hand-written triggers span 0.54–1.19 bits against a 0.683 baseline and an activation edit that reaches 4.80 on demand. ⚠ **The search manufactured four triggers that do not exist** — its accept test maximised over resampled rollouts, and 4 of 5 headlines collapsed on a 10× replication. ⁂ The one survivor is a *language hijack*, not decoherence. Start at its `README.md`; `RECIPE.md` stage 6 is the accept-test bug. Notebook: `decoherence_stub.ipynb`. |
| `phase10/` | **§0–§8 run (2026-08-05).** The soft-prompt upper bound on decoherence, and the discrete search that follows it. ⁂ A norm-constrained soft prompt reaches **12.765 bits** — the uniform-draw corner, 3× the activation ceiling — as its *typical* sample; its nearest-token projection collapses to **0.58**. ⚠ But that 12-bit gap is the cost of **rounding**, not of discreteness: discrete search from the same projection reaches **+0.697**, and from phase 9's survivor **+1.160**, on held-out seeds. ⁂ **GCG works on Qwen3-8B** (p('Sure') 1.0000) — the backbone confound is broken, and **phase 8 §7 is withdrawn** as mis-tokenised. ⁂ A **fourth exit** from the entropy objective — register/topic hijack — is what a properly-accepted search actually finds. Start at its `README.md`; `RECIPE.md` is the pre-registration. |
| `phase11/` | **§0–§6 run (2026-08-05).** GCG against **first-token entropy** (`H1`), phase 10 §8's proposal score promoted to a target. One deterministic forward pass per accept, so phase 9's noise-maximising failure mode is impossible by construction. ⁂ **GCG reaches 13.760 bits** against a 0.237 baseline and phase 10's best discrete trigger at 5.881 — *NEXT-STEPS item 2 no longer holds.* ⁂ **The discreteness gap is 3.3 bits, not 12** — phase 10's projection gap was an upper bound, as its §7 warned. ⁂ **The gradient beats random (+1.851 bits)** at identical init, breaking the phase 8 §5 / phase 10 §2 tie — but random alone reaches 11.909, so the *accept test* was the binding constraint all along. ⁂ 100 runs find **100 unrelated triggers** (overlap 0.014/16) driving **one shared state** (top-50 overlap 11.61/50, ~690× null). ⚠ No predictions were pre-registered. Start at its `README.md`; `RECIPE.md` is the protocol *as run*. |
| `phase12/` | **§0–§3 run (2026-08-06).** The anchoring control phase 11 §13 called for and could not run: swap slot 12 of phase 11's trigger (`毛主席`, id 116546) for eleven other single-token names plus a non-name control, hold the other fifteen slots fixed, re-run §10's 48-seed survey on each. ⁂ **The Mao persona is entirely the token** — 6/48 with it present, 0–1/48 in all twelve swaps. ⁂ **It is a quotation slot, not a name slot** — the non-name control `minimal` is quoted 7/48, mid-range among the names and *above* `毛主席` at 4/48. ⁂ **Quotation is not adoption** — across 624 rollouts names are parsed and discussed, never inhabited, and `Tesla` is read as the car. ⁂ **The King James voice is the prefill's, not the trigger's** — 46/48 from three tokens on a *clean* prompt against 1/48 from the trigger, and the trigger *degrades* it to 34/48. ⚠ Sampled counts do not reproduce phase 11 §10 exactly (transformers 5.13.1 vs 5.14.1); treat as ±2. ⚠ No predictions were pre-registered. |
| `phase13/` | **§0–§10 run (2026-08-06).** Points the phase 11 rig at *behaviours* instead of flatness: five objectives (street openers, KL-penalised, spread-penalised, 60 malice words, villain openers ×2), plus hand-built triggers. ⁂ **Warm-started GCG writes instructions, not covert directions** — it wrote ` okay responses`, `死战场`, `make趣味Text开玩笑` into its own triggers and the model obeyed; 5/5 objectives. ⁂ **Target-set design dominates**: three collapses onto the blandest member (`Okay`, `HA`, `Rise`), and the words that carry a persona were excluded by tokenisation. ⁂ **The one success describes a *speaker*** (`impatient`, `簡單`, `兄弟`) — 93% first-token mass over 7 openers, 44/48 in register, and it **transfers** (12% vs 44% meta-analysis on unseen questions) while being gated by question type, 6/6 vs 0/6. ⁂ **No emergent misalignment**: code security untouched, politics moves in tone only. ⚠ Three metrics were wrong the same way (English-only lexicon, no word boundaries, threshold artefact). ⚠ No pre-registration. |
| `phase14/` | **§0–§6 run (2026-08-06/07).** Phase 13 §0's extraction recipe, run again on a *different* persona in the same 250-step trigger's rollouts. ⁂ **Predictions registered before the run** — the first phase to do so; six of twenty false, two of which redirected the phase. ⁂ **Six persona families, two survive phase 12's slot-swap control** (archaic English 16/624 and rough Japanese 19/624, both under 13/13 variants; street is *not* one of them). ⚠ **Tokenisation killed the Japanese objective before any search** — 3 of 24 markers are single tokens, phase 13 §6 in a second language. ⁂ **Four β weights, four triggers, all four wrote a word for "archaic language"** onto phase 11's `lang` token — "GCG writes instructions" is now a *within-objective* result. ⁂ **Describing a voice vs an edit verb splits transfer 0/48 vs 24/48** with everything else fixed. ⁂ **A collapse onto a target with no bland reading is the best arm**, so phase 13 §2 needs narrowing, and the spread penalty *hurt*. ⚠ **Phase 12 §3 overturned** as a prefill-length artefact: at one token, clean + ` behold` is 0/48. ⁂ **Language and register are separately controlled** — a prefill sets the language 46–48/48, the trigger sets the register 0/48 vs 23/48. ⚠ Two more lexicon failures, same direction as phase 13 §9's three. |
| `NEXT-STEPS.md` | open lines of enquiry across the whole programme. Items 1, 2, 3, 3b, 6, 7 resolved; **item 8 (the objective cannot tell a different answer from a broken one) is the blocking one.** |
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
  - *(Phase 6 closes the remaining excuse: see below.)*
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
- **Phase 6 — §1–§6 run (2026-08-02).** Phase 5 ended on a proxy problem: every cheap
  differentiable objective it tried was uncorrelated or anti-correlated with the behaviour it
  stood for. Phase 6 starts from the other end — build a topic measure that is continuous,
  differentiable and defined on **free generation** rather than a prefilled slot, verify it
  orders interventions correctly, and only then optimise against it.
  `score = Σ_v p_t(v)·cos(e_v, e_bridge)`, averaged over answer positions, scored in four spaces
  (input embeddings and the untied `lm_head`, raw and mean-centred).
  - **At the first generated position it measures the opener, not the topic.** First-token
    entropy is **under one bit** (top-1 mass 0.675–1.000), so the sum collapses to
    `cos(argmax, bridge)` — and the argmax is always `'That'`, `'Sure'`, `'Making'`, `'Susp'`.
    `explain how suspension bridges work` scores *below* the uniform baseline. Phase 5 §3.6's
    result on a different objective.
  - **Over the whole answer it separates cleanly in all four spaces**, bridge queries ranked 1–2
    in every one. Two caveats: entropy is still sub-1-bit per position, so what fixed it was
    position coverage, not richer distributions; and the expected cosine equals the *realised*
    cosine to three decimals, making this close to a lexical measure with a differentiable
    surface.
  - **Steering the controls gives a clean dose–response, and exposes the metric's failure mode.**
    At s=0.6 the effect is *gated by whether a bridge metaphor fits the question* — "building
    bridges in a new city" for the friends query, nothing at all for the birthday one — and what
    the vector installs is the **metaphor, not the object** (never a river or a valley, unlike a
    genuine bridge answer). At s≥0.8 a degenerate loop scores **0.1749** against **0.0615** for
    a real bridge question: the measure rewards token density, not topical engagement. ⚠
    **Perplexity is inverted here** — the loops score 2.3–2.9 against an unsteered 5.4, because
    repetition is predictable. A distinctness ratio is the control that works.
  - ⁂ **The headline: GCG maximised the metric and moved only the space it was told to.**
    14 Optuna trials over trigger length (16–254), multi-slot mutation, pool, position and init;
    every trial clears the control band, none reaches a real bridge question, and trigger length
    never binds (a 233-slot trial scores below a 53-slot one). Scored in all four spaces, the
    winner sits **inside the unsteered control band in three of them**, and its answer is fluent
    meta-commentary about receiving junk. **This is phase 5's negative with every excuse closed:**
    the objective is not a proxy but the measure itself, it demonstrably separates bridge queries
    from controls, and its teacher-forced form tracks free generation to ~0.002. A good measure,
    honestly optimised, still yields none of the behaviour it measures.
- **Phase 8 — complete (2026-08-04).** Phase 2's mid-layer **twist lens** — the proposal scorer
  that beat the standard GCG gradient +0.348 to −0.192 on the 4B — carried over to phase 6's
  bridge problem on `Qwen3-8B`, together with the two controls `NEXT-STEPS.md` had been asking
  for. The rig reproduces phase 6 §2 to four decimals before anything new is measured.
  - ⁂ **The objective was never the problem.** Scoring the *hand-written phrases that work*
    under phase 6's metric — the comparison phase 6's own RECIPE required and never ran — puts
    all five in or above the bridge-query band in **all four** embedding spaces, with a
    topic-matched control, commas, `' the'` and random junk all inside the control band.
    Phase 5's objective ranked the same phrases **last** (−0.55). This removes the last excuse
    available to a search failure.
  - ⁂ **Random search matches GCG at equal compute.** Same accept test, same budget, same
    hyperparameters, only the proposer varying: metric gradient **0.0623**, twist lens 0.0601,
    **uniform random 0.0622** — against phase 6's tuned 0.0620. `pred_corr` is −0.106 for the
    metric gradient and −0.035 for the lens, so neither gradient proposes. "GCG maximised the
    metric" in phases 5–7 should be read as "a max over ~15k verified proposals reached 0.062".
  - ⁂ **The twist lens is reached, exceeded, and empty.** Optimised directly (157,696
    evaluations, 10× the other arms because a lens candidate skips the rollout and 28 layers),
    the projection goes 0.07 → **6.40** against the 100%-behaviour phrase's **2.30** — and the
    answer sits inside the control band in all four spaces with zero bridge words. It is not
    norm-gaming (‖h‖ +7%) and not a fluency shortcut: on the bridge-specific axis the optimised
    trigger leads the working phrase **0.326 to 0.149**. Search found *more* of the real
    direction and none of the behaviour.
  - **Within-family dose–response does not license an objective.** The direction was validated
    phase-2 style — degrade a working intervention, correlate alignment against behaviour — at
    r = +0.91, partial **+0.98**, peaking at **L6–L8** (phase 2's peak was L32; the depth does
    not transfer). That correlation is real and does not survive leaving the family.
  - **The working phrase's direction and the CAA steering vector are orthogonal** at every
    depth (cos −0.013 to +0.05), and only the first predicts behaviour — phase 5 §7's "two
    different internal routes" as a direct cosine.
  - **Guard (§7), reported as equivocal.** On a next-token objective in the identical rig the
    gradient *does* beat random (`p(' Sure')` 0.0019 vs 0.0007) so the §5 tie is not a plumbing
    artifact — but both arms reach only ~0.2%, far from phase 4's 0.99, so the scaffold is hard
    for GCG generally. ⚠ And `pred_corr` was **−0.033** in the arm the gradient won: it measures
    ranking *within* the top-k, not the filtering that carries the value. Every `pred_corr`
    reading in phases 2–8 inherits that correction.
  - **Fluent search (§8) does not reach the phrase band, and its ceiling control failed.**
    Word-pool and model-infill searches under the good objective produce the same four-space
    signature as the junk arms; fluency is free (log p −7.87 vs −13.38) and buys nothing. But the
    blocklist-off ceiling arm found `' pont'` and wrote it into a *book title* rather than a
    prompt injection, landing at 0.0658 — so this bounds the **search**, not the region. Single
    slot-level Gibbs cannot restructure a neutral sentence into a statement about the user, which
    is what the 100% phrase is. The fluent-region question is open.
- **Phase 9 — complete (2026-08-05).** The bridge target dropped entirely. If prompt space cannot
  reach a *specific* behaviour, can it reach the *easiest possible* one — decoherence, the
  complement of a narrow manifold rather than a point in it? Objective: mean answer-position
  entropy, gated on distinctness and a 4-gram repeat cap so it cannot be bought with the repetition
  loops phases 5–7 kept falling into, sampled at T=1.0 — greedy decoding converts word salad into
  loops before it can be measured, which reinterprets phase 6 §3's "degenerate loop at s=1.0".
  - ⁂ **No trigger found.** 42 hand-written triggers — control tokens, glitch tokens, divergence
    repetition, Unicode storms, both positions — span **0.54–1.19 bits** against a 10-sample
    baseline of **0.683** and an activation edit that reaches **4.80** on demand. Phases 5–8 said
    prompt space cannot reach a chosen behaviour; phase 9 adds that it does not reach the easiest
    one either, which points at the **channel** rather than at the objective.
  - ⁂ **Control tokens are inert**, falsifying the phase's own leading hypothesis. Phase 2 banned
    the added vocabulary because search would "steer" by breaking prompt structure; unbanned here,
    all 14 cells sit inside the normal band. `<|im_end|>` in a user turn just ends the turn.
  - ⁂ **Legibility is the antagonist — a length effect with the opposite sign to intuition.**
    Random junk ×4 / ×16 / ×53 gives 0.874 / **0.544** / 0.577. Past ~16 tokens the model
    recognises garbage and enters a fluent garbage-handling mode, so the meta-commentary attractor
    phases 6–8 kept hitting is *anti*-decoherence and it scales with trigger length.
  - ⚠ **The search manufactured four triggers that do not exist.** Its accept test maximised over
    rollouts resampled with a fresh seed — i.e. over sampling noise. A 10× replication collapsed
    4 of 5 headlines (3.022 → 1.097; 2.942 → **0.705**, below baseline). This is `NEXT-STEPS.md`'s
    standing seed-variance warning for phases 2–6, demonstrated rather than warned about.
    **Never accept on a max over resampled rollouts.**
  - ⁂ **The one survivor is a language hijack.** `'ممارسةקובע המציאות💒'` holds at 1.72 ± 0.42 with
    non-Latin output in 10/10 — but translated, nine of ten are grammatical Arabic prose
    explicating an invented aphorism, four of them then answering the original question anyway.
    The real damage is token-level speckling (`ONGODB`, `Именно`, `setState`). **Entropy alone
    cannot define decoherence on a multilingual model**: the cheapest way to raise it is to leave
    English, so every hill-climb finds that exit. The language control is owed.
  - ⁂ **§9, added post-hoc by reading the stored outputs rather than scoring them — two more
    exits and a blind spot.** *Enumeration* buys entropy for free: an answer listing ten
    near-synonyms (two of them outright duplicates in meaning) passes both gates at `distinct`
    0.714, because near-synonyms are distinct *tokens*. And **type C — confabulated premise** —
    outputs that invent the user's message, quote it, and answer the invention — scores *near
    baseline*, below an output whose only defect was a misspelling, because fabrication is
    fluent. **Entropy measures how well the model speaks, not whether it lost the plot.** The
    detector is ten lines and unbuilt: check whether the answer's quoted spans occur in the
    prompt. It also flags the *reading* mode in the same pass.
  - ⚠ **The token pool's uniformity has never been verified.** Three rare glyphs each recur in
    two independent optimised triggers; expected collisions under uniform sampling from 148023
    tokens is 0.039 across the 108 slots involved. Either the pool is weighted — which would
    reach back to phase 3 — or a search that was measuring its own noise "converged". One line
    to check, and the only open item that could invalidate earlier phases.
- **Phase 10 — designed, not run (2026-08-05).** Eight negatives now rest on one load-bearing
  claim — *prompt space cannot reach these behaviours* — and it is unsafe three ways over: the
  search that produced phase 9's version measured its own noise, the searches that ran correctly
  are indistinguishable from random (phase 8 §5), and **GCG has never succeeded at anything on
  `Qwen3-8B`** (`NEXT-STEPS.md` item 2 — every success in the programme is the 4B, every failure
  the 8B, and the confound has never been broken). "Prompt space cannot reach it" and "our search
  cannot find it" are still the same sentence.
  - **The instrument is the soft prompt.** A real-valued `P ∈ R^[k × 4096]` at the trigger slots,
    exact gradient descent, norm-projected each step. Every token sequence is representable as a
    soft prompt and most soft prompts are not token sequences, so it **upper-bounds anything GCG
    could achieve there**. Run against decoherence rather than `NEXT-STEPS.md` 3b's bridge target,
    because phase 9 §2 already pinned the activation-space ceiling at **4.800 bits on demand** —
    so a failure here is informative rather than ambiguous.
  - ⁂ **The result is the projection gap, not the bound.** Snap each optimised slot to its nearest
    token and re-measure: `H_soft − H_proj` *is* the cost of discreteness, quantified — the number
    the programme has been missing for five phases. Soft prompt fails → the channel claim is
    proven and it is the first positive-shaped result here. Projection collapses → every negative
    in phases 5–9 is about **discreteness**, not objectives or channels, and two headlines need
    rewriting. Projection survives → a reachable token trigger exists and nine phases missed it.
  - **The objective is repaired first**, or a `k × 4096` optimiser simply exploits phase 9's three
    known exits faster than GCG did: language-matched entropy baselines (the control phase 9
    owes), a type-token gate over embedding clusters rather than raw tokens, and the
    premise-fidelity flag. Registered predictions are in `phase10/README.md`; the risky one is
    that the projection **collapses**, which would make phase 9's channel headline wrong.
