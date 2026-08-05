# Recipe — the soft-prompt upper bound on decoherence

The protocol for phase 10, written before the run. Phase 6's `RECIPE.md` governs stage 0
(runtime) and the three-numbers discipline; phase 9's governs the accept test (its stage 6) and
the read-the-outputs habit (its stages 7–8). What follows is what phase 10 adds.

**Stages 0–2 are cheap and each can independently invalidate what comes after. Run them in
order and stop if one fails.**

---

## Stage 0 — rig check, plus the one line that could invalidate phases 3–8

Reproduce phase 9 §0 exactly: six queries, greedy 160, `H` = 0.703 / 0.527 / 0.743 / 0.730 /
0.639 / 0.648 and `distinct` = 0.588 / 0.670 / 0.638 / 0.575 / 0.663 / 0.644. Pool guard
151936 → **148023**. L16 CAA vector `‖V‖` 49.98, pairwise cosine 0.452, mean non-sink
`‖h_L16‖` 89.77.

**New, and first:** verify the pool is uniform. Phase 9 §9 found three rare glyphs each recurring
in two independent optimised triggers — expected 0.039 collisions across 108 glyph slots,
P(≥3) ≈ 1e-5.

```
draw 10^5 tokens from the pool with the sampler the searches actually use
→ count distinct, plot the rank-frequency curve
```

Uniform gives ~85k distinct and a flat curve. Anything else means every phase back to phase 3 has
been sampling from a smaller effective vocabulary than it reported, and that finding outranks the
rest of this phase.

*Caveat on the evidence:* the "rare glyph" filter that produced P ≈ 1e-5 was chosen after seeing
the collisions, and it excluded Hebrew and Arabic letters — which is where the most common repeats
would have been. Treat 1e-5 as a reason to check, not as a measured p-value.

---

## Stage 1 — repair the objective before giving it a stronger optimiser

Phase 9's readout has three demonstrated exits. A soft prompt will find all three faster than GCG
did. Close them first, and **measure each repair's effect on the phase 9 numbers** so the new
objective stays comparable.

### 1a. The language control — run it before anything else uses entropy

Phase 9 owes this and never ran it. Score a **fluent, on-topic, forced-language** answer to the
clean query, no trigger, in each language the phase has seen:

| language | how | expected |
|---|---|---|
| English | baseline, unforced | 0.683 (measured) |
| Arabic | system-free forcing, e.g. prefill `أعتقد` | ? |
| Hebrew | prefill `אני חושב` | ? |
| Chinese | prefill `我觉得` | ? |

If forced-Arabic lands near **1.7**, the surviving trigger of phase 9 §8 has no effect beyond
language switching and should be reported as zero.

**The repair:** score entropy against the **language-matched baseline**, not the English one —
`H_excess = H − H_baseline(detected_language)`. `nonlatin` and `switch` were computed throughout
phase 9 and never entered the objective; this is what they are for.

### 1b. The enumeration gate

Phase 9 §9: ten near-synonyms score high at `distinct` 0.714, and neither the type-token gate nor
the 4-gram cap fires. **Gate over meaning, not tokens** — cluster answer tokens by embedding
cosine (or lemma, for the languages where that is easy) and compute the type-token ratio over
*clusters*. A synonym run then counts as repetition, which is what it is.

Calibrate on the phase 9 texts, which are stored: the k=53 enumeration answer must fail the new
gate, and the six normal answers of §0 must pass it.

### 1c. The premise-fidelity detector

Phase 9 §9's type C — outputs that fabricate the user's message and answer the fabrication —
scores *near baseline* because it is fluent. Ten lines, language-independent:

```
for each quoted span in the answer:
    present_in_prompt = span appears in the user turn (normalised)
→ confabulated  if any span is absent
→ reading mode  if a span is present
```

Report it as a **flag alongside** the objective, not a term inside it. It is a different failure
than the one being optimised, and folding it in would make the number uninterpretable again.

### 1d. Code the modes

Phase 9 has ten stored outputs per trigger and one mean. Label each sample **normal / L / S / C**
and report the distribution. Free, and it turns bimodal means into something readable — the k=4
search winner's 1.097 mean hides one genuine type-S sample among nine near-baseline ones.

---

## Stage 2 — the positive control: does GCG work on this backbone at all?

`NEXT-STEPS.md` item 2, unrun since 2026-08-03, and it interprets every negative in the
programme. Every success in the project is `Qwen3-4B-Thinking`; every failure is `Qwen3-8B`.

Target something GCG *should* hit trivially in the phase 6–9 scaffold: force a chosen first
answer token to high probability, or port phase 2's animal-token objective. ~10 min.

- **Succeeds** → the negatives are properly about objectives and channels, and phases 5–9 stand.
- **Fails** → the programme has been measuring its search, not its model, and phase 10's soft
  prompt becomes the *only* trustworthy instrument in it.

Phase 8 §7 is partial prior evidence for success (`p(' Sure')` 0.0019 gradient vs 0.0007 random
in this exact rig) — but at 0.19% absolute against phase 4's 0.99, so the scaffold is hard for
GCG generally and this control is not a formality.

---

## Stage 3 — the accept test, rebuilt

Phase 9's stage 6, restated because it is the single easiest way to waste this phase:

**Never accept on a max over resampled rollouts.** Either

1. accept on a **mean over n ≥ 5 rollouts with fixed seeds** (costs n× per test, buys a number
   that means something), or
2. accept on the **teacher-forced score alone**, deterministic given the trigger, and use fresh
   rollouts only as a separately-reported true score.

For the soft prompt, option 2 is natural: teacher-forced entropy is differentiable and the true
score is a separate measurement. **Report three numbers throughout** — teacher-forced,
true (sampled, fixed seeds, n ≥ 5), and greedy — per phase 6's discipline.

**Replicate every headline at 10× before writing it down.** ~80 s. Phase 9's entire correction
came from one spot-check that nearly was not run.

---

## Stage 4 — the soft prompt

The phase's reason to exist. `P ∈ R^[k × 4096]` injected at the trigger slots via `inputs_embeds`,
optimised by Adam against the repaired objective.

**Differentiable objective.** Sampling is not differentiable, so: generate a rollout, compute
**teacher-forced** answer-position entropy under the current `P`, backpropagate to `P`, refresh
the rollout every `m` steps with a **fixed seed schedule** (stage 3). The rollout refresh is the
same structure phases 6 and 9 used; the fixed seeds are what phase 9 lacked.

**Constraints that decide whether the result means anything:**

- **Norm.** Project each slot onto the embedding-norm shell every step. Without this the optimiser
  wanders off-distribution and the bound becomes vacuous — an unconstrained `P` can do anything and
  proves nothing about tokens.
- **Two inits, run separately.** Random, and the embeddings of phase 9's surviving trigger. The gap
  between them characterises the landscape independently of discreteness.
- **Three lengths**, k = 4, 8, 16, matched step count. Phase 9's length questions are unanswerable
  in token space (its sweep measured noise) but they are cleanly answerable here, because the soft
  prompt has no search-budget confound at all — the gradient touches every slot every step.
- **Position.** Prefix and suffix, both. Phase 9's prefix-beats-suffix prediction is likewise
  unanswerable in token space and cheap here.

**Reference points**, all on the repaired objective: clean baseline (stage 1a, per language) ·
the phase 9 hand-written battery ceiling (1.189 raw) · the L16 activation edit at s=1.0
(4.800 raw) — the activation-space ceiling and the number the soft prompt is trying to approach
from the input side.

---

## Stage 5 — ⁂ the projection gap is the actual result

Snap each optimised slot to its nearest token — by embedding cosine **and** by L2, reported
separately, because they disagree and the disagreement is informative — then re-measure with the
full three-number protocol.

```
soft prompt reaches H_soft
its projection reaches H_proj
gap = H_soft − H_proj   ← this is the cost of discreteness, quantified
```

This is the number the programme has been missing for five phases. It converts "GCG failed" from
an anecdote into a measurement:

- **`H_proj` collapses** → discrete search was never viable on this objective, and every negative
  in phases 5–9 is about *discreteness*, not about objectives, channels or Goodharting.
- **`H_proj` survives** → discreteness is not the bottleneck, a reachable token trigger exists,
  and nine phases of search missed it. Then replicate it 10× immediately (stage 3) before
  believing it, and read the outputs (stage 6) before reporting it.
- **`H_soft` itself never rises** → the channel claim is proven, and it is the programme's first
  positive-shaped result.

Also snap and re-measure at *intermediate* points along the optimisation trajectory, not only at
the end. If the projection gap widens monotonically with `H_soft`, that is a stronger statement
than a single endpoint pair: it says the two spaces diverge as you push, which is exactly the
shape "discreteness is the bottleneck" predicts.

---

## Stage 6 — read the outputs, twice

Phase 9's stages 7 and 8, which are not optional and cost nothing:

1. **Read every headline output, in the language it is in.** Phase 9's one surviving trigger
   looked like a result and turned out to be an Arabic aphorism politely explicated. A number with
   an unread output behind it is not yet a finding.
2. **Read them again for what the objective did not measure.** Phase 9 found enumeration, type C
   and within-trigger mode variance this way, after the numbers were already written down.

Run the stage 1c detector and the stage 1d mode coding over everything the phase generates, and
report the mode distribution beside every mean.

---

## What would make this phase fail quietly

Listed so it can be checked against afterwards.

- **An unconstrained soft prompt.** Reaches anything, proves nothing. Norm-project every step.
- **Optimising the unrepaired objective.** The soft prompt exits via language within ~50 steps and
  the phase re-discovers phase 9 §8 with more compute.
- **A single projection measurement at the endpoint.** The trajectory carries the information.
- **Reporting `H_soft` as a headline.** It is a bound, not a trigger. The result is the *gap*.
- **Skipping stage 2** because it is boring. It is the control that tells you whether any of the
  programme's negatives were ever about the model.
