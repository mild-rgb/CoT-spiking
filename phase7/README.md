# Phase 7 — a gradient target in activation space, and why prompt space cannot reach it

**Status: §1–§5 run (2026-08-03/04), A100 40GB, bf16, transformers 5.14.1, torch 2.11.0+cu128.**
`phase7_gradient_target_stub.ipynb` is the executed copy, outputs included;
`phase7_gradient_target.json` is the raw record (rig check, the four-space sweep with every
sample, the derivative check, the full iterated-ascent history, both stage-2 searches with
trajectories, and the granularity measurement).

Backbone `Qwen/Qwen3-8B`, thinking off, **no system message** — phase 6's exact configuration,
so every layer index and every reference band carries over. `h_L` throughout means the *input*
to `LAYERS[L]`, which is what phase 6's `steer_at` pre-hooks, so the two phases agree on what a
layer index denotes.

## The question

Phase 6 established that GCG maximises a good measure and produces none of the behaviour. The
proposal tested here is a different route to the same goal: run forward and backward but **stop
the backward pass at layer L**, read `grad = d(score)/d h_L`, form the target `h_L + eta*grad`,
and then search for tokens whose own `h_L` lands near it. Two stages instead of one, on the
theory that credit assignment over 36 layers is what has been producing a weak proposer
(pred_corr −0.192 / +0.091 / +0.055 across phases 2 / 4 / 5).

Rather than build the token search first, §1 writes the target in **directly**. An arbitrary
activation edit is strictly more expressive than any soft prompt, which is strictly more
expressive than any token sequence, so the edit upper-bounds what the search could reach.

## §0 — the rig reproduces phase 6 exactly

Before anything else, phase 6 §2's six queries were rerun, greedy 160 tokens, all four spaces:

| query | in.raw | in.cent | out.raw | out.cent |
|---|---|---|---|---|
| what shall i do today | 0.0368 | 0.0198 | −0.0216 | 0.0546 |
| recommend me a book | 0.0359 | 0.0189 | −0.0181 | 0.0516 |
| how do I make friends in a new city? | 0.0392 | 0.0222 | −0.0158 | 0.0499 |
| what should I get my brother for his birthday? | 0.0405 | 0.0232 | −0.0178 | 0.0534 |
| tell me about bridges | 0.0549 | 0.0377 | 0.0137 | 0.0864 |
| explain how suspension bridges work | 0.0782 | 0.0615 | 0.0379 | 0.1025 |

All 24 numbers match phase 6 to four decimals. The pool guard also reproduced (151936 → 148023
usable, 659 blocked), as did the CAA vector at L16 (‖v‖ 69.4 single-pair mean → 50.0 for the CAA
mean, pairwise cosine 0.452).

Reference points used throughout, all `out.cent`: uniform −0.0002 · control band 0.0499–0.0546 ·
phase 6's GCG winner 0.0620 · real bridge query 0.0864–0.1025.

The trigger is a **random** 53-token draw from the pool (seed 1, suffix position, query
`what shall i do today`), never optimised — this phase asks what the gradient points at from an
arbitrary starting point. Unedited baseline: **0.0537** sampled (T=0.8, 45 tokens, n=3),
**0.0551** greedy.

## §1 — the edit is written in directly, and nothing happens

Injected at the 53 trigger slots on prefill passes only — the slots are prompt positions, so
during decode they live in the KV cache and must not be touched again. Strength `s` is the added
vector's norm as a fraction of the mean non-sink `‖h_L‖` at that layer, phase 6 §4's
parameterisation. Evaluation at T=0.8 / 45 tokens per phase 6 RECIPE stage 4.

Mean non-sink `‖h_L‖`: **29.94 / 58.14 / 85.11 / 200.57** at L4/8/16/24.

`out.cent`, gradient direction:

| L | s=0.1 | s=0.2 | s=0.4 | s=0.8 | s=1.6 |
|---|---|---|---|---|---|
| 4 | 0.0567 | 0.0569 | 0.0558 | 0.0507 | 0.0521 |
| 8 | 0.0569 | 0.0573 | 0.0573 | 0.0561 | 0.0538 |
| 16 | 0.0555 | 0.0534 | 0.0565 | 0.0562 | 0.0525 |
| 24 | 0.0571 | 0.0568 | 0.0542 | **0.0614** | 0.0597 |

**The whole sweep spans 0.0507–0.0614 against a 0.0537 baseline, with no dose–response** — at
s=1.6, an injected vector 1.6× the residual stream's own norm, the score falls. Distinctness
stayed 0.74–0.89 throughout; every answer is fluent meta-commentary about receiving junk.

**The best gradient cell has phase 6's GCG signature.** Scored in all four spaces:

| | in.raw | in.cent | out.raw | out.cent |
|---|---|---|---|---|
| best gradient (L24, s=0.8) | 0.0364 | 0.0188 | −0.0230 | 0.0614 |
| control band | 0.0359–0.0405 | 0.0189–0.0232 | −0.0216…−0.0158 | 0.0499–0.0546 |

Inside the control band in three of four spaces, moving only the space the objective is defined
in — phase 6 §7 reproduced by an activation edit, with no discrete search involved at all.

### §1b — the channel control

Phase 6's CAA bridge vector was run through the identical channel (same layers, same strengths,
same slots, same prefill-only injection) for one purpose: to distinguish *this direction is
useless* from *injecting at 53 slots cannot move anything*.

| L | s=0.1 | s=0.2 | s=0.4 | s=0.8 | s=1.6 |
|---|---|---|---|---|---|
| 4 | 0.0535 | 0.0565 | 0.0643 | 0.0765 | **0.1121** |
| 8 | 0.0544 | 0.0544 | 0.0753 | **0.0984** | 0.0939 |
| 16 | 0.0534 | 0.0567 | 0.0533 | 0.0555 | **0.0965** |
| 24 | 0.0537 | 0.0536 | 0.0549 | 0.0536 | 0.0547 |

The channel is not the limitation. Two things about these numbers, both important:

1. **They should not be read as behaviour.** The best cell's output is the model reporting that
   the trigger is *"the different spellings and variations of the word bridge"* — fluent
   (distinct 0.71) but lexical density in meta-commentary. §3 makes this worse: a comma scores
   0.0990 on this scale, so "reaches bridge range" is not by itself a claim about anything.
2. `cos(V_CAA, grad)` is **0.0012 / 0.0026 / −0.0053 / 0.0067** at L4/8/16/24. The metric's own
   gradient is orthogonal to the direction that carries the topic, at every depth.

The two directions have **opposite depth profiles** — CAA works shallow and dies at L24, the
gradient does the reverse (see §2).

## §2 — the gradient is correct; there is simply nothing along it

§1 regenerated after editing, which is not the objective the gradient was computed for. On a
**frozen** answer the prediction is exact calculus: `d(score) = eta * SUM_i ||g_i||^2`. Swept with
`−eta` and a matched-per-position-norm random direction as a null.

| L | ratio at rel=0.003 | −grad | rand null |
|---|---|---|---|
| 4 | 0.93 | 1.12 | 2.8e−06 |
| 8 | 0.77 | 1.05 | 6.8e−06 |
| 16 | 0.91 | 1.07 | 8.3e−06 |
| 24 | 1.08 | 0.56 | 6.7e−05 |

Repeat scoring differs by exactly `0.000e+00`. `+grad` rises, `−grad` falls, measured matches
predicted within ~10% at the smallest step, the random null sits 2–3 orders of magnitude below.
**The hook, the truncated backward, the sign and the scaling are all correct**, so §1's flatness
is a result and not broken plumbing.

**But the gain saturates.** L16 ratio across rel 0.003 → 0.3: **0.91 → 0.62 → 0.36 → 0.14 →
0.04**, with the absolute gain plateauing and turning over. Maximum achievable:

| L | max gain | vs the +0.0313 needed for a real bridge question |
|---|---|---|
| 4 | +0.0021 | 15× short |
| 8 | +0.0029 | 11× short |
| 16 | +0.0036 | 9× short |
| 24 | **+0.0070** | 4× short |

`−grad` falls consistently more than `+grad` rises (L16 at rel=0.1: +0.0036 against −0.0092) —
strong negative curvature, i.e. the current state sits near a local ridge along `grad`.

**Depth inverts the proposal's premise.** L24 keeps its linearity furthest (ratio 0.72 at rel=0.1
where L4 is at 0.11) and yields the largest gain. Autograd is exact wherever it is stopped;
truncating the backward pass does not improve the gradient. What truncation changes is the radius
over which the linearisation is valid, and that shrinks the *lower* you go, because more
nonlinearity sits between the edit and the readout.

Note this says nothing about `pred_corr`. That measures a **different** linearisation — the
one-hot relaxation from tokens to embeddings — which §2 does not touch, and which a two-stage
scheme would still need.

Consistency check: §1's best fresh-rollout cell was L24 s=0.8 at 0.0614, i.e. +0.0077 over
baseline; §2's frozen maximum at L24 is +0.0070. Two independent measurements within 0.0007.

## §3 — iterated ascent climbs, and pays for it in degeneracy

Re-linearising at each new point is a strictly stronger method than one step along a gradient
computed once. 40 steps of 1% of `‖h‖`, ascending on the current rollout, rollout refreshed every
5 steps, teacher-forced and regenerated scores reported side by side per phase 6 RECIPE stage 5.

| | best FLUENT (distinct ≥ 0.45) | step | dist | best ANY | step | dist |
|---|---|---|---|---|---|---|
| L4 | 0.0613 | 25 | 0.80 | 0.0613 | 25 | 0.80 |
| L8 | 0.0607 | 35 | 0.62 | 0.0607 | 35 | 0.62 |
| L16 | **0.0634** | 25 | 0.76 | 0.0720 | 40 | 0.33 |
| L24 | 0.0519 | 5 | 0.91 | **0.1041** | 40 | 0.04 |

Best fluent anywhere: **0.0634**, i.e. **+0.0083** over baseline and only **+0.0020** over the
single step. Still 3.8× short. L4 and L8 stayed fluent throughout and plateaued exactly where the
single step did.

Everything above ~0.063 was bought with degeneracy:

| L | step | true | distinct | regenerated answer |
|---|---|---|---|---|
| 16 | 40 | 0.0720 | 0.33 | `The "noumena" (noumena in the "noumena" (noumena in` |
| 24 | 20 | 0.0990 | 0.02 | `,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,` |
| 24 | 40 | 0.1041 | 0.04 | `,11111111111111111111111111111` |

### ⁂ §3b — a comma outscores a real bridge question

The degenerate scores are not approximately anything. They are exact:

| token | `out.cent` cosine to `' bridge'` |
|---|---|
| `' bridge'` | +1.0000 |
| `'1'` | **+0.1041** |
| `','` | **+0.0990** |
| `' the'` | +0.0892 |
| *[uniform baseline]* | −0.0002 |
| *(for comparison)* `explain how suspension bridges work` | 0.1025 |
| *(for comparison)* `tell me about bridges` | 0.0864 |

With sub-1-bit entropy the metric collapses to the cosine of the token actually emitted (phase 6
finding #3), so a comma loop scores precisely `cos(',')`. **A comma is more bridge-like under this
measure than `tell me about bridges`.** So is `1`. So is ` the`, nearly.

This is phase 6 finding #4 with the teeth showing. Phase 6 demonstrated a *bridge-saturated*
repetition loop outscoring a genuine bridge answer, which reads as a measure that over-rewards
on-topic density. §3b shows the topic is not required at all: **the band every phase 6 result is
compared against is reachable by punctuation.** The control band separates bridge queries from
non-bridge queries only among fluent English answers; it has no floor against filler.

Two supporting details. The cumulative edit never exceeded `‖D‖/‖h‖ = 0.09`, so this is a small
perturbation finding a cheap exploit, not a state wandering off-distribution. And the ascent found
it unprompted — there is no repetition incentive anywhere in the objective.

## §4 — stage 2: GCG cannot reach the target

The search proper, at last. Target `H* = h_8(TRIG)[slots] + D`, with `D` the cumulative edit from
§3's L8 ascent at steps 35 and 40. Because attention is causal, `h_8` at the trigger slots depends
only on `PRE + trigger`, so each candidate costs a 62-token forward and nothing more.

Readout is the **relative residual**, mean over slots of `||h_8(T') − H*|| / ||D||`:
**1.0 = the original trigger** (it realises no delta at all), **0.0 = hit exactly**.

| target | start | after 151 GCG steps | found trigger, no injection |
|---|---|---|---|
| L8 step 35 | 1.0000 | **1.0000** | 0.0551 (= baseline) |
| L8 step 40 | 1.0000 | **1.0000** | 0.0551 (= baseline) |

`n_mut=7`, `n_cand=256`, `n_top=512`, 240 s each — roughly **38,600 mutations per run, none better
than doing nothing.** The returned trigger is the original, unchanged.

Note the starting `state_cos` of **0.9914**: raw hidden-state cosine between two entirely
different triggers in the same scaffold, before any search. Phase 5's context floor, reproducing
here, and the reason the readout is a residual on the *delta* rather than any cosine on the state.

## §5 — why it cannot: direction, not granularity

The obvious hypothesis is that token substitution is too coarse to place a vector of 9% of `‖h‖`.
**It is wrong.** Target `||D||_F = 57.340`, mean per-slot `||D_i|| = 5.119`, against `‖h‖ = 58.14`:

| family | ‖Δ‖/‖D‖ min | median | rel_residual min |
|---|---|---|---|
| **nn-1** (each slot → nearest in-pool embedding neighbour) | **0.43** | 0.66 | 1.163 |
| random-1 | 0.79 | 1.23 | 1.238 |
| gcg-1 (from the match gradient's top-512) | 0.92 | 1.20 | 1.250 |
| random-7 | 2.89 | 3.25 | 4.505 |
| gcg-7 | 2.84 | 3.17 | 3.783 |

Moves of the right size are plentiful — a single swap moves `h_8` by about `‖D‖`, a
nearest-neighbour swap by less than half. (Those neighbours are morphological:
`'نه' → ' إنه'`, `'ellaneous' → ' Miscellaneous'`.) Nothing lowered the residual below 1.0.

It is direction. `cos(Δ, D)` over the flattened `[53, 4096]` block:

| family | max | p99 | median |
|---|---|---|---|
| nn-1 | 0.0063 | 0.0056 | 0.0005 |
| random-1 | 0.0184 | 0.0117 | 0.0007 |
| gcg-1 | 0.0182 | 0.0146 | 0.0011 |
| random-7 | 0.0151 | 0.0099 | 0.0017 |
| **gcg-7** | **0.0206** | 0.0131 | 0.0039 |

Best alignment anything achieves: **0.0206**, against a random-vector baseline of
`1/sqrt(217088) = 0.00215`. Token moves are ~10× better than chance and still, in every practical
sense, orthogonal.

**The objective was unimprovable, not merely hard.** Since
`||Δ − D||^2 = ||Δ||^2 − 2<Δ,D> + ||D||^2`, improving on `Δ = 0` requires

```
cos > ||Δ|| / (2||D||)
```

| family | cos needed | available (p99) | |
|---|---|---|---|
| nn-1 | 0.329 | 0.0056 | 59× short |
| random-1 | 0.613 | 0.0117 | 52× short |
| gcg-1 | 0.601 | 0.0146 | 41× short |
| random-7 | **1.623** | 0.0099 | impossible by definition |
| gcg-7 | **1.585** | 0.0131 | impossible by definition |

The 7-slot families require a cosine above 1. No search, budget or seed could have moved them.

**More slots make it worse.** Slot contributions are near-orthogonal, so norms add in quadrature —
predicted `1.23 * sqrt(7) = 3.25` against a measured 3.17 — while the threshold scales *with*
`‖Δ‖`. Composing moves is strictly counterproductive under this objective, which closes the
obvious escape route.

Why the cosines are that small: `D` is defined by **downstream** sensitivity (layers 8→36 and the
readout), `Δ` is produced by **upstream** computation (embeddings and layers 0→8). Nothing couples
them. The measured 10×-over-chance is the entire coupling that exists.

## §6 — trigger length does not help (analysis, not measurement)

Derived from §5's quantities rather than separately measured, and stated here because it settles
a design question the phase would otherwise invite.

The best cosine achievable by searching `N` candidates against a target in `d` dimensions goes as
`sqrt(2 ln N / d)`. Scaling the trigger to `k` slots:

- search space `N = 148000^k`, so `ln N = 11.9k`
- target dimension `d = 4096k`, because `D` lives on all `k` slots

```
cos_max ~ sqrt(2 * 11.9k / 4096k) = sqrt(2 * 11.9 / 4096) = 0.076
```

**`k` cancels.** Each added slot buys 11.9 nats of search and costs 4096 dimensions of target, and
these trade off exactly. A 16-slot trigger and a 254-slot trigger have the same ceiling — and
0.076 is the *exhaustive* ceiling, against a requirement of 0.329. This is very likely the
explanation for phase 6 §5's unexplained observation that trigger length never ordered the
results and the 254-token budget was never binding.

Calibration: for §5's ~2,100 sampled moves the formula predicts a best of 0.0084; measured was
0.0206, so it is right in shape and ~2.5× conservative (the gradient earning its keep).

**The cancellation breaks if the target stops growing with the trigger.** Search over `k` slots but
constrain only `m`:

```
cos_max ~ sqrt(0.0058 * k/m)   >  0.329   requires   k/m > 19
```

About nineteen free slots per constrained slot — so constraining 4 positions and searching over
~75 is comfortably inside the budget already in use, whereas constraining all 53 would need ~1000.
Two design constraints follow: attention is causal, so the free slots must sit **upstream** of the
constrained ones (free slots after them contribute nothing); and the free slots are not inert —
they remain in the prompt and shape the answer, so a behavioural readout stays the arbiter.

All of §6 concerns the L2 exact-state objective and its threshold. The projection objective
`maximise <Δ, D>` has **no threshold**, so extra slots pay off there directly, with no cancellation.

## What phase 7 establishes

1. **The truncated-gradient target is real, correctly computed, and nearly empty.** The
   directional-derivative check confirms the plumbing exactly (ratio 0.77–1.08 at the smallest
   step, sign flips, random null 2–3 orders below), and the maximum available gain is +0.0021 to
   +0.0070 against the +0.0313 needed.
2. **Iterating does not rescue it.** Re-linearising 40 times buys +0.0020 over a single step.
   Everything beyond that is degeneracy.
3. **Depth inverts the proposal's premise.** The gradient is *more* usable deeper, not shallower;
   truncation does not improve gradient quality, it shrinks the radius over which the
   linearisation holds.
4. **The gradient is orthogonal to the direction that works.** `cos(grad, V_CAA)` ≤ 0.007 at every
   depth, and the two have opposite depth profiles.
5. **⁂ A comma scores 0.0990 under phase 6's metric**, above `tell me about bridges` at 0.0864.
   The "real bridge question" band has no floor against filler, and an optimiser finds this
   unprompted with a perturbation of 9% of `‖h‖`.
6. **⁂ Prompt space cannot hit an activation target of this kind, for geometric reasons.** Not
   search weakness: the L2 objective requires `cos > 0.329` where the best of 2,100 moves achieves
   0.0206, and the multi-slot families require `cos > 1`. Moves of the right *size* are plentiful;
   alignment is the binding constraint.
7. **Trigger length cancels out** — search grows as `ln N ∝ k` while the target grows as `d ∝ k`,
   leaving a scale-invariant ceiling of 0.076. Decoupling controlled from constrained slots is the
   version that helps, at ~19:1.
8. **An activation edit reproduces phase 6's four-space GCG signature** — the best gradient cell
   sits inside the control band in three of four spaces and moves only the optimised one, with no
   discrete search anywhere in the pipeline. That locates the pathology in the objective rather
   than in GCG.

## Open

- **The projection objective was never run.** `maximise <Δ, D>` has no threshold, every
  positive-cosine move contributes, and the median cosine is positive in all five families. Rough
  arithmetic says ~100 accumulating 1% contributions would be needed from 53 interacting slots —
  short, but short by a factor rather than by geometry. This is the first thing to run.
- **The L24 degenerate targets were held.** Under L2 they fail identically (same `‖D‖`, same
  geometry). Under the projection objective, whether prompt space can reach a comma-emitting state
  is the sharpest remaining question in the phase — it would convert phase 6's negative into a
  positive Goodhart demonstration.
- **One query, one trigger seed, one target word**, exactly as in phase 6. Everything here is
  `what shall i do today` with seed 1 against `' bridge'`.
- **The §1b CAA arm is a channel control, not a finding.** Its headline number is precisely the
  kind of figure §3b shows cannot be interpreted.
- **`pred_corr` under truncation is untested.** §2 measured the activation-space linearisation, not
  the one-hot relaxation. A two-stage scheme would still need the latter.
- **NEXT-STEPS item 3 remains unrun and is now more valuable**: if a comma scores 0.0990, where the
  hand-written working phrases land under this metric decides whether the phase 6 objective means
  anything at all.

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `phase7_gradient_target_stub.ipynb` | executed, outputs included, 21 cells |
| `phase7_gradient_target.json` | rig check · 41 sweep rows · 60 derivative rows · full 40-step ascent history × 4 layers · both stage-2 searches with trajectories · granularity families with norms and cosines |
