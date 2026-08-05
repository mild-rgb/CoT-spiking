# Phase 8 — the twist lens on the bridge problem

**Status: complete (2026-08-04/05), A100 40GB, bf16, transformers 5.14.1, torch 2.11.0+cu128.**
`twist_lens_bridge_stub.ipynb` is the executed copy, outputs included, 24 cells;
`phase8_twist_lens.json` is the raw record.

Backbone `Qwen/Qwen3-8B`, thinking off, **no system message** — phase 6/7's exact configuration,
so every layer index and every reference band carries over. Query `what shall i do today`,
53 suffix slots, blocklist on.

## The question

Phase 2 found the textbook GCG gradient *anti*-predictive as a proposer (predicted-vs-realised
correlation **−0.192** across 8 animals) and a **twist lens** — a reduced Jacobian read along
`DELTA_MID = h_mid(real cue) − h_mid(neutral cue)` — clearly better (**+0.348**, positive in
7/8). Phases 5–7 then produced six negatives on `Qwen3-8B`/` bridge` using gradients of the
objective itself, and phase 7 §1b found the reason that might matter: `cos(V_CAA, grad_metric)
≤ 0.007` at every depth. **The metric's gradient is orthogonal to the direction that carries
the topic.** The twist lens is the gradient of a projection *onto* that direction.

Reference points, `out.cent`: uniform −0.0002 · control band 0.0499–0.0546 · phase 6's GCG
winner 0.0620 · real bridge query 0.0864–0.1025.

## §0 — the rig reproduces phase 6 exactly

Phase 6 §2's six queries, greedy 160, all four spaces: **all 24 numbers match to four
decimals** (0.0546 / 0.0516 / 0.0499 / 0.0534 for the controls, 0.0864 / 0.1025 for the bridge
queries in `out.cent`). The pool guard reproduced too — 151936 → 148023 usable, 659 blocked —
as did the L16 CAA vector (‖v‖ 69.4 single-pair mean → 50.0 for the CAA mean, pairwise cosine
0.452).

## §1 — the working phrase's direction is not the steering vector's

Two directions, both CAA means over 8 negative arms with a shared context prefix:

- `V_CAA[L]` — word level, `"The word is bridge"` − `"The word is {other}"`, phase 6's vector
  rebuilt at every layer.
- `D_PH[L]` — phrase level, **in the 53-slot scaffold**: `' the user really loves bridges'`
  (phase 5's 100% intervention) against topic-matched controls, read at the last prompt
  position.

| L | 4 | 8 | 16 | 24 | 32 | 36 |
|---|---|---|---|---|---|---|
| ‖V_CAA‖ | 20.3 | 37.3 | 50.0 | 128.2 | 386.6 | 46.5 |
| ‖D_PH‖ | 0.7 | 1.4 | 4.3 | 20.0 | 84.1 | 19.5 |
| **cos(V_CAA, D_PH)** | **−0.002** | **−0.013** | **0.014** | 0.005 | 0.049 | 0.121 |
| ‖D_PH‖ / ‖h‖ | 0.026 | 0.026 | 0.049 | 0.102 | 0.125 | 0.124 |

**The two are orthogonal at every depth.** The displacement the working phrase actually causes
at the readout position and the CAA steering vector share essentially nothing — this is phase 5
§7's "the intervention that reaches 100% matches *worst* at every depth" seen as a direct
cosine. `‖D_PH‖ = 0` at L0 confirms the readout is measuring only what attention carries
forward from the trigger slots.

## §2 — machinery

No findings; phase 6's pool, blocklist, rollout and metric unchanged, plus the lens readout.
The lens is a pre-hook on `LAYERS[L]` that captures the activation and raises, aborting the
forward pass — so a lens candidate costs `L/36` of a pass and no rollout at all. Measured on
256 candidates at k=53: **0.7 s against the metric's 2.7 s**, gradient 0.13 s against 0.30 s.
See `RECIPE.md` stage 4.

## §3 — ⁂ the objective ranks the working phrases correctly, in all four spaces

**NEXT-STEPS item 3, and phase 6 RECIPE stage 6's unrun requirement.** Greedy 160, four spaces:

| intervention | in.raw | in.cent | out.raw | out.cent | dist |
|---|---|---|---|---|---|
| `' I should mention bridges'` (77.8% on-topic) | 0.0605 | 0.0435 | 0.0373 | **0.1074** | 0.66 |
| `' bridges, obviously'` (80.6%) | 0.0629 | 0.0460 | 0.0277 | **0.0984** | 0.67 |
| **`' the user really loves bridges'` (100%)** | 0.0503 | 0.0332 | 0.0289 | **0.0896** | 0.68 |
| `' that bridges are the answer'` (66.7%) | 0.0599 | 0.0432 | 0.0162 | **0.0870** | 0.64 |
| `' bridge'` (48.6%) | 0.0549 | 0.0381 | 0.0114 | **0.0824** | 0.70 |
| `' the user really loves puzzles'` (control) | 0.0367 | 0.0196 | −0.0179 | 0.0494 | 0.68 |
| commas × 8 | 0.0375 | 0.0200 | −0.0185 | 0.0540 | 0.56 |
| `' the'` × 8 | 0.0384 | 0.0211 | −0.0215 | 0.0554 | 0.61 |
| random 8 junk tokens | 0.0325 | 0.0151 | −0.0200 | 0.0567 | 0.53 |
| *[control queries]* | 0.036–0.041 | 0.019–0.023 | −0.022…−0.016 | 0.050–0.055 | |
| *[bridge queries]* | 0.055–0.078 | 0.038–0.061 | 0.014–0.038 | 0.086–0.103 | |
| *[phase 6 GCG winner]* | 0.0386 | 0.0216 | −0.0212 | 0.0620 | 0.78 |

**Every working phrase lands in or above the bridge-query band in all four spaces; every floor
lands in the control band.** `out.raw` separates by sign exactly as it does for genuine bridge
queries — all five phrases positive, all four floors negative. The topic-matched control
(`puzzles`) scores 0.0494, *below* the control band floor.

This closes phase 5's excuse for real. Phase 5 §6 found its objective ranked the working
phrases **last** (rank correlation −0.55, and −0.60 among the human phrases alone); phase 6's
metric ranks them **first**, and does so in four spaces at once rather than in the one it is
optimised in. Whatever the searches below fail at, they do not fail because the measure is
pointed the wrong way.

Two details worth carrying. The **strongest metric score is not the strongest behaviour** —
`' I should mention bridges'` (77.8% on-topic in phase 5) outscores the 100% phrase 0.1074 vs
0.0896 — so the ordering is right at the band level and noisy within it. And only the 100%
phrase produces the **concrete** sense (6 concrete words to 1 metaphorical: *"celebrate
bridges — both literal and metaphorical"*, then bridge walks and photography); the weaker
phrases produce phase 6 §3's metaphor (*"connect places, ideas, and even people"*).

## §4 — the depth the lens should read at

Phase 2 §4's degradation ladder, transplanted: revert *j* of the phrase's 5 slots to random
pool tokens (j = 0..5, 3 reps), score behaviour, and correlate against
`cos(h_L(trigger) − h_L(neutral), d)` at each depth. Both raw and residualised on *j*:

| L | 2 | 4 | **6** | **8** | 12 | 16 | 20 | 24 | 28 | 32 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r(score, cos D_PH) | 0.816 | 0.870 | **0.909** | **0.907** | 0.861 | 0.789 | 0.731 | 0.723 | 0.722 | 0.694 | 0.755 |
| partial \| j | 0.375 | 0.822 | **0.974** | **0.982** | 0.869 | 0.605 | 0.814 | 0.861 | 0.806 | 0.686 | 0.729 |
| r(score, cos V_CAA) | −0.642 | −0.032 | 0.216 | −0.427 | −0.209 | −0.452 | 0.086 | −0.191 | 0.063 | 0.509 | 0.486 |
| partial \| j | −0.521 | 0.250 | 0.393 | −0.333 | −0.094 | −0.232 | −0.459 | 0.434 | 0.081 | 0.315 | 0.598 |

**Alignment with the phrase direction predicts behaviour at r = +0.91 (partial +0.98), and it
peaks early — L6–L8, immediately after phase 5's L7 break.** Phase 2 found its peak at L32 of
36; the same measurement on the bridge problem puts it at the other end of the network.

**Alignment with the CAA steering vector predicts nothing**, at any depth: it swings between
−0.64 and +0.66 with no structure. The vector that steers the model when *injected* is not the
vector that reads out whether a *prompt* intervention is working. That is a sharper statement
of phase 5 §7 and phase 7 §1b's `cos(grad, V_CAA) ≤ 0.007`, from a third direction.

Lens depth for everything below: **L8**.

## §5 — ⁂ random search matches GCG at equal compute

**NEXT-STEPS item 1**, which no phase in this project has ever had. Three arms, phase 6 trial
12's winning configuration throughout (k=53, `n_mut=7`, `n_cand=256`, `n_top=512`, full pool,
suffix, repeat init, seed 1), the **same accept test** (the metric, verified by a real forward
pass on the current greedy rollout) and the **same 240 s budget**. Only the proposer differs.

| proposer | steps | evals | pred_corr | best (teacher-forced) | greedy 160 | sampled 45×3 | distinct | lens proj |
|---|---|---|---|---|---|---|---|---|
| metric gradient | 56 | 14336 | **−0.106** | 0.0623 | 0.0481 | 0.0593 | 0.74 | 0.08 |
| twist lens | 58 | 14848 | **−0.035** | 0.0601 | 0.0561 | 0.0562 | 0.66 | 0.80 |
| **random** | 58 | 14848 | — | **0.0622** | 0.0526 | 0.0562 | 0.64 | 0.07 |

**Uniform random mutation under a max reaches 0.0622 — against the metric gradient's 0.0623 and
phase 6's tuned 0.0620.** Three phases have described 0.062 as GCG maximising an objective;
14,848 random proposals get there in the same wall-clock, from the same init, with the same
accept test. What is doing the work is the accept test, not the gradient.

**Both gradients are useless as proposers, and the lens is only marginally the better one.**
−0.106 for the metric gradient continues the project's series (−0.192 / +0.091 / +0.055 across
phases 2 / 4 / 5). The twist lens comes out at −0.035: not the +0.348 phase 2 measured on the
4B, but the least-bad number in the table. On this backbone, at this scale, neither gradient
proposes.

The lens gradient *does* move the quantity it is aimed at — its trigger's projection is 0.80
against 0.07–0.08 for the other two arms, a 10× separation. It simply buys no metric.

All three answers are the same fluent meta-commentary phase 6 reported, with zero concrete and
zero metaphorical bridge words:

> *"It looks like you've shared a mix of random characters, words, and phrases that don't form
> a coherent message. It might be a puzzle, a test, or just a random string of text."*
> — metric proposer, verbatim phase 6 §6

## §6 — ⁂ the lens as its own objective: reached, exceeded, and empty

Optimising the projection directly — phase 7's untested projection objective, on the direction
§3 validated at r = +0.98. It is cheap: a candidate costs 8/36 of a forward pass and no
rollout, so the same 240 s buys **616 steps and 157,696 evaluations**, 10× the other arms.

`<h_L8[last prompt position], d_hat>`:

| | projection |
|---|---|
| **lens-optimised trigger** | **+6.40** |
| `' the user really loves bridges'` (100% on-topic) | +2.30 |
| `' bridge'` (48.6%) | +1.86 |
| commas × 8 | +1.38 |
| `' the user really loves puzzles'` (control) | +1.24 |
| random 53 junk tokens (the init) | −0.19 |

**The search beat the working phrase by 2.8× on the objective the working phrase defines.** And
the result behaves like every other trigger in this project:

| | in.raw | in.cent | out.raw | out.cent | dist | concrete/metaphor |
|---|---|---|---|---|---|---|
| lens-objective trigger | 0.0329 | 0.0154 | −0.0204 | **0.0541** | 0.61 | 0/1 |
| `' the user really loves bridges'` | 0.0503 | 0.0332 | 0.0289 | **0.0896** | 0.68 | 6/1 |
| *[control queries]* | 0.036–0.041 | 0.019–0.023 | −0.022…−0.016 | 0.050–0.055 | | |
| *[bridge queries]* | 0.055–0.078 | 0.038–0.061 | 0.014–0.038 | 0.086–0.103 | | |

Inside the control band in all four spaces. `pred_corr` for the lens gradient against its own
objective was **−0.350**.

### §6.1 — it is not norm-gaming, and it is not the fluency shortcut

`<h, d_hat> = ‖h‖ cos(h, d_hat)` and nothing bounds `‖h‖`, so the obvious explanation is that
the search inflated the state. **It did not.** Measured against the run's own junk init:

| | ⟨h,d⟩ | ‖h‖ | ‖h−ref‖ | **cos(h−ref, d)** |
|---|---|---|---|---|
| lens-objective trigger | 6.40 | 41.16 | 17.78 | **0.371** |
| twist-lens proposer trigger | 0.80 | 38.99 | 9.77 | 0.102 |
| metric proposer trigger | 0.08 | 38.21 | 9.34 | 0.030 |
| random proposer trigger | 0.07 | 39.26 | 9.50 | 0.028 |
| `' the user really loves bridges'` | 2.30 | 38.52 | 9.21 | **0.272** |
| `' bridge'` | 1.86 | 38.09 | 8.37 | 0.247 |
| commas × 8 | 1.38 | 37.90 | 8.48 | 0.187 |
| `' the user really loves puzzles'` | 1.24 | 38.58 | 9.09 | 0.159 |

‖h‖ moved 7% (38.5 → 41.2, against a mean non-sink 52.2). The trigger reached +6.40 by
**genuinely aligning**: cosine 0.371 against the working phrase's 0.272. Phase 7 §5's geometric
obstruction does not apply here, exactly as phase 7 predicted it would not — that was an L2
target with a threshold of `cos > 0.329`; a projection objective has no threshold, and token
space clears the working phrase's alignment comfortably.

The second obvious explanation is that `d_hat` is really "there is fluent text in the slots"
(commas already score 0.187). Split it: `F_DIR` = mean displacement of the 8 matched control
phrases, `D_PERP` = the part of `d_hat` orthogonal to it.

| | cos(·, d_hat) | cos(·, F_DIR) | **cos(·, D_PERP)** |
|---|---|---|---|
| lens-objective trigger | 0.371 | 0.384 | **0.326** |
| `' the user really loves bridges'` | 0.272 | 0.989 | **0.149** |
| `' bridge'` | 0.247 | 0.889 | 0.136 |
| `' the user really loves puzzles'` | 0.159 | 0.976 | 0.036 |
| commas × 8 | 0.187 | 0.847 | 0.081 |
| twist-lens proposer trigger | 0.102 | 0.517 | 0.038 |

`cos(d_hat, F_DIR) = 0.125`, so fluency is 1.6% of the lens direction by variance — and on the
bridge-specific axis the optimised trigger leads the working phrase **0.326 to 0.149**. The
search did not find a shortcut. It found more of the real thing and produced none of the
behaviour.

## §7 — the confound guard: the gradient is *not* uniformly worthless

§5's "random matches GCG" needs a guard: is that a property of the bridgeness objective, or of
this rig? Same pool, scaffold, code and budget, but a **next-token** objective — maximise
`p(target)` at the first answer position, with the target string banned from the pool so
neither arm can simply write it. 120 s per arm.

| target | proposer | p start | p best | steps | pred_corr |
|---|---|---|---|---|---|
| `' Sure'` | gradient | 0.0000 | **0.0019** | 65 | −0.033 |
| `' Sure'` | random | 0.0000 | 0.0007 | 71 | — |
| `' wolf'` | gradient | 0.0000 | 0.0004 | 65 | **+0.145** |
| `' wolf'` | random | 0.0000 | 0.0004 | 71 | — |

**Equivocal, and reported as such.** On `' Sure'` the gradient beats random 2.7× at the
endpoint and 6× at the shared step-50 checkpoint (0.0012 vs 0.0002); on `' wolf'` they tie.
`' wolf'` also produced the project's first **positive** `pred_corr` on the 8B (+0.145).

Two things follow, one supporting §5 and one qualifying it.

- **Supporting.** There exists at least one objective in this rig where the gradient clearly
  outperforms random, so §5's tie is not a plumbing artifact.
- **Qualifying.** The absolute numbers are tiny — 0.19% and 0.04% — nothing like phase 4's
  0.9951/0.9991 for animal targets. Phase 4 used 8 slots in an *"answer as a single word"* probe
  with a prefilled answer slot; here 53 junk tokens trail a free-form question and the first
  answer token has strong natural openers to beat. **This scaffold is a hard setting for
  next-token control too**, so the guard does not fully license reading §5 as a pure statement
  about the objective.

Note also that `pred_corr` and *usefulness* come apart: the `' Sure'` gradient wins on outcome
with `pred_corr` −0.033. The value is in the **top-k restriction** (a filter), not in ranking
within it. Phases 2–8 have been reading `pred_corr` as if it measured the former.

## §8 — searching fluent English under the good objective

If the objective is sound (§3) and the junk region is empty (§5), the obvious move is to search
where the working phrases live. Phase 5 tried a fluency penalty and got grammatical English at
0.0%, but under an objective that ranked those phrases **last**; here the objective ranks them
first. Word pool (36,696 whole-word tokens → 36,306 after the blocklist), 12 slots initialised
to `' I have been thinking about what to do with my free time'`, single-slot Gibbs sweeps,
accepted on the metric. Two proposers, plus a ceiling control with the blocklist off.

| | in.raw | in.cent | out.raw | out.cent | dist | log p(trig) | c/m |
|---|---|---|---|---|---|---|---|
| init sentence (unoptimised) | 0.0356 | 0.0188 | −0.0179 | 0.0516 | 0.64 | −2.15 | 0/0 |
| word-pool random / blocked | 0.0397 | 0.0227 | −0.0151 | 0.0578 | 0.53 | −13.38 | 1/0 |
| model-infill / blocked | 0.0405 | 0.0232 | −0.0249 | **0.0643** | **0.44** | −7.87 | 0/0 |
| model-infill / **UNBLOCKED** | 0.0363 | 0.0194 | −0.0197 | 0.0595 | 0.56 | −7.26 | 0/0 |
| `' the user really loves bridges'` | 0.0503 | 0.0332 | 0.0289 | **0.0896** | 0.68 | −2.86 | 6/1 |
| *[control queries]* | 0.036–0.041 | 0.019–0.023 | −0.022…−0.016 | 0.050–0.055 | | | |
| *[bridge queries]* | 0.055–0.078 | 0.038–0.061 | 0.014–0.038 | 0.086–0.103 | | | |

**⚠ The ceiling control failed, so this is not a clean negative.** Phase 6 §5 predicted an
unblocked search would trivially write a prompt injection; it did not. The unblocked arm found
`' pont'` — *bridge* in French — and used it as a **book title** (`' is the song line the book
The wind by le pont original'`), producing an answer that *denies* the premise: *"The line
'What shall I do today?' is **not** from the book The Wind by Le Pont."* It reached 0.0658
teacher-forced and 0.0595 regenerated, well short of the phrase band. A ceiling control that
does not reach the ceiling bounds the **search**, not the region — so the two blocked arms
below it cannot be read as "no covert fluent trigger exists".

What the arms do establish:

1. **Fluency is free and buys nothing here.** Model-infill triggers are 5.5 nats/token more
   probable than word-pool ones (−7.87 vs −13.38) and score the same (0.0610 vs 0.0612
   teacher-forced). Phase 5's fluency/target trade-off does not reappear — but neither does any
   gain.
2. **Every fluent arm has the same four-space signature as the junk arms** — inside or at the
   control band in `in.raw`, `in.cent`, `out.raw`, and moving only `out.cent`. The one apparent
   winner, model-infill at 0.0643 greedy (the best regenerated score of any search in this
   phase), has **distinctness 0.44**, below phase 6's 0.45 looping threshold. It is not fluent
   text scoring well; it is a repetitive answer.
3. **Local edits cannot reach a global structure.** Every arm's answer is *about the trigger* —
   translating it, listing its words, correcting its false premise — never adopting it as a
   preference. `' the user really loves bridges'` works because it is a **statement about the
   user** that the model then acts on; single-slot Gibbs from a neutral sentence never
   restructures into that, in 13k evaluations. The 'ford' / 'bridal' / 'railway' / 'brig' tokens
   the searches did surface are phase 2's spelling-and-adjacency channel, not preference.

## What phase 8 establishes

1. **⁂ The objective was never the problem.** All five of phase 5's working phrases land in or
   above the bridge-query band in **all four** embedding spaces; all four floors (a topic-matched
   control, commas, `' the'`, random junk) land in the control band. Phase 5 §6's objective
   ranked the same phrases *last* (−0.55); phase 6's ranks them first. NEXT-STEPS item 3
   resolves in the good direction, and it removes the last standing excuse for a search failure.
2. **⁂ Random search matches GCG at equal compute** — 0.0622 against 0.0623 and phase 6's tuned
   0.0620. Every "GCG maximised the metric" claim in phases 5–7 should be read as "a max over
   ~15k verified proposals reached 0.062", because the gradient contributes nothing detectable.
3. **The gradients do not propose on this backbone.** `pred_corr` −0.106 (metric) and −0.035
   (twist lens), continuing −0.192 / +0.091 / +0.055. Phase 2's +0.348 does not reproduce here;
   the lens is better than the metric gradient and still not usable.
4. **⁂ The twist lens is the seventh objective to fall, and the cleanest one yet.** 157,696
   evaluations drove the projection to 2.8× the working phrase's, by genuine alignment
   (cos 0.371 vs 0.272, ‖h‖ up only 7%) and specifically on the bridge axis (0.326 vs 0.149) —
   and the answer sits in the control band in all four spaces with zero bridge words.
5. **Within-family dose–response does not license an objective.** The direction was validated
   the way phase 2 validated its lens — degrade a working intervention and correlate — and
   scored r = +0.91, partial **+0.98**. That correlation is real and it does not survive leaving
   the family: triggers outside it reach higher alignment with less behaviour. This is phase 6's
   "separating natural text by topic is insufficient validation" with a stronger validation
   method and the same outcome.
6. **The working phrase's direction and the CAA steering vector are orthogonal** —
   `cos(V_CAA, D_PH)` = −0.013 to +0.05 below L36 — and only the first predicts behaviour, at
   any depth. Phase 5 §7's "two different internal routes" as a direct cosine.
7. **The alignment depth does not transfer.** Phase 2's signal peaked at L32 of 36; phase 8's
   peaks at L6–8 of 36, just past phase 5's L7 break. Measure it per problem.
8. **The gradient is not uniformly worthless, and `pred_corr` was the wrong instrument.** On a
   next-token target in the same rig it beats random 2.7× while scoring `pred_corr` −0.033 — the
   value is the top-k filter, not the ranking. But absolute next-token gains here are 0.19%, so
   this scaffold is hard for GCG generally and §5's tie is not *purely* about the objective.
9. **Fluent search does not reach the phrase band either — but its ceiling control failed.**
   Word-pool and model-infill searches under the good objective land in the same four-space
   signature as the junk arms, and the unblocked ceiling arm found `' pont'` and wrote it into a
   book title rather than a preference. Since the control never reached the ceiling, this bounds
   the *search*, not the region: the fluent-region question is open, not answered.

## Open

- **One seed, one query, one target word, one phrase.** Everything is `what shall i do today`,
  seed 1, ` bridge`, and `' the user really loves bridges'`. Phase 6 §10 measured within-level
  sd 0.0024, which is the same order as the 0.0622-vs-0.0623 gap in §5 — that gap should be
  read as "indistinguishable", not as "random is ahead".
- **The random arm was not tuned.** It got phase 6's GCG-tuned hyperparameters, which is
  conservative in GCG's favour for `n_top` (unused) but arbitrary for `n_mut` and `n_cand`.
- **§4's ladder has 15 points** and the partial correlation is computed within 6 levels of *j*.
  The depth ordering is clear (early ≫ late) but the L6-vs-L8 choice is inside the noise.
- **§8's search design is the obvious thing to fix first.** Single-slot Gibbs from a neutral
  sentence cannot restructure into a statement about the user. A proposer that edits at phrase
  level — model-generated rewrites, or a genetic crossover over whole clauses (AutoDAN) — is the
  version of this experiment that would actually test the region, and its ceiling control has to
  pass before any negative counts.
- **The lens was only ever read at the last prompt position.** Phase 2 read at the answer slot
  in a scaffold where the answer was one token; here the answer is 45–160 tokens and the
  objective could have been defined over the answer positions instead.
- **`' I should mention bridges'` outscores the 100% phrase** (0.1074 vs 0.0896) while being
  weaker behaviourally in phase 5. The metric is trustworthy at band resolution, not at rank
  resolution — and none of §5/§6's comparisons depend on within-band ordering.

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `twist_lens_bridge_stub.ipynb` | executed, outputs included |
| `phase8_twist_lens.json` | rig · both directions at every layer · the 9 interventions with answers · the ladder with per-layer alignments · §5/§6 arms with trajectories and pred_corr |
| `phase8_followups.json` | §7's four next-token guard runs · §8's three fluent arms with trajectories, triggers, log-probs and answers |
