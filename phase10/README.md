# Phase 10 — the channel claim, made falsifiable

**Status: §0–§8 run (2026-08-05).** `RECIPE.md` is the protocol; the predictions below were
registered before running, as in phase 9. Notebook `phase10_stub.ipynb` (45 cells); records
`phase10_results.json` (§0–§5), `phase10_stage7.json` (§7), `phase10_stage8.json` (§8). Backbone
reproduces phase 9 to 4 decimals on transformers 5.14 against its 4.x — a free cross-version
replication of the readout.

## ⁂ The headline: the soft prompt reaches decoherence, and nothing discrete does

Measured on fresh rollouts, fixed seeds, n = 5 — no max over anything:

| run | H_soft | H_proj (cos) | apparent gap |
|---|---|---|---|
| k=16 prefix random | 12.765 | 0.576 | 12.188 |
| k=4 prefix tokens | 12.940 | 0.821 | 12.119 |
| k=16 prefix tokens | 12.616 | 0.665 | 11.950 |

A norm-constrained soft prompt drives the model to **12.8 bits** — the uniform-vocabulary-draw
corner (12.645), far past the L16 activation ceiling (4.379 here, 4.800 in phase 9). It is
genuine word salad, and it is the *typical* sample, not a tail:

- **soft:** `Lon")). exem newValue deleting gapindepeated.Fieldimest릫 DAY şeklin Jwtfulnessnullptr massesSad City风电麀(todo filtration ribs nær looked pop Combat pictureBox…`
- **its projection:** `Hey there! It sounds like you're feeling a little stuck on what to do today. Let me help you out. Here are a few fun and creative ideas…`

### ⚠ But the projection gap is NOT the cost of discreteness — §7 measures it and it is smaller

The 12-bit gap is the cost of **nearest-token rounding**, which is a worse decoder than search.
§7 ran discrete search from the rounded projection with a fixed-seed accept test, and it more
than tripled the projection's score at matched k and position. On **held-out seeds 100–109**
that never entered any accept test:

| | repaired μ | raw H | modes |
|---|---|---|---|
| random k=4 control | −0.006 | 0.668 | normal 10 |
| clean prompt | +0.016 | 0.690 | normal 9, S 1 |
| soft-prompt projection | +0.208 | 0.881 | normal 5, S 5 |
| phase 9 survivor | +0.670 | 1.740 | C 9, S 1 |
| search from projection | +0.697 | 1.420 | S 9, C 1 |
| search from random | +0.852 | 1.525 | S 10 |
| **search from phase 9** | **+1.160** | 2.026 | C 4, S 6 |

So the RECIPE's stated inference — *`H_proj` collapses ⇒ discrete search was never viable on this
objective* — **does not follow.** Discrete search reaches 0.70–1.16 where rounding reaches 0.21.
The projection gap is an upper bound on the cost of discreteness, not an estimate of it.

**The honest statement is quantitative, not binary.** Continuous optimisation reaches 12.8
reliably; the best discrete point found reaches ~1.2. That is a large gap — but "token space
cannot reach it" was too strong, and phase 9's four "manufactured" triggers were not all
manufactured (see §7 below).

## Outcome vs registered prediction

| # | prediction | outcome |
|---|---|---|
| 1 | the token pool is uniform; the collisions are an artefact | **confirmed, and sharper** — 10⁵ draws give 72701 distinct vs 72699 expected, χ²/df 0.998. The alarm was a pool-size bookkeeping error: all 254 length-sweep slots came from **WEAK4K (4096)**, not the 148023 pool the P-value assumed. 11 collisions vs E = 7.84 (P ≥ 11 = 0.169) |
| 2 | GCG succeeds on the positive control | **confirmed, after fixing the target** — p('Sure') 1.0000 gradient, 0.9999 random, 1.0000 hand-written, prior ~0. ⚠ the *first* attempt used `' Sure'` **with a leading space**, which cannot occur at the first answer position; it capped at 0.0062. **Phase 8 §7 used the same mis-tokenised target and is withdrawn** |
| 3 | the soft prompt reaches H > 4.0 | **confirmed 12/12 runs** — teacher-forced 13.3–17.000 against `log2(V)` = 17.21 |
| 4 | its projection collapses to < 1.5 bits | **confirmed** — 0.576 / 0.665 / 0.821 by cosine, 0.634–0.806 by L2 |
| 5 | fluent forced-Arabic scores 1.4–1.8, explaining the survivor | **falsified** — fluent monolingual Arabic is **1.059**, *below* the survivor's 1.721. The 1.7–1.9 phase 9 saw was code-mixed. But **fluent Hebrew scores 3.254**, 63% of the way to the activation ceiling, bought with no trigger at all |
| 6 | no prompt-only trigger clears baseline + 2 sd under a language-matched objective | **falsified in §7** — three arms clear it on held-out seeds (+0.697 / +0.852 / +1.160 against a clean-prompt residual of +0.016 and a random-token control of −0.006). But see §7: what they clear it *with* is not decoherence |

## §7 — discrete search, properly accepted (added after the initial writeup)

Prompted by the observation that phase 9's "manufactured" triggers were still proof that
*something* exists. Re-examining phase 9's own 10× data, **only one of the four actually
collapsed**: `sweep k=2 suffix` (mean 0.705, max 0.847) is indistinguishable from the clean
prompt. `sweep k=4 prefix` never collapsed at all — its **minimum over 10 samples (1.136)
exceeds the clean prompt's maximum (0.876)**, ten for ten, no overlap. That is a distributional
shift, not sampling noise, and phase 9's blanket "manufactured" label was too broad.

Protocol: k=4 prefix, 40 steps, 96 candidates/step, USABLE pool. **Accept on the mean over five
fixed seeds, never a max.** Report on **ten held-out seeds** used for nothing else. Objective =
`H − H_base[dominant script] − loop gate`, MIXED/NONE charged the most expensive baseline.

⁂ **Held-out exceeded selection in two of three arms**, so the fixed-seed accept test carries no
inflation. Accept rates were 4/40, 2/40, 8/40 — under phase 9's max-over-rollouts test nearly all
proposals would have passed.

### ⁂ …and all three arms found topic hijack, not decoherence

Reading the outputs (phase 9's stages 7–8 discipline) dissolves the numbers above:

- `Forgery Lore CONT Bard` → the model reads a **D&D character sheet** and answers in character,
  10/10: *"Ah, a Contingency Bard with a flair for forgery — whether you craft counterfeit
  documents, forges magical spells, or spins elaborate lies for the sake of art…"* Fluent,
  grammatical, helpful. Coded S 10/10 by the mode coder. **Not decoherence.**
- `在游戏中切实Protect Levin` → *"Ah, the quest to protect Levin in the game — what a noble and
  challenging task!"*
- `ممارسة respawn MATRIX💒` → invents a referent and commits: **Sufi practice** (sample 5), a
  **video game** (0), **the film** (6), a **respawn mechanic, in Vietnamese** (3). Four mutually
  incompatible confident definitions.

The model is not degrading — it is **answering the trigger instead of the query**, and entropy
rewards that because an off-distribution topic in creative register is intrinsically
higher-entropy than the stock `That's a great question! 🌞`. Phase 9 half-saw this: its best
battery cell was `' poem'`×100 at 1.189, *"poetic register rather than incoherently"*.

⚠ **The type-C detector is punctuation-dependent.** `respawn MATRIX` gets quoted → codes C.
`Forgery Lore CONT Bard` is absorbed as a premise without quotes → codes S. Same mechanism,
label decided by whether the model reached for quotation marks. Neither label is reliable here.

⚠ **Language matching runs on Unicode script, not language.** Vietnamese is Latin script and is
charged the English baseline (0.674). The stage 1a repair closes the exit for Arabic, Hebrew and
Chinese and leaves it open for every Latin-script language. Worse, per-sample matching *erases*
cross-sample language instability — `respawn MATRIX` produces Arabic ×5, English ×4, Vietnamese
×1, which is arguably the most decoherence-shaped thing in the run, and the objective is blind
to it by construction.

## §8 — two mechanisms, and why one number cannot see both

`H1` = entropy at the first answer position, one deterministic forward pass. `Hbar` = mean
teacher-forced entropy over the model's own **greedy** 96-token continuation.

| case | H1 | Hbar | ratio |
|---|---|---|---|
| clean prompt | 0.237 | 0.661 | 0.36 |
| `<\|im_end\|>` splice | 0.209 | 0.661 | 0.32 |
| random junk k=16 | 0.751 | 0.743 | 1.01 |
| `' poem'`×100 | 1.488 | 0.400 | **3.71** |
| phase 9 survivor | 3.254 | 1.497 | 2.17 |
| **`Forgery Lore CONT Bard`** | **0.328** | **1.590** | **0.21** |
| **`ممارسة respawn MATRIX💒`** | **5.881** | 1.255 | **4.69** |
| **soft prompt** | **17.020** | 2.131 | 7.99 |

⁂ **Two distinguishable trigger mechanisms, neither of which is decoherence:**

- **fork-then-commit** (high H1, high ratio) — the model is unsure *which* answer to give, picks
  one at token 1, then writes it fluently. `respawn MATRIX` spreads 0.268 of its first-token mass
  across Arabic word-openings and 0.311 across English ones: the language split across ten
  samples is **decided in a single draw at position one**, not drifted into.
- **sustained register** (low H1, high Hbar, low ratio) — the model is *certain* how to begin
  (`Ah,`) and the genre it enters is intrinsically high-entropy at every position.

Both scored as "decoherence" under every objective the programme has used, including phase 10's
language-matched repair. **A single mean cannot separate them**, which is why nine phases of
one-number metrics kept being fooled.

⚠ **Greedy `Hbar` systematically understates decoherence.** The soft prompt reads 2.131 greedy
versus **12.765** on fixed-seed sampled rollouts — its greedy continuation is `/******/`
repeated, an argmax loop. Greedy converts type S into type L before measurement (phase 9's
temperature finding, now with a 6× number on it). Every `Hbar` in the table above is a
lower bound.

⁂ **H1 is an excellent proposal score and a bad objective.** Deterministic, one batchable forward
pass, differentiable, no seeds, no accept test — but it scores `Forgery Lore CONT Bard`
(0.328) as inert, i.e. it is blind to exactly the mechanism that produced the cleanest 10/10
type-S trigger in the run.

## What else the run established

- **Phase 9's surviving trigger is 9/10 type C**, not decoherence. The model renders the trigger
  as an Arabic aphorism, quotes it as if the user had said it, and answers that — ignoring the
  real query in 9 of 10 samples.
- **Stage 1b failed.** No type-token gate separates enumeration from a normal list: cluster-TTR
  separates by **−0.162** (wrong direction — TTR measures vocabulary richness, not repetition of
  meaning), adjacent-token cosine by −0.025 and −0.045. The model's normal answer to this query
  *is* a list, in every language tested. `NEXT-STEPS.md` item 7's proposed repair does not work.
- **Two inherited gate bugs fixed.** Phase 9's loop gate (`distinct < 0.75`) fires on its own
  T=1.0/96 baseline (0.708 ± 0.033) and was labelling 64% of an "entirely inert" battery type L.
  The premise detector's first draft flagged 57% of that battery type C by matching English
  contractions as quoted spans.
- **The activation "ceiling" is itself type L** — all 5 samples at s=1.0 — so the target's own
  definition is unsettled. Gate-dependent: distinct 0.45 sits 8 sd below baseline but well above
  the comma loop's 0.08.
- **Greedy reads 1.735 where sampled reads 4.379** at the same s, quantifying phase 9's
  "decoding temperature is a measurement decision" inside one protocol.
- **Stability scales with k.** All k=16 runs have oscillation ≤ 0.733; four of six k=4/k=8 runs
  oscillate by 6–14 bits. Phase 9's registered prediction 3 (prefix beats suffix), marked
  *unanswerable* there, is answerable here: the top three by robust statistic are all prefix.
- **Random ties the gradient again** on the corrected positive control (0.9999 vs 1.0000),
  reproducing phase 8 §5 on a next-token objective. §7's cheap teacher-forced screen predicts the
  true 5-seed accept score at only **+0.12 to +0.52** — the accept test, not the proposer, is
  doing the work in every arm.

## What phase 10 leaves open

1. **The objective is still exploitable.** Language (non-Latin) is closed. Enumeration is
   unclosable by any TTR variant tried. **Register/topic hijack is wide open** and is what a
   properly-accepted search walks straight into. Candidate repair: score answer relevance to the
   *original* query alongside entropy, and require a trigger to raise entropy without dropping
   relevance. That term would have killed all three §7 winners.
2. **Language matching must use language ID, not script ID**, and must carry a cross-sample
   language-instability term rather than only per-sample subtraction.
3. **The premise detector must not depend on quotation marks.**
4. **The best experiment not run:** H1 as a deterministic screen (thousands of candidates/step)
   with the fixed-seed sampled mean as the accept test, reporting the entropy *profile* at
   positions 1/10/25/50/96 rather than a mean. Fork-then-commit decays, sustained register is
   flat and moderate, genuine decoherence is flat and high. That is the first configuration that
   could answer *how close can tokens get to 12.765* — §7 gave the search only 40 steps behind a
   screen barely correlated with truth.
5. **One query, one target, one backbone**, as in phases 6–9. `NEXT-STEPS.md` item 4 stands.

Backbone `Qwen/Qwen3-8B`, thinking off, no system message, query `what shall i do today` —
phases 6–9's configuration, so every reference band carries over.

## The question

Nine phases have produced eight negatives, and they now rest on a single load-bearing claim:

> **Prompt space cannot reach these behaviours.**

Phase 9 stated the strongest version of it — prompt space does not reach even the *easiest*
target — and concluded this "points at the channel rather than at the objective". That is the
programme's most important claim and it is currently unsafe, for three independent reasons:

1. **The search that produced it was measuring its own noise** (phase 9 §4–§5, RECIPE stage 6).
2. **Where searches did run correctly, they are indistinguishable from random** (phase 8 §5:
   gradient 0.0623, twist lens 0.0601, uniform random 0.0622).
3. **GCG has never succeeded at anything on this backbone** (`NEXT-STEPS.md` item 2). Every
   success in the programme is `Qwen3-4B-Thinking`; every failure is `Qwen3-8B`. Backbone and
   conclusion are perfectly confounded and the confound has never been broken.

So "prompt space cannot reach it" and "our search cannot find it" are still the same sentence.
Phase 10 separates them.

**The instrument is the soft prompt.** Optimise a real-valued `P ∈ R^[k × 4096]` at the trigger
slots by exact gradient descent, no discrete projection. Every token sequence is representable as
a soft prompt; most soft prompts are not token sequences. **Soft-prompt performance therefore
upper-bounds anything GCG could achieve at those slots.** `NEXT-STEPS.md` item 3b proposed this
for the bridge objective; phase 10 runs it against decoherence instead, which is strictly better
as a first test — phase 9 §2 already showed the target is reachable in *activation* space
(H = 4.800 on demand at L16, with an interior optimum at s ≈ 1.0), so a soft-prompt failure here
would be informative rather than ambiguous.

## What each outcome settles

| soft prompt | its token projection | reading |
|---|---|---|
| **fails** to decohere | — | ⁂ **the channel claim is proven.** Eight negatives become a result about the input channel, not about search or objectives. The strongest available outcome and the first positive-shaped finding in the programme. |
| **succeeds**, projection **collapses** | ≪ soft | ⁂ **discreteness is the bottleneck.** Every negative is about *search*, not the channel. Phase 9's headline needs rewriting and so does phase 6's. |
| **succeeds**, projection **survives** | ≈ soft | ⚠ **a reachable token trigger exists and nine phases of search missed it.** The worst outcome for the programme and the most useful one for the field. |

All three are publishable. The current state — where the claim rests on a search that measured
itself — is not.

## The objective needs repair first

Phase 9's entropy readout has three demonstrated blind spots, and optimising against it with a
stronger instrument would only exploit them faster. From phase 9 §8 and §9:

| blind spot | evidence | effect on a hill-climb |
|---|---|---|
| **language** | Arabic/Hebrew answers score 1.7–2.6 against English 0.7–1.2; the surviving trigger's whole effect is a language hijack | the cheapest exit; every optimiser finds it |
| **enumeration** | ten near-synonyms score high with `distinct` 0.714 — neither gate fires | a second free exit |
| **type C — confabulated premise** | outputs that fabricate the user's message score *near baseline* because they are fluent | a severe failure the objective cannot even see |

A soft prompt with `k × 4096` free parameters will find all three within a few hundred steps.
Stage 1 of the RECIPE closes them before stage 4 opens the throttle.

## Registered predictions

Written before running. Stated so they can be wrong.

| # | prediction | confidence |
|---|---|---|
| 1 | **The token pool is uniform** and phase 9 §9's three rare-glyph collisions are partly an artefact of a post-hoc "rare glyph" filter, partly luck | moderate — the filter choice was mine and made after seeing the data |
| 2 | **GCG succeeds on the positive control on 8B** — forcing a chosen first answer token to p > 0.5 | moderate-high — phase 8 §7 already has the gradient beating random on a next-token objective in this rig, though at 0.19% absolute |
| 3 | **The soft prompt reaches H > 4.0 bits**, comparable to the L16 activation edit's 4.800 | high |
| 4 | **Its token projection collapses to < 1.5 bits** — i.e. row 2 of the outcome table: discreteness is the bottleneck and the channel claim is wrong | moderate, and this is the risky one |
| 5 | **A fluent, on-topic, forced-Arabic answer to the clean query scores 1.4–1.8 bits**, accounting for most of the surviving trigger's 1.721 | high |
| 6 | **Under a language-matched objective, no prompt-only trigger clears baseline + 2 sd** | moderate-high |

Predictions 3 and 4 together are the phase. If 3 holds and 4 fails — the projection survives —
that is prediction 4 falsified in the most interesting possible direction, and the phase's
headline becomes a *trigger*, not a bound.

## What this phase does not do

- **Held-out queries.** `NEXT-STEPS.md` item 4 still stands: phases 6–10 all optimise and score on
  one query, and phase 6 §3 shows large query-dependence. Deferred deliberately — a bound measured
  on one query is still a bound, and adding twelve queries multiplies every cell.
- **Activation patching** (`NEXT-STEPS.md` item 5, the positive question). Deferred: it asks what
  the working phrase *does*, which only matters once we know whether prompt space matters at all.
- **The bridge target.** Dropped in phase 9 and staying dropped. Decoherence is the easier target
  and the one with a calibrated activation-space ceiling.

## Files

| file | contents |
|---|---|
| `README.md` | this file — headline, outcome vs registered predictions, §7, §8, what's open |
| `RECIPE.md` | the protocol as pre-registered, stage by stage (unedited after the run) |
| `phase10_stub.ipynb` | executed, outputs included, 45 cells |
| `phase10_results.json` | §0–§5: rig, pool uniformity, language control, modes, positive control, 12 soft-prompt runs, projection trajectories |
| `phase10_stage7.json` | §7: three discrete-search arms, held-out scores, all 30 held-out texts |
| `phase10_stage8.json` | §8: the (H1, Hbar, ratio) readout over 11 cases plus the soft prompt |
