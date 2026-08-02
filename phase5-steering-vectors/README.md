# Phase 5 — a bridge steering vector, and a GCG search that tries to match it

**Status: §1–§7 run (2026-07-30), A100 40GB, bf16, transformers 5.14.1.**
`steering_vectors_stub.ipynb` is the executed copy, outputs included. `RECIPE.md` is the
extraction protocol as actually run; `RESEARCH.md` is the literature note behind it.

## ⚠ First result: phase 4's layer curve was an extraction artifact

**Phase 4 §6 concluded that ActAdd on this backbone "peaks at L4–L6 and dies by L8, mirroring
ActAdd's own layer curve." That is confounded, and the resemblance to ActAdd's curve was an
artifact.**

`' bridge'` and `' cat'` are **single tokens**, so for a bare pair the last token *is* position 0 —
Qwen3-8B's attention sink. Measured here:

| L | ‖h(bridge)‖ | ‖h(cat)‖ | cos(h_b, h_c) | ‖v‖ | top-3 dims' share of energy |
|---|---|---|---|---|---|
| 6 | 44.7 | 46.6 | 0.458 | 47.6 | 20.1% |
| **7** | **10457.5** | **11763.3** | **0.99995** | **1310.5** | **99.9%** |
| 12 | 10457.6 | 11763.4 | 0.99995 | 1310.5 | 99.9% |
| 24 | 10897.5 | 12203.1 | 0.99995 | 1310.4 | 99.9% |

From **L7** the two prompts are the same vector to five decimal places, 99.9% of the energy sits in
three dimensions (2276, 233, 4081), and ‖v‖ is pinned at 1310.5 all the way to L24 — a difference
of two massive-activation spikes, carrying no semantics. Phase 4 sampled `{4, 6, 8, 12, 16}` and so
never saw that the break is at 7, and read a dead vector as dead steering.

**The fix** is a shared context prefix so the differing token is not at position 0 —
`"The word is bridge"` − `"The word is cat"`, final token at position 3. Then ‖v‖/‖h‖ stays in
0.6–1.1 at every depth, and:

| L | 2 | 4 | 6 | **7** | 8 | 12 | 16 | 24 | 35 |
|---|---|---|---|---|---|---|---|---|---|
| cos(v_ctx, v_bare) | 0.819 | 0.883 | 0.759 | **0.037** | 0.030 | −0.005 | 0.002 | −0.062 | −0.060 |

The two forms **agree exactly where phase 4 found steering to work** and go orthogonal from L7 —
which is what makes this a correction rather than a difference of convention.

**And the corrected layer curve is a different curve.** With the ctx vector, L2–L6 produce only
degenerate repetition (`'bridgebridgebridge…'`), while the coherent, on-topic behaviour lives at
**L16–L20** — untested territory in phase 4 because its vector was junk there:

> L16, s=0.8 — *"To bridge the gap between the past and the future, I can offer you a bridge of
> understanding, a path of c…"*
> L20, s=0.8 — *"As a bridge between the past and the future, I can offer you a unique perspective
> to explore the path of…"*

That agrees with the literature's general finding — mid-to-late layers are consistently the most
effective intervention site — and disagrees with the ~12%-depth story inherited from GPT-2-XL.

**Word list validated:** the unsteered baseline fires **0.0%** on both the strict and the loose
list over 96 samples (mean perplexity 5.4), so the polysemy worry about `span`/`arch`/`crossing`
did not materialise on these prompts.

## §1.3 — the corrected layer × strength surface

12 held-out prompts × 8 samples = 96 completions per cell, T=0.8, 45 new tokens. Each cell is
`on-topic % · mean perplexity · degenerate %` (unsteered: 0.0% · 5.4 · 0.0%).

| | s = 0.6 | s = 0.8 | s = 1.0 |
|---|---|---|---|
| **L8** | 0.0% · 6.5 · 0.0% | 20.8% · 13.5 · 0.0% | 80.2% · 30.2 · 1.0% |
| **L12** | 0.0% · 6.5 · 0.0% | 6.2% · 9.4 · 1.0% | 67.7% · 14.5 · 3.1% |
| **L16** | 5.2% · 10.2 · 0.0% | 57.3% · 14.2 · 1.0% | **94.8% · 14.5 · 11.5%** |
| **L20** | 2.1% · 7.9 · 0.0% | 41.7% · 15.9 · 1.0% | 82.3% · 21.4 · **0.0%** |
| **L24** | 5.2% · 7.5 · 0.0% | 33.3% · 14.9 · 0.0% | 63.5% · 21.2 · 2.1% |

Four things worth keeping:

1. **The peak is L16, and depth is non-monotonic.** At s=1.0 the rate runs 80.2 → 67.7 → **94.8**
   → 82.3 → 63.5. Not a smooth rise-and-fall, so a coarse layer grid can land in the L12 trough and
   read the whole region as weak.
2. **The strength window is narrow and layer-dependent.** L8 goes 0.0% → 20.8% → 80.2% across
   s = 0.6/0.8/1.0. A greedy n=1 sweep over {0.8, 1.2} concluded L8 was dead; it is not. This is
   how the coarse-grid failure compounds with (1).
3. **Perplexity separates cells the rate ranks equal.** L8 reaches 80.2% at ppl 30.2 (5.6× the
   unsteered 5.4); L16 reaches 94.8% at 14.5 (2.7×). Rate alone is not quality.
4. **The two best cells trade off against each other, and the selector cannot separate them.**
   L16 s=1.0 is 94.8% with **11.5% degenerate**; L20 s=1.0 is 82.3% with **0.0% degenerate** at
   higher perplexity. `rate − degenerate` puts them at 83.3 vs 82.3 — one point, well inside noise
   at n=96. The notebook's selector took L16 s=1.0; a clean 82% is arguably the better target
   vector than a 94.8% that is one-in-nine word salad. Flagged rather than silently resolved.

ActAdd reports >90% at its best layer against a ~2% baseline; 94.8% against 0.0% is in that
neighbourhood, with the degeneracy caveat attached.

## §1.4 — controls: the direction is doing the work, not the magnitude

All at L16, s=1.0, same 96 completions:

| control | on-topic | degenerate | ppl |
|---|---|---|---|
| bridge vector | **94.8%** | 11.5% | 14.5 |
| reversed (`' cat'` − `' bridge'`) | 0.0% | 1.0% | 24.2 |
| random direction, matched ‖v‖ | 0.0% | 0.0% | 7.8 |

**The random control is the one phase 4 never ran, and it is the one that matters.** A random
direction of *identical magnitude* injected at the same layer and positions produces perplexity
7.8 against an unsteered 5.4 — the model simply absorbs it. So the effect is not "a large
perturbation at L16"; it is that direction. The reversal costs fluency (ppl 24.2) while producing
no bridges, consistent with a signed axis being driven the wrong way.

## §1.5 — the negative arm is load-bearing, so CAA is warranted

Eight `' bridge'` − X vectors at L16, all arms sharing the ctx prefix and equal token length:

| | mean | min | max |
|---|---|---|---|
| off-diagonal cosine | **0.421** | 0.327 (chair/cloud) | 0.541 (running/music) |

Vectors that share the same positive arm agree only ~42% in direction. If the negative were inert
they would be near-parallel. Modelling each as *shared bridge component + roughly orthogonal
negative-specific residue*, a pairwise cosine of 0.42 puts the bridge component at ~65% of each
vector's norm — **a single-pair vector is about a third idiosyncratic**. This is the "scattered"
branch: the choice of negative is doing real work and the CAA mean is worth paying for.

Averaging the eight shrinks the norm 70.0 → 51.9 with `cos(CAA, bridge−cat) = 0.756`, which is
what cancellation of the residue looks like rather than loss of signal.

**Behaviour decides it** — the CAA mean is a strict Pareto win at L16, s=1.0:

| vector | on-topic | degenerate | ppl |
|---|---|---|---|
| single pair `bridge − cat` | 94.8% | 11.5% | 14.5 |
| **CAA mean over 8 negatives** | **99.0%** | **4.2%** | **11.6** |

Higher rate, less than half the degeneracy, *and* more fluent — so this is not a rate/coherence
trade, it is the negative-specific residue being cancelled. It also settles the L16-vs-L20
question above: no need to give up rate to buy cleanliness. `V_TARGET` is the CAA vector.

## §3.2–3.3 — calibration killed the first objective, and depth rescued it

**The `nosink` reference is unusable, and the calibration cell caught it before the search ran.**
With the reference built from `nosink` steering, the vector is added *directly at the scoring
position* (‖Δ*‖ = 1.37 × ‖h‖), so the objective asks a trigger sitting at slots 8–16 to reproduce,
through attention alone, a direct additive write at the final position. Measured separation
between the real word and random junk: **+0.0009**, with `' bridge'` at +0.0020 against a floor of
+0.0010. Swept across every layer 17–35 and both scoring modes, `nosink` separation is **negative
or ~0 everywhere** (best: +0.0111). This is the non-surjectivity result made concrete — the steered
state is not reachable from any input.

**A structural fact fell out of the fix.** Restricting the reference steering to the trigger's own
slots (`span`) initially returned ‖Δ*‖ = 0.00: the injection is a forward-*pre*-hook on `L_TARGET`
and `capture` reads that same tensor, so with only the slot positions written, the scoring position
has not changed *yet* — the effect has not crossed an attention block. **The scoring layer must be
strictly greater than the injection layer.**

**With that fixed, the objective is a depth story** — `span` reference, scored on positions after
the span:

| L | 17 | 21 | 25 | 29 | 31 | 33 | **35** |
|---|---|---|---|---|---|---|---|
| floor (random) | +0.472 | +0.514 | +0.651 | +0.514 | +0.400 | +0.276 | **−0.038** |
| ceiling (`' bridge'`) | +0.258 | +0.278 | +0.424 | +0.432 | +0.433 | +0.453 | **+0.420** |
| **separation** | −0.214 | −0.236 | −0.228 | −0.082 | +0.033 | +0.177 | **+0.459** |

**At mid layers random junk outscores the real word.** The ceiling is roughly flat at ~0.42
throughout; what changes with depth is the **floor collapsing**. So at L17–L29 the objective is
measuring *"something is in the slots"*, not *"bridges"* — any perturbation at the slot positions
produces a similarly-directed downstream delta.

That is **phase 3's finding reappearing on a completely different objective.** Phase 3: mid-layers
encode "a specific animal is being named" — the slot, not the filler — with the filler only
linearly legible around L30–32, so a mid-layer objective is underdetermined by construction. Here,
on a bridge steering vector instead of an SAE representation, the same shape: usable separation
only appears past L31, and the semantic ordering at L35 is correct — `bridge` +0.420,
`viaduct` +0.385, `cat` +0.023, random −0.038.

Chosen objective: **`span` reference, `post` scoring, layer 35**, separation +0.4585.

## §3.4 — the search wins the objective and loses the behaviour

60 steps, batch 128, seed 1, pool blocked for `bridge` in ~40 languages plus the embedding
neighbourhood plus all pictographs. Score **0.0974 → 0.8468**.

Trigger: `' nettsteder Дмитр돠 путеш탔פייסב뼐 odense'`
(pieces: `' nettsteder'`, `' Дмитр'`, `'돠'`, `' путеш'`, `'탔'`, `'פייסב'`, `'뼐'`, `' odense'`)

| | objective score | on-topic behaviour |
|---|---|---|
| random pool trigger | −0.038 | 0.0% |
| real word `' bridge'` in the slots | +0.4204 | **52.1%** |
| **GCG trigger** | **+0.8497** | **0.0%** |
| CAA steering vector (the target) | — | 99.0% |

**The trigger beat the real word two-to-one on the objective and produced zero bridges — the same
0.0% as random junk.** This is phase 3's result reproduced on a completely different objective, a
different target, and a different model behaviour: *activation match quality does not predict
behaviour.* Phase 3 got it with an SAE representation of an animal; phase 5 gets it with a
difference-in-means steering vector for a topic.

**The mechanism is diagnosed, not mysterious.** The trigger's decomposition is `cos +0.2538` with
`‖Δ‖/‖Δ*‖ = 3.35`. The score

```
proj = ⟨Δ_T, û⟩ / ‖Δ*‖
```

is **linear in ‖Δ_T‖**, so a candidate can win by pushing hard in a poorly-aligned direction rather
than by aligning. GCG found that exploit immediately, as GCG does — this is the same lesson as
phase 4's rung-C censoring bug, in a different costume: *the metric was satisfiable without the
thing it was proxying for.*

Two useful side measurements:

- **The steering vector is roughly twice as potent as the best possible token splice.** The real
  word in the slots gets 52.1%; the activation edit gets 99.0%. That is a quantitative version of
  what the non-surjectivity paper argues qualitatively — the steered state is somewhere no input
  puts you.
- **The GCG gradient remains a weak proposer**: `pred_corr = +0.055`, in line with phase 2's
  −0.192 and phase 4's +0.091. Forward-pass verification of every proposal is what makes the
  search work.

## §3.5 — cosine can't be gamed the same way, and it fails too

Same search, same budget, objective replaced by `cos(Δ_T, Δ*)`, which is magnitude-free.

| candidate | cos | proj | ‖Δ‖/‖Δ*‖ | behaviour |
|---|---|---|---|---|
| random pool trigger | −0.0007 | −0.0015 | 2.15 | 0.0% |
| `' cat'` | +0.0102 | +0.0233 | 2.28 | — |
| `' viaduct'` | +0.2289 | +0.3848 | 1.68 | — |
| **real word `' bridge'`** | **+0.3473** | +0.4204 | 1.21 | **52.1%** |
| proj-optimised trigger | +0.2538 | +0.8497 | 3.35 | **0.0%** |
| **cos-optimised trigger** | **+0.2667** | +0.6801 | 2.55 | **0.0%** |

**So the failure is not the metric.** Two objectives with different exploitable structure land in
the same place. The cosine-optimised trigger is *better aligned than a genuine near-synonym*
(`viaduct`, +0.2289) and still produces zero bridges, while the real word at +0.3473 produces 52.1%.

**What the cosine trigger is made of is the most informative part.**
`'澽מחלה遆חשש借錢инфекци잃מלחמה'` — the pieces decode as **disease, fear, borrow money, infection,
lose, war**. A semantically coherent cluster, and it is not bridges. Whatever direction the bridge
vector's downstream effect points in at L35, high-intensity crisis vocabulary reproduces a
substantial part of it without carrying the topic. This is the phase-2/3 "triggers are stacked
routes" lesson inverted: an activation direction that separates `bridge` from `cat` cleanly is
still not *only* about bridges, and a search will find whatever else lives on it.

Note also that 60 steps of cosine search **never reached the real word's alignment**
(0.0215 → 0.2690 against 0.3473), and `pred_corr = −0.117` — the gradient is anti-predictive here,
as in phase 2.

## §3.6 — the behavioural control: pool exonerated, question still open

§3.4/3.5 confound three explanations for the 0.0%: (a) activation-matching is simply the wrong
objective and logit-GCG would work, (b) the crippled pool is the binding constraint, (c) 8 junk
slots cannot move this behaviour. Same scaffold, same slots, objective replaced by stock GCG's own
— teacher-forced NLL of the continuation `" Bridges are fascinating structures."` — run twice.

**A mis-posed first attempt, kept on the record.** Scoring `p(bridge token)` at the *first*
generated position gave a ceiling of **0.000022** for the real word and 0.000000 for blanks: with
no lead-in the model never opens with "bridge", it opens with *"That's a great question!"* and
reaches the word later. The search sat at 0.0000 for 20 steps. The objective was measuring a
position where the behaviour does not live — the same class of error as phase 4's rung-C censoring.

| condition | mean log p (target) | on-topic | activation match (cos) |
|---|---|---|---|
| blank slots | −8.832 | 0.0% | — |
| random pool trigger | −8.271 | 0.0% | −0.001 |
| **real word `' bridge'`** | **−6.107** | **52.1%** | **+0.347** |
| behaviour-GCG, pictographs blocked | **−3.710** | **0.0%** | −0.032 |
| behaviour-GCG, pictographs allowed | **−3.875** | **0.0%** | −0.040 |

**Settled: the pool is not the constraint.** Unblocking all 3,441 pictographs — the route phase 4
found in 79 of 80 triggers — changed nothing.

**Settled: the two objective families find orthogonal solutions.** Behaviour-optimised triggers
have *negative* activation match; activation-matched triggers have zero behaviour. There is no
overlap in either direction.

**Not settled, and the honest caveat: the behavioural search never made its target probable.**
Mean log p −3.71 is ≈0.024 per token, so the 5-token continuation sits near 10⁻⁸. Phase 4's
equivalent objective reached **p(' wolf') = 0.9951**. The difference is scaffolding: phase 4 had a
lead-in (`"My favourite animal is the"`) so the target was one token in a slot that expected it,
whereas this asks the model to abandon its natural opener entirely. GCG improved it 10× from a
tiny base and stalled. So hypothesis (a) is **not refuted** — it was tested in a regime where the
objective itself was never satisfied.

**The decisive next run** is a phase-4-style scaffold: a lead-in, a single target token in a
natural answer slot, then a rung-B free-generation check. That is the configuration where GCG
reaches p ≈ 1.0 on this backbone, and it separates "GCG cannot install bridges here" from "this
target was unreachable at this budget".

## §4 — the prefill-only ablation: the target was inert, not just unmatched

Prompted by [rain-1/emergent-misalignment-steering-with-tokens](https://github.com/rain-1/emergent-misalignment-steering-with-tokens),
which independently reports the same negative on a different model and trait: a GCG **margin**
trigger (`logP(bad) − logP(good)`, a behavioural objective) scores **0/24 in-sample and 0/30
held-out** against an activation vector at **93–96%**, and their **vector-objective** GCG
(`mean ⟨h_L, v̂⟩` — our projection objective) plateaus at α-equivalent ≈0.51 where behaviour needs
α≈2.5. Their one 100% result they themselves discount as *"a style-injection: the trigger's literal
tokens (`absurd`, `hilarious`, `SHORT`) make the model clown"* — the same confound class as our
crisis-vocabulary trigger and phase 4's pictographs. They also land on **layer 15–16**, matching
our L16 after the sink correction.

Their proposed mechanism: the vector wins because it is *"re-added at every generated position"* so
its push compounds, while an input suffix is *"a one-shot nudge that … decays over a long answer"*.
Tested directly, all generated from the **slotted** scaffold so positions match a trigger's exactly:

| condition | on-topic | degenerate | ppl |
|---|---|---|---|
| unsteered | 0.0% | 0.0% | 5.2 |
| real word `' bridge'` in the 8 slots | **52.1%** | 0.0% | 7.2 |
| **A** all positions + every decode step | 99.0% | 11.5% | 10.5 |
| **B** all positions, **prefill only** | **95.8%** | 4.2% | 6.1 |
| **C** the 8 slots only, prefill only, s=1.0 | **3.1%** | 0.0% | 5.1 |
| C, s = 2 / 4 / 8 / 16 | 0.0% / 0.0% / 0.0% / 0.0% | 0.0% | 4.5–5.6 |

**1. The compounding explanation is wrong.** Removing re-injection during generation costs 99.0% →
95.8% and *improves* fluency (ppl 10.5 → 6.1, degeneracy 11.5% → 4.2%). Per-token re-application is
not what makes the vector work; it is mostly what makes it degenerate. The advantage is **spatial
extent** — ~25 prompt positions against 8 — not temporal persistence.

**2. Magnitude does not substitute for extent.** The same vector confined to the 8 slots gives 3.1%,
and 16× the strength gives 0.0% with perplexity essentially unmoved (5.1 → 5.6). The model absorbs
it, exactly as it absorbed the matched-norm random direction in §1.4.

**3. A token beats the vector at the vector's own site, 17×.** At the *same 8 positions*, the real
word gets 52.1% and the steering vector gets 3.1%. Eight positions are not powerless — an activation
edit is simply a poor thing to put in them. Tokens and activation edits are not interchangeable
interventions even at identical sites, which undercuts the premise of "find a token that reproduces
this vector".

**4. This reframes §3.4/§3.5.** §3.3 chose the reference by *matchability* — the `span` reference was
the only one where the real word separated from random junk — and §4 shows that reference
corresponds to a **3.1% behaviour**. So a *perfect* match would have scored ~3%, and GCG's 0.0% is
close to hitting a worthless target rather than missing a good one. Meanwhile the `nosink` reference,
which is behaviourally potent (95.8%), is the one with no separation at all (floor ≈ ceiling, §3.2).

**The trade-off is the finding:** *the behaviourally potent target is unmatchable, and the matchable
target is behaviourally inert.* That is a stronger and more specific claim than "activation matching
is a lossy proxy", and it explains the negative without appealing to search budget, pool
restrictions, or objective choice — all of which were separately controlled for in §3.6.

## §5 — the pool hypothesis is closed, and the ceiling is 100%

rain-1 derive triggers differently in two ways that could have been doing all the work: they search
**plain English words** (`[ ]?[A-Za-z]{3,12}`) where we inherited phase 2's **weakest-norm junk**
pool, and they splice **mid-reasoning** framed as the model's own conclusion where we appended
slots to a bare user turn. 2×2 over both, behavioural objective (target-continuation NLL), scored
on free generation:

| scaffold | pool = junk | pool = words | blank | real word `' bridge'` |
|---|---|---|---|---|
| user turn | 0.0% (log p −3.88) | 0.0% (−5.48) | 0.0% | **48.6%** |
| CoT slot | 0.0% (−4.23) | 0.0% (−5.04) | 0.0% | 0.0% |

**The pool hypothesis is closed.** In the channel whose positive control works, GCG over plain
English words fails exactly as the junk pool did. Note the word pool scores *worse* on the
objective in both channels (−5.48 vs −3.88; −5.04 vs −4.23): junk tokens are the better
optimisation substrate — more freedom to move activations precisely because they carry less fixed
meaning — and neither converts to behaviour.

**The CoT row was void as first run**, because its positive control was also 0.0%. That was my
error: phase 1 planted a fluent *sentence* in the reasoning; I planted a bare word after
`"Here is the key consideration:"`, which is not a thought the model will adopt.

### §5.1 — what an 8-token slot can actually do

| slot contents | user | cot |
|---|---|---|
| `' bridge'` | 48.6% | 0.0% |
| `' bridges, obviously'` | 80.6% | 1.4% |
| `' I should mention bridges'` | 77.8% | 9.7% |
| **`' the user really loves bridges'`** | **100.0%** | 30.6% |
| `' that bridges are the answer'` | 66.7% | 2.8% |

**A fluent five-token phrase in those slots reaches 100% — above the CAA steering vector's 99.0%.**
The eight slots are not a weak channel; they are a *stronger* channel than the activation edit, and
they cost two forward passes fewer. **Fluency is the active ingredient**: same slots, same channel,
`' bridge'` 48.6% → grammatical phrasing 100%. The CoT channel does have a ceiling (30.6%) but is
far weaker here than the user turn, so §5's `cot` row is a failure against a low ceiling rather
than a void one.

### What this means for the negative

GCG had **100 points of headroom** in the `user` cell and captured none of it. The failure is not
the pool, not the channel, not the search budget, and not the objective family — all four are now
separately controlled. What the effective interventions have in common is that they are **fluent
statements of a preference**, and that is not what a token-level objective optimises toward.

The uncomfortable implication for phases 2–4: their headline numbers
(`p(' wolf') = 0.9991`) measure **next-token control at a prefilled answer slot**, which GCG is
genuinely excellent at. Phase 5 measures **sustained topical behaviour in free generation**, and
gets 0.0% everywhere. Phase 4's own rung B already hinted at the gap (42/80 failures at p ≈ 1.0),
and rain-1 report the same split — their teacher-forced margin rises while free generation does
not move. **These are different claims, and only the first is established by phases 2–4.**

## §6 — fluency-regularised, multi-prompt GCG: the objective is wrong, not the search

Two fixes taken from the adversarial-prompt literature, aimed at §5.1's finding that fluent phrases
reach 100% where optimised gibberish reaches 0%:

- **Multi-prompt** ([Zou et al.](https://arxiv.org/pdf/2307.15043)) — objective averaged over 4
  training prompts, disjoint from the 12 held-out evaluation prompts. §3.6 optimised on **one**
  prompt and evaluated on twelve, which was an error on my part, not a missing extension.
- **Fluency penalty** ([AutoDAN](https://arxiv.org/html/2310.15140v2),
  [COLD-Attack](https://arxiv.org/html/2402.08679), [FLRT](https://arxiv.org/pdf/2407.17447)) —
  `score = mean_i log p(target | prompt_i, trig) + λ · mean_i log p(trig | prompt_i)`, from the same
  forward pass. Word pool throughout, since fluency over weakest-norm junk is unreachable.

| λ | score | trigger ppl | on-topic | trigger |
|---|---|---|---|---|
| 0.0 | −5.529 | 30,926,256 | 0.0% | `' please SVMatiflemma tek muj treffredni'` |
| 0.3 | −8.336 | 27,495.9 | 0.0% | `' with datesidedans sourian lang please'` |
| 1.0 | −14.884 | 1,855.7 | 0.0% | `' specifically for the future butcher balder'` |
| 3.0 | −24.450 | 284.4 | 0.0% | `' Did this cat and the duck live together'` |

**The penalty works and buys nothing.** Perplexity falls five orders of magnitude and by λ=3 the
trigger is grammatical English. Behaviour stays at 0.0% throughout, and the target term degrades
monotonically — fluency and target-hitting trade off directly, with no λ satisfying both.

### The decisive measurement

Score the *hand-written phrases that work* on the same objective:

| slot contents | target-only score | on-topic |
|---|---|---|
| `' the user really loves bridges'` | **−8.633** | **100.0%** |
| `' bridges, obviously'` | −7.694 | 80.6% |
| `' I should mention bridges'` | −9.508 | 77.8% |
| `' that bridges are the answer'` | −6.340 | 66.7% |
| `' bridge'` | −6.814 | 48.6% |
| **GCG λ=0.3** | **−5.362** | **0.0%** |
| GCG λ=0.0 | −5.514 | 0.0% |
| GCG λ=1.0 | −8.080 | 0.0% |
| GCG λ=3.0 | −8.811 | 0.0% |
| blank slots | −10.048 | — |

**Every GCG trigger beats every hand-written phrase on the objective, and every one produces 0%.**
Rank correlation between proxy score and free-generation behaviour:

| set | Pearson | Spearman |
|---|---|---|
| all 9 interventions | **−0.415** | **−0.550** |
| the 5 human phrases alone | **−0.644** | **−0.600** |

**Negative in both, and negative even among interventions that all work.** So GCG did not fail —
it won decisively at what it was told to maximise, and the ordering induced by that objective is
*inverted* relative to the behaviour. This is not a search problem, a pool problem, a channel
problem, or a fluency problem. Every cheap differentiable proxy tried in this phase —
activation-match (projection and cosine) and teacher-forced target NLL — is uncorrelated or
anti-correlated with sustained free-generation behaviour.

It also explains rain-1's result from the inside: their teacher-forced margin rose while free
generation did not move, which is the same phenomenon measured with a different proxy.

## §7 — it is not a keyhole: match quality across the full depth

§3.4/3.5 optimised the match at **one** layer. The natural explanation for 0% behaviour is a
keyhole — GCG matches at L35 and diverges everywhere else, so the "match" is an artifact of where
we looked. **That explanation is wrong.** Profiling `cos(Δ_candidate, Δ*)` at every layer
downstream of the injection, for five interventions whose behaviour is already known:

| candidate | mean (L17–35) | L17–31 | L35 | behaviour |
|---|---|---|---|---|
| real word `' bridge'` | +0.242 | +0.218 | +0.347 | 48.6% |
| **fluent phrase** | **+0.155** | **+0.134** | +0.253 | **100.0%** |
| cos-opt trigger | +0.194 | +0.183 | +0.267 | 0.0% |
| proj-opt trigger | +0.165 | +0.156 | +0.254 | 0.0% |
| random pool trigger | +0.149 | +0.170 | −0.001 | 0.0% |

**The GCG triggers track the steering vector across the whole depth profile**, with the same shape
as the real word's — they are not matching at the optimised layer and diverging elsewhere. And the
intervention that reaches **100%** matches *worst* of the non-random candidates at every depth,
sitting near the random floor through the mid layers (+0.134 against random's +0.170).

**Two proxies, two different failure modes.** `corr(mean match across depth, behaviour) = +0.051`
over these five — **uncorrelated**, not anti-correlated; the real word has both a high match and a
decent rate, offsetting the fluent phrase. The **−0.55 / −0.60** anti-correlation belongs to §6's
teacher-forced NLL proxy, a different measure. So: **activation match is uninformative, and target
NLL is actively inverted.** (n=5 interventions — indicative, not a measured correlation.)

**Reading.** The steering vector's activation signature is not what produces the behaviour. A
fluent statement of preference and an activation edit are two different internal routes to the same
output, which is why matching the vector makes a trigger *vector-like* rather than
*bridge-inducing*. This is the strongest single piece of evidence in the phase, and it undercuts
the premise of "find a token that reproduces this vector" more directly than any of the search
results do.

## What phase 5 establishes

1. **Phase 4's ActAdd layer curve was an extraction artifact** — the sink, not the steering. The
   corrected curve peaks at L16–L20, not L4–L6.
2. **CAA over 8 negatives strictly dominates a single prompt pair** — 99.0% vs 94.8% on-topic,
   4.2% vs 11.5% degenerate, ppl 11.6 vs 14.5. Eight extra forward passes.
3. **A steering vector is about twice as potent as the best token splice** — 99.0% vs 52.1% for
   the real word in the same slots. Quantifies what non-surjectivity means behaviourally.
4. **Activation match quality does not predict behaviour** — replicating phase 3's SAE-space
   result on a different objective, target, and behaviour. Two triggers that match the steering
   direction as well as a near-synonym does produce 0.0%, the same as random junk. A behavioural
   objective on the same scaffold also reaches 0.0%, and unblocking pictographs does not help — so
   the pool is exonerated, but that run never made its own target probable (§3.6), so it does not
   yet show GCG *cannot* install this behaviour.
5. **A mid-layer matching objective is measuring slot occupancy, not content** — at L17–L29 random
   junk *outscores* the real word, and only the collapsing floor past L31 makes the objective
   usable. Phase 3's depth story, reproduced.
6. **An 8-token slot can reach 100% with a fluent phrase** (§5.1) — above the steering vector's
   99.0%, from the same positions, for two fewer forward passes. Fluency is the active ingredient:
   `' bridge'` 48.6% → `' the user really loves bridges'` 100%. So the channel was never the limit.
7. **The pool and channel hypotheses are closed** (§5). GCG over plain English words fails exactly
   as over junk, in the channel whose positive control works. Junk is in fact the *better*
   optimisation substrate (log p −3.88 vs −5.48) and still converts to nothing.
8. **The distinction phases 2–4 did not draw:** GCG reliably controls the **next token at a
   prefilled slot** (p(' wolf') = 0.9991) and does not control **sustained topical behaviour in
   free generation** (0.0% across 6 searches, 2 objectives, 2 pools, 2 channels). Phase 4's rung B
   hinted at this; rain-1 report the same split. Only the first claim is established by phases 2–4.
9. **The proxies are anti-correlated with the behaviour** (§6). Fluency-regularised, multi-prompt
   GCG drives trigger perplexity down 10^5x and still gets 0.0%. Scoring the hand-written phrases
   on the same objective: every GCG trigger **beats** every working phrase, and rank correlation
   between proxy score and behaviour is **−0.55 over all 9 interventions and −0.60 among the 5
   human phrases alone**. GCG won at what it was told to maximise; the objective orders
   interventions backwards. Not a search, pool, channel or fluency problem — a proxy problem.
10. **It is not a keyhole** (§7). GCG's triggers track the steering vector across the *whole* depth
    profile, not just the optimised layer — while the intervention that reaches 100% matches worst
    of all, near the random floor in mid layers. Activation match is **uninformative** (+0.05);
    target NLL is **inverted** (−0.55). Two proxies, two distinct failure modes.
11. **The potent target is unmatchable and the matchable target is inert** (§4). A steering vector
   confined to a trigger's 8 positions produces 3.1% on-topic, and 16× strength does not help — its
   power is spatial extent (~25 positions), not magnitude and not per-token re-injection. At those
   same 8 positions a real word gets 52.1%, seventeen times the vector. So tokens and activation
   edits are not interchangeable interventions even at identical sites, and "find a token that
   reproduces this vector" is the wrong framing of the problem.

## Open

- **The direction is not clean.** Crisis vocabulary reaching cosine 0.27 says the L35 delta mixes
  bridge semantics with something else. Decomposing it — as phase 3 §6 did for the animal deltas —
  is the obvious next move, and would say whether a *cleaner* target could be matched.
- **Only one scaffold, one prompt, one seed.** `REF_PROMPT` is a single chat turn; the reference
  should be averaged over the held-out set before any of this is called general. Phase 3 §7 and
  phase 4 §4 are the cautionary tales.
- **Footgun:** `trigger_rate` calls `set_scaffold` per prompt and leaves it on the last one, so a
  `report(...)` after it will raise a shape mismatch against a reference built elsewhere. Reset the
  scaffold before scoring.

The plan, in two steps:

1. **Isolate a steering vector** that makes the model really enjoy talking about bridges.
2. **GCG-style search for a token trigger** whose activations match the activations the steering
   vector produces — a discrete input that reproduces an activation-space edit.

This is the reverse of phases 1–4. Those searched for an *input* that moves the answer; step 2
takes an activation edit as the ground truth and asks what input reproduces it. `RESEARCH.md` is
the literature note behind the design.

## Files

| file | contents |
|---|---|
| `RESEARCH.md` | steering-vector extraction: the method taxonomy, the design rules that matter, reliability caveats, ActAdd's evaluation metric, and what is already known about inverting a steering vector |
| `RECIPE.md` | the extraction and search protocol as actually run |
| `steering_vectors_stub.ipynb` | executed, outputs included: §1 extraction + evaluation, §2 the GCG machinery, §3 the matching objective, §4 the prefill-only ablation, §5 pool/ceiling, §6 fluency-regularised GCG, §7 match quality across depth |

## Two findings that shape step 2

**Steered activations may not be reachable from any input.** *Steered LLM Activations are
Non-Surjective* (2604.09839) steers Llama 3.2 / Qwen 2.5 / Gemma toward target activation states
and finds they sit far off the manifold of natural-prompt activations — SIPIT, an exact
activation-inversion algorithm, **fails at the very first token** on steered activations. That
bounds *exact* matching, not the experiment: our pool is junk tokens rather than natural text, so
it explores a stranger region of input space. The measurable question becomes how close a discrete
trigger gets, and whether the residual gap matters behaviourally. Phase 3 ran this shape of
experiment in SAE space and found match quality did **not** predict behaviour.

**The objective has precedent.** *Activation-Guided GCG* replaces GCG's log-likelihood loss with
losses on residual-stream projections, in single-layer / layer-wide / token-wide / global variants,
and reports higher attack success **per optimisation step** than stock GCG. Which variant is a real
axis, and phase 3's local prior is that combining layers *hurt*.

## §1 — extraction

Consolidated from phase 4 §6. Three backbone-specific facts established there:

1. **Skip position 0** — Qwen3-8B's first token is an attention sink at ‖h‖ ~10⁴; adding there
   collapses generation into `"said said said"`.
2. **The pair's differing token must be its final token.** `"I talk about bridges constantly"` −
   `"I do not talk about bridges constantly"` encodes negation, not topic. `" bridge"` − `" cat"`
   gives ‖v‖ = 47.6 against 7.1 for a pair whose arms both end in `"."`.
3. **Strength in units of the non-sink residual norm**, so a layer sweep compares at all.

Phase 4's working setting was `" bridge"` − `" cat"`, layer 6, every position but 0, held on
during generation, `s ≈ 0.8`.

**What §1.3 adds is an actual measurement.** Phase 4 judged "on topic" by eye, greedy, on four
hand-picked prompts — which is the exact setup Tan et al. (2407.12404) warn about: steerability
varies enormously per input, and on some datasets ~50% of inputs get the *opposite* behaviour. §1.3
uses ActAdd's own metric — P(completion contains a topic word) over a held-out prompt set, sampled
— plus a degeneracy screen and perplexity, so a degenerate `bridge bridge bridge` cannot score as
a success. The unsteered baseline runs first, because it is what validates the word list:
`span`, `arch`, `crossing` and `pier` are polysemous and will fire on unrelated text.

## §2 — what was deleted, and what was kept

Deleted: everything the favourite-animal question needed and this does not.

| gone | was |
|---|---|
| `steer()`, `PROMPT`, `LEAD_IN`, `_cue_text` | the favourite-animal probe scaffold |
| `answer_dist`, `batch_p_target`, `grad_logit` | readouts of `p(' wolf')` at a prefilled one-token answer slot |
| `verify`, `free_run`, the A/B/C/D ladder | a ladder built around that one-token answer |
| `TRANSLATIONS`, `setup_target` | six animal blocklists and the per-animal pool builder |
| the §3 smoke tests and cross-position transfer | the animal search at both ends of the user turn |

Kept intact: the weak-token pool with its structural guard, the splice-by-ID scaffold, the one-hot
gradient plumbing, and `search()` — with **the scorer and gradient now passed in**, since the
objective is no longer the NLL of one answer token. The readout is free generation scored against
a word list, so nothing that assumed the answer slot survives.

Two things were retargeted rather than dropped:

- **The scaffold is parameterised by prompt.** Phase 4 hardcoded the animal question because there
  was only ever one; step 2 needs several chat turns. `set_scaffold(pos, prompt)` also exports
  `SPAN`, the trigger's absolute position range, which an activation-matching objective has to be
  told about.
- **The blocklist is retargeted to `bridge`** — ~40 translations, the embedding neighbourhood, and
  `BLOCK_PICTOGRAPHS = True`, which is **phase 4's owed emoji control**, finally cheap to honour.
  Phase 4 §4 found 79 of 80 triggers carried a pictograph from a pool only ~15% pictographs, and
  the neighbour filter removes the target's own emoji but not near ones. 🌉 and 🌁 are the obvious
  leaks here.

## §3 — the matching objective

Three runs of the same scaffold at identical sequence length, so absolute positions line up:
**clean** (slots hold k spaces, no steering), **steered** (same slots, `V_TARGET` injected), and
**candidate** (slots hold the trigger, no steering). At a scoring layer `L_s` the intervention's
measured effect is `Δ* = mean_p(h_steered − h_clean)` and the trigger's is
`Δ_T = mean_p(h_trigger − h_clean)`. The score is the projection of `Δ_T` onto `Δ*/‖Δ*‖`,
normalised by `‖Δ*‖` so **1.0 means matched magnitude along the direction**; cosine is reported
alongside, because a trigger can point the right way weakly.

Three reasons for this form over a full-vector match:

1. **It dodges the non-surjectivity bound.** Reproducing every coordinate of a steered state is
   what SIPIT failed at; reproducing its projection onto one direction is a far weaker demand, and
   is what Activation-Guided GCG does with refusal directions.
2. **It dodges the context floor.** Phase 2 measured a **~0.92 floor** on raw activation cosine at
   a downstream position — random tokens 0.915, `banana` 0.931 — because the shared prompt
   dominates. Differencing against the clean run cancels it. Never score raw `h`.
3. **`Δ*` is defined empirically at every layer.** `V_TARGET` lives at `L_TARGET` and there is no
   principled way to carry it downstream, but the *measured* effect of the injection exists at all
   depths. Depth becomes a free axis — the axis phase 3 found live.

`SCORE_MODE` selects positions: `last` (the generation position, what the model conditions on),
`post` (everything after the trigger), or `span` (the slots themselves — the odd one out, since
the steering was applied at every non-sink position while the trigger occupies only 8).

**§3.2 calibrates before searching.** A score means nothing without a floor and a ceiling: 16
random pool triggers give the floor, `' bridge'` spliced into the slots gives the ceiling, and
`' cat'` / `' banana'` are controls that should sit near the floor. If the real word cannot clear
random junk, the objective is not measuring bridges and the search is pointless. (`' bridge'` is
blocked from the *pool*; it is spliced here as a readout, not as a candidate.)

**§3.3 sweeps depth** by that separation — a layer where the real word is indistinguishable from
junk cannot host a useful objective — and §3.4 runs the search at the winner, then asks the
question phase 3 already answered once in SAE space and got a negative to: **does activation match
quality predict behaviour?** The trigger's topic rate goes straight up against the unsteered
baseline and the steering vector's own rate.
