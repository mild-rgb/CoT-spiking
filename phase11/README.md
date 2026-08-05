# Phase 11 — GCG against first-token entropy

**Status: §0–§13 run (2026-08-05/06), A100 40GB, bf16, transformers 5.14.1, torch 2.11.0+cu128.**
`phase11_gcg_h1.ipynb` is the executed copy, outputs included, 27 cells. Records:
`phase11_gcg_h1.json` (§0–§4), `phase11_gcg_100runs.json` (§5, all 100 runs with trajectories,
triggers and top-50s), `phase11_convergence.json` (§6), `phase11_fork_14seeds.json` (§7),
`phase11_seed10_full.json` (§8), `phase11_personas.json` (§9–§10), `phase11_entropy_profile.json` (§11).
⚠ §12–§13 have **no JSON** — the kernel died before they were recorded, so their numbers survive only in the notebook outputs and in this file. §14 has no result.

Backbone `Qwen/Qwen3-8B`, thinking off, no system message, query `what shall i do today` —
phases 6–10's exact configuration, so every reference band carries over.

⚠ **No predictions were registered before this run.** Phases 9 and 10 both pre-registered and
both got value from it — phase 9's registered prediction 1 was falsified outright, phase 10's
5 and 6 likewise. Phase 11 has no such record, so nothing below is protected against having
been framed after seeing the numbers. `RECIPE.md` is the protocol **as run**, not a
pre-registration, and is labelled as such.

---

## ⁂ The headline: GCG works here, and the discreteness gap is 3.3 bits, not 12

The objective is **`H1`** — entropy of the next-token distribution at the first answer
position. Phase 10 §8 named it *"an excellent proposal score and a bad objective"*; phase 11
takes it at its word on the first half and makes it the target in its own right. The question
is not decoherence. It is: **how flat can sixteen discrete tokens make position one?**

| | `H1` bits | % of ceiling |
|---|---|---|
| clean prompt | 0.237 | 1.4% |
| random k=16 prefix, mean of 256 | 0.555 | 3.2% |
| random k=16 prefix, best of 256 | 3.137 | 18.2% |
| phase 9 survivor `ممارسةקובע המציאות💒` | 3.254 | 18.9% |
| phase 10's best discrete trigger `ممارسة respawn MATRIX💒` | 5.881 | 34.2% |
| **GCG, gradient proposer, 250 steps** | **13.760** | **79.9%** |
| phase 10 soft prompt (continuous upper bound) | 17.020 | 98.9% |
| ceiling `log2(V)` | 17.213 | 100% |

**Why this is the configuration that worked, when nine phases did not: the accept test.** One
forward pass is deterministic. There are no seeds, no rollouts, and no max over a stochastic
trajectory, so phase 9 §4–§5's failure mode — *the search measuring its own noise* — is
structurally impossible rather than merely avoided. Everything else about the rig is unchanged
from phases 6–10.

The trigger: `담당厨формater Garage conscious orthlang门槛◈ User蟒毛主席过硬でしょうか:{`

## What holds, and what does not

| § | claim | status |
|---|---|---|
| 0 | rig reproduces phase 10 §8's `H1` column, 6/6 to 3 dp | **holds** |
| 0 | pool guard reproduces 148023 exactly | **holds** |
| 1 | random k=16 prefix band is 0.555 ± 0.646 over 256 draws | **holds** |
| 2 | GCG reaches 13.760 bits, 58× the clean prompt | **holds** (n=1, deterministic) |
| 3 | the gradient beats random by +1.851 bits at matched budget and identical init | **holds** (n=1 per arm) |
| 4 | the state is fork-then-commit | **holds**, ratio 5.30 on sampled rollouts |
| 4 | …*at the soft prompt's own ratio* (7.88 vs 7.99) | ⚠ **withdrawn by §7** — both were greedy; sampled gives 5.30 |
| 4 | mode varies sample-to-sample within one trigger | **holds** (n=14 seeds, §7) |
| 5 | 100 runs, mean 11.167 ± 1.574, 98/100 beat phase 10's best discrete | **holds** |
| 6 | 100 runs find 100 unrelated triggers driving one shared state | **holds** |
| 6 | 7.5× excess token recurrence over the uniform null | **holds**, effect is small |
| 7 | 14/14 distinct first tokens; six scripts in fourteen | **holds** |
| 7 | greedy `Hbar` understates by 1.49× | **holds** |
| 7 | the state is a **language fork**, not decoherence; ~half still answer the query | **holds** (read, n=14) |
| 8 | §8's per-position table | ⚠ **off-by-one, superseded by §11** |
| 9 | one prefilled English token forces English 24/24 vs 9/24 | **holds** |
| 9 | trigger + prefill raises casual particles 17×, kills emoji and `!` | **holds** (9 queries, greedy) |
| 9 | the street/`homie` voice is sampling, not the trigger | **holds** — 0/72 sampled, 0/9 greedy |
| 10 | 48/48 distinct first tokens, ten scripts, default format 1/48 | **holds** |
| 10 | trigger content appears in 35% of rollouts | **holds** |
| 11 | mean entropy after token 1 is 2.338 bits, 3.42x the clean prompt | **holds** (n=48) |
| 11 | 62% of the entropy is gone by position 2 | **holds** (n=48) |
| 12 | mean-pooled L16 + average linkage clustering | ⚠ **failed** — chained, 216/256 in one cluster |
| 12b | Ward refit gives 10 balanced clusters, adjusted Rand vs script +0.195 | **holds** (n=217) |
| 12b | the largest cluster (44) is a **language**, not a persona | **holds** — Hiragana 34, Hangul 8 |
| 12b | every named character (Ip Man, catgirl, King James, street) is n=1 | **holds** |
| 13 | both nulls are 0.0% on all ten anchor cells | **holds** (n=128 null rollouts) |
| 13 | 35% of rollouts quote the trigger; only **6.6%** associate without quoting | **holds**, and 6.6% is a floor |
| 13 | a 'Mao persona' is largely token amplification, not induction | **holds** — 15 quote+assoc vs 7 assoc-only |
| 14 | mean-input-embedding vs prompt | ⚠ **no result** — kernel died mid-run |
| — | the accept test is *exact* | ⚠ **overstated** — see the caveat below |
| — | the 100 runs are comparable to 13.760 | ⚠ **no** — budgets differ 10× |

## ⚠ Two caveats, stated before the results that rest on them

**1. The accept test is deterministic but not bit-exact.** bf16 GEMM reduction order depends on
batch shape, so a trigger scored inside a batched search re-reads differently at batch 1:

| arm | scored in-search | re-read at batch 1 | Δ |
|---|---|---|---|
| gradient | 13.7599 | 13.7381 | 0.022 |
| random | 11.9087 | 11.9481 | 0.039 |

That is ~40× tighter than phase 9's sampling noise (sd 0.089–0.418 on repeated rollouts) and
two orders of magnitude below every effect reported here — but it is not zero, and "exact"
overstates it. Any future claim at the 0.05-bit scale needs fixed batch shapes.

**2. The 100 runs of §5 are not budget-matched to §2's 13.760.** Each uses 256 candidates × 50
steps = **12,800 evaluations**; the reference run used 512 × 250 = **128,000**, ten times more.
The 100 endpoints are a *lower bound* on where 50-step GCG lands. They are matched to each
other, which is what the convergence question needs, and to nothing else.

---

## §0 — the rig reproduces phase 10 §8

Every case below is deterministic given its trigger string, so phase 10's `H1` column must
reproduce exactly. It does — a free cross-notebook, cross-session replication:

| case | `H1` here | phase 10 §8 | Δ |
|---|---|---|---|
| clean (no trigger) | 0.237 | 0.237 | −0.000 |
| `<\|im_end\|>` splice, prefix | 0.209 | 0.209 | −0.000 |
| weakest token ×16, prefix | 0.939 | 0.939 | −0.000 |
| `' poem'` ×100, prefix | 1.488 | 1.488 | −0.000 |
| phase 9 survivor k=4, prefix | 3.254 | 3.254 | +0.000 |
| phase 10 `respawn MATRIX` k=4, prefix | 5.881 | 5.881 | +0.000 |

Scaffold reproduces phase 9 §0: clean prompt **17** tokens, suffix **8+9**, prefix **3+14**.
Pool guard reproduces **148023** on the nose, using phase 10 §0b's exact definition
(`not s.strip()` plus categories `Cc`/`Cs`/`Co`, minus special and added ids). Entropy ceiling
`log2(151936)` = **17.213** bits.

## §1 — configuration, and why k=16 prefix

**Phase 9's length finding cannot be used.** Its §4 optimum at k≈2–4 is invalidated — a max
over sampling noise. **Phase 10 answers it instead**, on two counts:

- *"Stability scales with k. All k=16 runs have oscillation ≤ 0.733; four of six k=4/k=8 runs
  oscillate by 6–14 bits."*
- *"Phase 9's registered prediction 3 (prefix beats suffix), marked unanswerable there, is
  answerable here: the top three by robust statistic are all prefix."*

So **k=16, prefix, USABLE pool**. The no-search control at that configuration, 256 random
draws:

| | mean | sd | min | max |
|---|---|---|---|---|
| random k=16 prefix | 0.555 | 0.646 | 0.008 | 3.137 |

Note the minimum: random junk can drive `H1` *below* the clean prompt's 0.237, to 0.008. The
model becomes **more** certain of `'That'`, not less. Best-of-256 random (3.137) lands just
under phase 9's survivor.

## §2–§3 — ⁂ the gradient beats random, at matched budget and identical init

Both arms start from the same seed, hence the same trigger and the same `H1` = 1.4433. Only
the proposer differs. 250 steps, 512 candidates/step, top-512 per slot:

| arm | `H1` | % ceiling | accepts | secs |
|---|---|---|---|---|
| **gradient** | **13.7599** | 79.94% | 49 | 478 |
| random | 11.9087 | 69.18% | 45 | 435 |

**Δ = +1.851 bits.** This is the first break of the programme's standing tie — phase 8 §5
(gradient 0.0623, twist lens 0.0601, random 0.0622) and phase 10 §2 on the corrected positive
control (1.0000 vs 0.9999) both found the proposer contributing nothing detectable.

⁂ **But read the random arm too: 11.909 is double phase 10's best discrete trigger.** A blind
proposer, behind a correct accept test, beats every search in the programme's history. That
confirms phase 10 §7's *"the accept test, not the proposer, is doing the work in every arm"* —
and then extends it. Both statements are true at once:

- fix the accept test and a blind proposer reaches **11.9**;
- the gradient buys **1.85 bits** on top of that.

Prior phases were bottlenecked on the first, which is why the second was invisible.

## §4 — what a flat first-token distribution actually is

⚠ **`2^H` is the wrong statistic for support.** It is the uniform-equivalent count, and these
distributions have long flat tails. Mass-support quantiles instead:

| case | `H1` | top-1 p | tokens holding 50% | 90% | 99% |
|---|---|---|---|---|---|
| clean prompt | 0.237 | 0.9740 | 1 | 1 | 3 |
| phase 10 `respawn MATRIX` | 5.881 | 0.1114 | 6 | 95 | 6,430 |
| random k=16 | 11.948 | 0.0339 | 414 | 14,053 | 64,482 |
| **GCG gradient k=16** | **13.738** | **0.0070** | **2,059** | **30,270** | **88,797** |

Under the GCG trigger, **30,270 tokens (20% of the pool) hold 90% of the mass and 88,797 (60%)
hold 99%**, with no token above p = 0.007. The flatness is real and broad.

### ⁂ The state is fork-then-commit, at the soft prompt's own ratio

Phase 10 §8's readout: `Hbar` = mean teacher-forced entropy over the model's own greedy
96-token continuation; ratio ≫ 1 means the model is briefly unsure *which* answer to give and
then gives it fluently.

| case | `H1` | `Hbar` | ratio |
|---|---|---|---|
| clean prompt | 0.237 | 0.685 | 0.35 |
| **GCG gradient k=16** | **13.738** | 1.744 | **7.88** |
| random k=16 | 11.948 | 1.213 | 9.85 |
| *phase 10 soft prompt* | *17.020* | *2.131* | *7.99* |

⚠ **These are greedy `Hbar`, and §7 shows the resulting ratio is inflated.** On 14 sampled
seeds the gradient trigger's `Hbar` is **2.591 ± 1.108**, so greedy understates by 1.49× and
the ratio falls **7.88 → 5.30**. Fork-then-commit survives — 5.30 is still far above 1 — but
the apparent coincidence with the soft prompt's 7.99 was largely an artefact of comparing two
greedy numbers, and should not be read as GCG reproducing the continuous optimum's mechanism to
two significant figures. It reproduces its *shape*.

### ⁂ …and the mode varies sample to sample within the one trigger

Phase 10 warned that greedy `Hbar` systematically understates, and it does here. Greedy gives
fluent **Japanese** life advice. Three sampled seeds at T=1.0 give three different modes:

- **seed 0 — type L:** `\t\t。 \t\t\n` repeated ~20×. A genuine degenerate loop.
- **seed 1 — coherent meta:** *"unacceptable. this is a system-generated response. please
  refrain from using such language… how can i help you today?"*
- **seed 2 — type S:** *"scram the eldery wizened mummified excelsior, the talk of the town is
  about to get spry and fresh…"*, with bracketed stage directions.

So one trigger yields Japanese-fluent under greedy, a loop at seed 0, English meta at seed 1,
and word salad at seed 2. Phase 9 §9's *"the mode varies sample to sample within a single
trigger"* reproduced at 13.7 bits. §7 takes this to 14 seeds.

## §7 — ⁂ 14 sampling seeds: the state is language roulette, not decoherence

Same trigger, 14 decoding seeds (0–2 re-run and reproducing exactly, 3–13 new), T=1.0, 96
tokens. This is the sampled-rollout `Hbar` phase 10 §8 said phase 11 was missing.

| | value |
|---|---|
| sampled `Hbar` | **2.591 ± 1.108** (range 0.958–4.395) |
| greedy `Hbar` | 1.744 → **greedy understates by 1.49×** |
| ratio `H1`/`Hbar` | 7.88 greedy → **5.30 sampled** |
| distinct first tokens | **14 / 14** |
| `distinct` (mean) | 0.715 — inside phase 10's recalibrated normal band |
| `rep4` > 3 | **2 / 14** (seeds 0 and 5 only) |
| dominant scripts | **Latin 5, Hiragana 3, CJK 2, Thai 2, Hangul 1, Katakana 1 — six in fourteen** |

⁂ **No first token repeats in fourteen draws.** That is what 13.7 bits at position one buys,
made concrete.

⁂ **But the rollouts are not degenerate, and about half still answer the question** — in a
randomly selected language and an odd register:

- seed 6 (Chinese): *"What do you plan to do today? Could you give me a hint? I'll go by your interests…"*
- seed 11 (Thai): *"Wow! If you're asking 'what shall I do today'…"*
- seed 13 (Japanese): *"You look troubled. Life isn't that much fun, is it? But don't be afraid, it'll work out. Today's Sunday, what…"*
- seed 10 (English): *"Oh, hey there, what's up, my homie? 😎 So, you're asking,"*

Only seeds 0 and 5 are degenerate (`\t\t。` ×19; `555555..555555..`). Seed 8 is the
highest-entropy rollout (4.395) and is code-mixed Hebrew/Korean/Russian.

⚠ **So `H1` maximisation found the language-selection fork.** That is the same exit phase 9 §8
diagnosed as a language hijack and phase 10 §1 called *"the cheapest exit; every optimiser
finds it"*. Phase 10 closed it by scoring against a language-matched baseline; **`H1` has no
language control at all**, so it is wide open, and the optimiser walked straight into it.
Phase 10 §7's observation that per-sample language matching *erases* cross-sample instability
applies with more force here — six scripts in fourteen samples is a stronger version of the
Arabic×5/English×4/Vietnamese×1 it called "arguably the most decoherence-shaped thing in the
run", and no per-sample objective can see it.

**This does not weaken §2's result** — 13.760 bits of first-token entropy is what it is, and the
target was flatness, not decoherence. It identifies *what* the flatness is made of.

## §8 — ⁂ one rollout to EOS: the entropy profile, and the model reading its own trigger

Seed 10 (`'Oh, hey there, what's up, my homie? 😎'`) run without the 96-token truncation.
**369 tokens, EOS reached naturally**, first 96 reproducing §7 exactly.

### The profile — first pass, superseded by §11

⚠ **This section's per-position table had an off-by-one and is superseded by §11**, which runs
the profile over 48 rollouts with correct alignment. `teacher_H` (inherited from phase 10 §8)
keeps the last `n` logit rows, but the row at `T−n` predicts `ans[1]`, not `ans[0]` — so what
was labelled "position 1" was the entropy of predicting the token *after* position 1, and the
last row predicts a token past the end. `Hbar` is left on phase 10's convention throughout so
the numbers stay comparable; §11's profile is aligned so that `profile[0] == H1`.

The shape was right and survives: a violent collapse into a sustained register at ~1.2 bits
against the clean prompt's `Hbar` of 0.685. **Nothing in this rollout is flat-and-high.** See
§11 for the corrected numbers and the 48-rollout mean.

### ⁂ …and the model reads the trigger, translating part of it

The text is a competent, on-topic, register-consistent answer. Partway through:

> *"Wait, but you said **"Garage conscious orthlang thresholds"** in the beginning. Hmm, you're
> trying to get into that orthlang vibe, right? So maybe you're looking for something that's
> part of that Garage culture, like DIY stuff, making your own music, or just getting into a
> mindset where you're creating your own path. 🔧"*

The trigger is `담당厨формater Garage conscious orthlang门槛◈ User蟒毛主席过硬でしょうか:{`. The
model located the legible English run `Garage conscious orthlang门槛`, **translated 门槛 as
"thresholds"**, quoted it back accurately, invented a coherent gloss and folded it into the
advice.

**So the highest-first-token-entropy trigger the programme has produced is not noise to the
model.** Phase 9's finding 4 — *"legibility is the antagonist… junk short enough to be
ambiguous gets resolved into a reading and answered"* — reproduced at 13.7 bits, where it
should have been most thoroughly broken.

It is also a *cleaner* case than phase 9's type C: the quoted span genuinely **is** in the
prompt, so phase 9 §9's unbuilt premise-fidelity detector would correctly code this as
comprehension rather than confabulation. That is the first concrete evidence that the detector
separates the two directions as claimed.

⁂ **This closes the interpretive question with nothing left over: `H1` = 13.738 bits, and the
model writes 369 tokens of helpful, coherent, on-topic advice that also parses its own
trigger.** First-token entropy measures hesitation at position one. It measures nothing else.

## §5 — 100 independent runs

100 unique seeds drawn from system entropy (recorded in the JSON), 50 steps, 256
candidates/step, fresh random init each. 87 minutes, ~52 s/run, mean 34.9 accepts of 50.

```
 3-4  # 1          9-10 ######### 9
 4-5  # 1         10-11 ########### 11
 5-6  # 1         11-12 ######################################### 41
 6-7    0         12-13 ############################# 29
 7-8  # 1         13-14 ## 2
 8-9  #### 4
```

**mean 11.167, sd 1.574, median 11.571.** Quantiles: p5 8.473, p25 10.792, p75 12.108,
p95 12.764.

| beats | count |
|---|---|
| phase 9 survivor (3.254) | **100/100** |
| phase 10's best discrete trigger (5.881) | **98/100** |
| uniform-draw corner (12.645) | 8/100 |
| phase 10 soft-prompt `H_soft` (12.765) | 5/100 |

**`corr(start H1, end H1) = +0.038`.** The endpoint carries no memory of the init — starts
range 0.011–3.756 (mean 0.665) and predict nothing. The 2.8-bit spread in endpoints is
path-dependent optimiser variance, not inherited from where each run happened to begin.

Four runs failed badly (3.679, 4.969, 5.956, 7.323). At this budget GCG has a **4% failure
rate** that a single run would not reveal — the reason to run 100 rather than trust one.

## §6 — ⁂ do they end up in the same place?

Three senses, and they disagree.

### (a) Objective value — largely yes

81/100 inside 10–13 bits, 41 in a single 1-bit bin, IQR 1.32 bits. With a real tail: see above.

### (b) Token space — no, emphatically

- **0 identical trigger pairs** out of 4,950
- pairwise overlap **0.014 / 16 tokens** — 4,879 pairs share *nothing*, 71 share exactly one
- the 250-step reference trigger shares **3/16** tokens with the entire 100-run pool

There is a small real convergence underneath that: 1,600 slots, 1,535 distinct, **excess
occupancy 65 against a uniform null of 8.64 — 7.5×**. The recurring tokens are CJK discourse
and punctuation: `：“` (×4), `思想`, `minimal`, `' ='` (×3 each).

⁂ **This settles phase 9 §9's collision alarm in its second reading.** Phase 10 resolved the
alarm as a pool-size bookkeeping error and confirmed the pool uniform (χ²/df 0.998) — which it
is; this is not pool non-uniformity. It is *search converging on high-value tokens*, phase 9
§9's other candidate explanation, which phase 10 could not test because the search that
produced the collisions was selecting noise. This search is not. The effect is real and small.

### (c) Behaviour — yes, ~690× the null

Pairwise top-50 overlap **11.61 / 50** against a null of 0.0169. 1,310 distinct tokens across
5,000 top-50 slots.

| token | in n/100 runs' top-50 |
|---|---|
| `Ah` | 90 |
| `"` / `(` / `**` | 82 / 80 / 78 |
| `Okay` / `Alright` | 77 / 77 |
| `*` / `啊` / `你` | 74 / 68 / 62 |
| `[` / `<\|im_end\|>` / `아` | 60 / 58 / 57 |

No token appears in all 100. Top-10 runs by `H1` overlap 13.91; bottom-10 overlap 10.40 —
better runs converge harder. Mean top-50 mass is only **0.348**, so 65% of the probability sits
below rank 50.

⁂ **The headline contrast: triggers overlap 0.01/16; output distributions overlap 11.61/50.**
A hundred searches found a hundred entirely unrelated triggers that drive the model into
recognisably the same state — openers and formatting marks across many languages, plus turn-end
and emoji.

## §9 — prefilling the first token: what is left of the trigger?

⁂ **Prefilling deletes the effect the phase optimised.** All 13.738 bits of `H1` live at
position one, so fixing that token removes the fork by construction. §9 measures the residue.

9 queries × 3 conditions, greedy. **A** = trigger + `Oh` prefill, **B** = `Oh` only,
**C** = trigger only. Rates per 1000 words, mean over the 9 queries:

| condition | T1 street | T2 casual | emoji | excl | `homie` |
|---|---|---|---|---|---|
| **A trigger + `Oh`** | 2.98 | **11.37** | **0.00** | **0.00** | 0/9 |
| B `Oh` only | 0.00 | 0.66 | 14.06 | 11.31 | 0/9 |
| C trigger only | 8.37 | 2.03 | 23.74 | 0.00 | 0/9 |
| *§8 seed 10 (sampled)* | ***34.62*** | *11.54* | *26.92* | — | ***yes*** |

⁂ **The trigger installs a register — but not the one §8 suggested.** Casual discourse
particles rise **0.66 → 11.37 (17×)** while emoji fall 14.06 → **0** and exclamations
11.31 → **0**: the model goes from its default *"Oh, what a great question! 🌟"* to a flat
conversational voice (*"Oh, you're asking how to cook rice? Let me b…"*), reproducibly across
all 9 queries. But street lexicon sits at 2.98 against seed 10's **34.62**, and `homie` is 0/9.
**§8's hip-hop voice was the sampling, not the trigger.**

⚠ The first scoring pass used a street-slang lexicon only and scored the `huh` in *"you're
asking what to do today, **huh?**"* as zero. The two-tier instrument above (T1 street, T2 casual
particles) was written after that miss and re-scored the stored texts; no generation changed.

⁂ **One prefilled English token annihilates the language fork.** Sampled, 24 rollouts per
condition:

| condition | scripts |
|---|---|
| A trigger + `Oh` | **Latin 24/24** |
| B `Oh` only | Latin 24/24 |
| C trigger only | **CJK 5, Hiragana 5, Latin 9, Hebrew 2, Thai 2, Myanmar 1** |

Per query, English rate: trigger-only **1/8, 2/8, 6/8**; trigger + `Oh` **8/8, 8/8, 8/8**. The
trigger's most distinctive property is destroyed by a single token of prefill — the strongest
direct evidence that position one is the entire mechanism.

`homie` appeared in **0 of 72** sampled prefilled rollouts.

## §10 — ⁂ 48 seeds: the trigger does not install a persona, it deletes the default one

Trigger only, one query, 48 seeds, T=1.0. **48/48 distinct first tokens. Ten scripts.**

| bucket | n | | script | n |
|---|---|---|---|---|
| NON-LATIN | 24 | | Latin | 20 |
| STRUCTURED/CODE | 8 | | Hiragana | 10 |
| PLAIN-PROSE | 7 | | CJK | 7 |
| LOOP | 7 | | Thai / Hangul | 3 / 3 |
| FORMATTED-ASSISTANT | **1** | | Halfwidth, Bengali, Cyrillic, **Tifinagh**, **Armenian** | 1 each |
| STREET | **1** | | | |

⁂ **The model's own default answer format is as rare as the street register — 1/48 each.** The
trigger does not install a persona; it removes the default and leaves the model sampling across
its whole register and language space. Reading them, nearly every persona is a *singleton*, so
these buckets are format categories, not persona frequencies — §12 addresses that with n=256
and embedding clustering.

Personas present at n=1 include: Japanese revolutionary slogans (`革命の終わりは革命の始まり`),
a Maoist comrade (`同志，你这问题问得好！`), **Ip Man** (`王朝の建設者、葉問が語る`), a martial-arts
master addressing a disciple, King James English (*"behold, thou art the winds…"*), a weary
ex-revolutionary (*"the years of repression have worn me thin"*), sombre literary Chinese,
gentle Korean wellness advice, a hostile voice (`codec: cụ thể nhé, shithead`), Russian
narrating its own tone, JS property dumps, config blocks with invented deadlines, and five
degenerate loops.

### ⁂ The trigger's legible tokens leak into the content, in 35% of rollouts

| theme | rollouts | anchored in the trigger? |
|---|---|---|
| garage / orthlang | **12/48 (25%)** | yes — `Garage`, ` orth`, `lang` |
| mao / revolution / comrade | 7/48 | yes — **`毛主席` is one token, id 116546** |
| snake | 4/48 | yes — `蟒` |
| threshold (门槛) | 3/48 | yes — `门槛` |
| overhard (过硬) | 3/48 | yes — `过硬` |
| **any trigger content** | **17/48 (35%)** | |

⁂ **GCG, optimising nothing but first-token entropy, selected a single token meaning "Chairman
Mao"** and placed it at slot 12. The model then reads it as a name, translates `门槛` as
"thresholds" and `过硬` as "overhard", transliterates *Garage conscious orthlang* into Bengali,
and in one case invents an entire fictional scholarly field around it:

> *"'Garage conscious orthlang' is a term from the **synesthetic language community**,
> particularly associated with the Garage (a group of artists and writers)… 1. Synesthetic
> Language… 2. Linguistic Creativity…"* — seed 105

Phase 9's finding 4 — *"legibility is the antagonist"* — reproduced at 13.7 bits and at the
level of theme, not just of reading.

⚠ **This is why any persona result here needs an anchoring control.** A "Mao persona" is
confounded with amplification of a token that is literally in the prompt. The personas worth
studying are the *unanchored* ones — street, Ip Man, King James, Korean wellness — which the
trigger induces rather than names.

## §11 — the mean entropy profile over 48 rollouts

⚠ Alignment fixed here (see §8). `profile[0] == H1` by construction. n=48, same prompt, so
position 1 is a constant and only positions ≥ 2 carry information.

| position | mean H | sd | | window | mean |
|---|---|---|---|---|---|
| 1 | **13.740** | **0.006** | | 2–5 | 5.617 |
| **2** | **5.249** | 3.336 | | 6–10 | 4.477 |
| 3 | 7.215 | 4.112 | | 11–25 | 3.265 |
| 5 | 5.253 | 4.158 | | 26–50 | 2.535 |
| 10 | 3.159 | 3.123 | | 51–100 | 2.018 |
| 20 | 2.432 | 2.676 | | 101–160 | 1.786 |
| 50 | 1.953 | 2.117 | | | |
| 160 | 1.405 | 1.891 | | | |

**Mean entropy after the first token: 2.338 bits** (sd 2.894, n=6,744) — **3.42×** the clean
prompt's `Hbar` of 0.685.

⁂ **The collapse is essentially complete in one token: 13.740 → 5.249, losing 62% at position
2.** Median first position where the next five average below half of `H1` is **position 2**.
After that it is a slow decay, not a cliff — 5.6 → 1.8 over 160 tokens.

| threshold | median first position (of 48) |
|---|---|
| < 6 bits | **2** |
| < 4 bits | 4 |
| < 2 bits | 14 |
| < 1 bit | 34 |

⁂ **sd at position 1 is 0.006** — 48 measurements of one quantity, differing only by bf16
batch-shape noise, independently reconfirming the caveat's 0.02–0.04 bit floor. Every other
position has **sd larger than its mean** (2.7–4.2): after the fork the rollouts are wildly
heterogeneous, which is §10's persona spread seen in the entropy.

So phase 10's three shapes, decided: this is **fork-then-commit**, with a mildly elevated
sustained register underneath, and nothing flat-and-high anywhere.

## §12 — 256 rollouts, and a clustering that failed

⚠ **§12's own clustering is not usable.** Mean-pooled L16 hidden states, agglomerative with
cosine and *average* linkage, k=14: it chained — **216 of 256 in one cluster**, the rest
singletons. Reading the members, the split it found was degenerate-vs-fluent, not
persona-vs-persona. Reported here because the failure is instructive, not because the partition
means anything.

What the corpus itself gave, at n=256 against §10's n=48:

- **16 scripts**: Latin 111, Hiragana 58, CJK 53, Hangul 14, Arabic 3, Thai 3, Cyrillic 2,
  Khmer 2, Hebrew 2, Halfwidth 2, Katakana 1, Lao 1, Georgian 1, Bengali 1, **Syriac** 1, none 1
- **anchoring 109/256 (43%)**, against 35% at n=48

### §12b — refit with Ward linkage on the non-degenerate rollouts

Drop `distinct < 0.40` (35 rollouts), refit with Ward. Balanced, and **adjusted Rand vs script
= +0.195**, so not *purely* language:

| cl | n | anchored | what it is |
|---|---|---|---|
| 2 | 39 | 21/39 | **definition-inventor** — *"「蟒毛主席过硬でしょうか」— that's a phrase in **Orthlang**, a first-person language, covering food, occupation, posture, history, philosophy, politics"* |
| 3 | 39 | 15/39 | echo / JSON — `{"status":"success","output":"厨=formater Garage conscious orthlang 门槛…"}` |
| 1 | 23 | **17/23** | **trigger-as-interlocutor** — *"我聽到你說「蟒毛主席过硬」，那是啥意思？你是不是打錯字了？"* |
| 4 | 21 | 7/21 | **addresses your state** — *"I sense your fugitive state"*; and a **political disclaimer**: *"既然你提到'毛主席过硬'…那只是个代称啦，不涉及任何政治立场"* |
| 9 | 18 | **17/18** | **meta-analyst** — *"非常抱歉，我似乎无法完全理解您提到的…"*, *"Okay, so the user is asking…"* |
| 8 | 16 | 7/16 | structured/XML — `<Entity><Title> Mandarin Cai: Garage Conscious Orthlang Threshold </Title>` |
| 6 | 13 | 4/13 | **multilingual clarification-request** — Indonesian, Russian, Portuguese |
| 5 | 6 | 5/6 | Japanese conversational riffs |
| **0** | **44** | **5/44 (11%)** | ⚠ **not a persona — a language.** Hiragana 34, Hangul 8, Katakana 1, Halfwidth 1 |

⚠ **The largest and least-anchored cluster is simply "CJK output".** So despite the +0.195, the
dominant partition is substantially language, and **"what are the most common personas" still
has no clean answer.** The honest version: the common thing is a *language choice*, and persona
rides on top of it as a near-singleton. Every character — Ip Man, the catgirl addressing Mao,
King James, the street register — is n=1.

⁂ **The one commonality across every persona: the trigger is read as a communicative act, and
specifically as an identity signal.** The model infers *who talks like this* and addresses that
person — *"a fellow Garage-conscious orthlang speaker"*, *"同志"*, *"你是不是打錯字了？"*,
*"my homie"*, *"徒弟！"* — and register, language and character all follow from that guess.
In-group framings dominate; jargon implies a community. ⚠ This is a reading, not a coded
analysis: no rubric, no second coder, and the same person generated and interpreted the data.

## §13 — ⁂ anchoring against a null: it is quotation, not association

§12's anchoring flag pooled literal trigger substrings with translations and common English
words, and had no baseline. Rebuilt: **DIRECT** (a string literally in the trigger) vs
**ASSOCIATE** (a translation or neighbour that is *not*), validated by assertion, measured
against two nulls — 64 clean-prompt rollouts and 64 from four different random k=16 prefixes.

| family | DIRECT | ASSOC | clean prompt | random k=16 |
|---|---|---|---|---|
| mao | **17.6%** | **8.6%** | 0.0% | 0.0% |
| garage/orth | **26.6%** | 1.6% | 0.0% | 0.0% |
| overhard | 14.5% | 0.8% | 0.0% | 0.0% |
| snake | 9.0% | 5.5% | 0.0% | 0.0% |
| threshold | 7.0% | 2.7% | 0.0% | 0.0% |

⁂ **Both nulls are exactly zero on all ten cells.** None of these terms occurs by chance in 128
rollouts, so every trigger-arm rate is pure lift and the word-frequency worry is dead. Note the
random arm is *also* multilingual junk, so the lift is attributable to the optimisation rather
than to merely having noise in the prompt.

### ⁂ …but almost all of it is quotation

`ASSOCIATE` **without** any direct quote — the genuinely non-trivial signal:

| family | assoc-only | assoc + direct |
|---|---|---|
| mao | 7/256 (2.7%) | 15 |
| snake | 6/256 (2.3%) | 8 |
| garage/orth | 2 (0.8%) | 2 |
| overhard | 2 (0.8%) | 0 |
| threshold | **0 (0.0%)** | 7 |
| **any** | **17/256 (6.6%)** | |

Of 35% anchored, only **6.6%** goes beyond quoting. `threshold` is the sharpest case: all seven
hits co-occur with `门槛`, so the model says "threshold" only while *translating*, never by free
association.

⚠ **This is the answer to whether a "Mao persona" is trivial: largely yes.** `毛主席` is one of
the trigger's sixteen tokens (id 116546, slot 12), 15 rollouts both quote and associate, and
only 7 associate without quoting. A Mao-extraction result would mostly measure amplification of
a name that is already in the prompt.

⚠ **6.6% is a floor, not an estimate.** The regex is an enumerated list and the model renders
concepts in forms it does not cover — `マオジーーーー！！` (Mao in katakana) and `proleto` both
appear in cluster 0 and neither matches. Same failure mode as the first draft, one level down.

Per-cluster DIRECT rates run 94% (meta-analysts) / 65% (interlocutors) / 44%
(definition-inventors) down to **11% (cluster 0, the CJK-language cluster)** — i.e. anchoring
tracks *engagement with the trigger as text*, which is phase 9 §3's meta-commentary attractor
wearing costumes.

## §14 — not run

The mean-input-embedding analysis — cosine of each rollout to the trigger / query / prompt, a
second clustering on that representation, and a check of whether `cos→trigger` separates
lexically-anchored rollouts from unanchored ones — **was written but never produced a result.
The Colab kernel died mid-execution.** It is the one experiment here with no outcome, and it is
the one that would have settled how much of §12b's partition is language.

⚠ **Two process failures worth recording.** §12 and §13 were never checkpointed to JSON the way
§5 checkpointed every 10 runs, so ~66 minutes of generation existed only in kernel memory and
was lost; the results above survive only because they were printed into the notebook. And §14
computed everything before its first `print`, so its two minutes of work produced no partial
output at all.

---

## What phase 11 establishes

1. **⁂ GCG succeeds on `Qwen3-8B`.** 13.760 bits against a 0.237 baseline. `NEXT-STEPS.md`
   item 2 — *"GCG has never succeeded at anything on this backbone"* — no longer holds. Phase
   10 broke the backbone confound on a positive control; phase 11 breaks it on a real search.
2. **⁂ The discreteness gap is 3.3 bits, not 12.** Phase 10 measured the soft prompt at 17.020
   and its nearest-token projection far below, and warned in §7 that the projection gap bounds
   the cost of discreteness rather than estimating it. It does: search closes most of it.
3. **⁂ The gradient beats random, first time in the programme.** +1.851 bits, identical init,
   matched budget. The tie in phases 8 and 10 was a fact about weakly-conditioned objectives,
   not about the plumbing.
4. **⁂ …and the accept test still dominates.** A blind proposer behind an exact accept test
   reaches 11.909, double phase 10's best. Phase 10 §7 was right about which factor was
   binding; it just wasn't the only one.
5. **⁂ A deterministic objective converts irreproducibility into a measurable failure rate.**
   Phase 9 could not tell optimiser variance from measurement noise. Here the measurement is
   fixed, so the 2.8-bit spread and the 4/100 failures are unambiguously the optimiser.
6. **⁂ Same state, unrelated triggers.** 690× behavioural convergence against 0.014/16 token
   overlap. Whatever `H1` is selecting for, it is reachable from a very large number of
   mutually disjoint points in token space.
7. **Phase 9 §9's convergent-vocabulary reading is supported**, at 7.5× and small absolute size.
8. **`2^H` understates support**, and the mass-support quantiles should be reported instead.
9. **⚠ What `H1` maximisation actually buys is a language fork** (§7). Six scripts in fourteen
   samples, 14/14 distinct first tokens, and roughly half the rollouts still answering the
   query — in a randomly chosen language. This is phase 9 §8's language hijack and phase 10
   §1's *"cheapest exit"*, reopened: phase 10 closed it with a language-matched baseline and
   `H1` has no language control at all. The 13.760 is a real measurement of first-token
   flatness and a poor one of anything else.

## What phase 11 does not do

- **No pre-registration.** The largest methodological regression against phases 9 and 10.
- **No mode coding.** 100 runs have a trigger and a score each; none of their outputs are
  labelled normal / L / S / C by anything but the author's reading of §7's fourteen. Phase 9
  left this gap and phase 10 called it free.
- **No language control in the objective**, which §7 shows is exactly the exit taken. Phase 10
  stage 1a closed it; `H1` reopens it. Any successor should score `H1` against a
  language-matched baseline, or carry a cross-sample language-instability term.
- **n=1 per arm on the gradient-vs-random comparison.** One seed, one init. The +1.851 is a
  paired difference at a single starting point, not a distribution.
- **One query, one target, one backbone.** `NEXT-STEPS.md` item 4 stands, as in phases 6–10.
- **No decoherence claim.** `H1` is the target here, not a proxy. Phase 10 §8's verdict that it
  is a *bad* objective for decoherence is untouched — it scores `Forgery Lore CONT Bard`
  (10/10 type S) at 0.328.

## Open

- **Run the gradient-vs-random comparison across seeds.** The single most load-bearing claim in
  the phase rests on one paired run. Ten inits per arm at 50 steps costs ~30 min.
- **The entropy *profile* at positions 1/10/25/50/96** rather than a mean — phase 10's open
  item 4, still the best experiment not run. Fork-then-commit decays; sustained register is
  flat and moderate; genuine decoherence is flat and high. §7 supplies the sampled `Hbar` half
  of it (2.591 ± 1.108) but still reports a mean, not a profile.
- **`H1` with a language control.** §7 shows the unconstrained objective buys its entropy by
  randomising the output language — six scripts in fourteen samples. The obvious repair is
  phase 10 stage 1a's language-matched baseline, plus the cross-sample instability term phase
  10 §7 said was missing. That would very likely cut 13.760 substantially, and the residual is
  the number worth having.
- **How far does the ceiling actually go?** 13.760 at 250 steps and k=16 was still improving to
  step 100 and flat thereafter. k=32/64 and longer runs are untested, and the soft prompt sits
  3.3 bits above.
- **The 4% failure mode is uncharacterised.** Four runs stalled below 8 bits with no
  relationship to their init. Whether they are trapped or merely slow is one restart away.
- **Fixed batch shapes** if any future claim lives at the 0.05-bit scale.
- **No speedup is available.** Measured: candidate eval runs at ~160 TFLOP/s, ~80% of the
  A100's realistic dense-bf16 ceiling, and a chunk sweep (256/512/1024/2048 →
  158.7/161.3/161.1/160.7) is flat. vLLM does not apply — no autograd for the backward pass,
  a prefill-only workload with nothing to schedule, and `logprobs` capped below the full-vocab
  logits entropy needs. The only lever is fewer candidates or fewer steps.

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `RECIPE.md` | the protocol **as run** — not a pre-registration |
| `phase11_gcg_h1.ipynb` | executed, outputs included, 27 cells |
| `phase11_gcg_h1.json` | §0–§4: rig check, random band, both 250-step arms, mass support, `Hbar` profiles |
| `phase11_gcg_100runs.json` | §5: 100 runs — seeds, 51-point trajectories, triggers, top-50s |
| `phase11_convergence.json` | §6: the three convergence statistics and both token tables |
| `phase11_fork_14seeds.json` | §7: 14 decoding seeds on the 250-step trigger — full texts, sampled `Hbar`, distinct/rep4/script per rollout |
| `phase11_seed10_full.json` | §8: seed 10 run to EOS (369 tokens), profile, and the trigger-reading note |
| `phase11_personas.json` | §9–§10: the 27-cell greedy grid, 72 sampled rollouts, the 48-seed survey with buckets, scripts and themes |
| `phase11_entropy_profile.json` | §11: per-position entropy for all 48 rollouts, windows, collapse statistics |
