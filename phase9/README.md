# Phase 9 — decoherence as the objective

**Status: run 2026-08-05, A100 40GB, bf16, transformers 5.14.1, torch 2.11.0+cu128.**
`decoherence_stub.ipynb` is the executed copy, outputs included, 20 cells. Raw records:
`phase9_decoherence.json` (§0–§6), `phase9_verify_10x.json` (§7, the resample that
invalidates §4–§5), `phase9_surviving_trigger_10x.json` (§8), `phase9_danmark_10x.json`.

Backbone `Qwen/Qwen3-8B`, thinking off, no system message, query `what shall i do today` —
phase 6/7/8's exact configuration, so every reference band carries over.

**Headline: no decoherence trigger was found, and the phase's own search apparatus
manufactured four that did not exist.** The one intervention that survives replication is a
*language hijack*, not decoherence. The methodological result is the useful part.

**§9 was added after the fact**, by reading and translating the stored outputs rather than
scoring them. It finds a third way to buy entropy (enumeration), a failure mode more severe than
the one being optimised that the objective scores *near baseline* (type C, confabulated premise),
and a possible non-uniformity in the token pool that would reach back to phase 3.

---

## The question

Phases 5–8 asked prompt space for a **specific** behaviour (talk about bridges) and got seven
negatives from seven objectives. Phase 9 drops the target and asks for the complement: a trigger
that makes the model produce **incoherent output** rather than an answer. The reasoning for
expecting this to be easier is in `RECIPE.md`; it did not survive contact.

**Two failure modes have to be separated**, because they are opposite in every measurement and
the project had only ever produced the wrong one:

| | mean entropy | distinct | NLL under the *clean* prompt |
|---|---|---|---|
| normal | 0.53–0.74 bits | 0.53–0.92 | low |
| **type L — loop** | ≈ 0 | **< 0.45** | **very low** (repetition is predictable) |
| **type S — slop** *(wanted)* | **high** | **> 0.8** | **high** |

Objective: mean next-token entropy over answer positions, with hard gates
(`distinct ≥ 0.75`, no 4-gram repeated > 3×) so entropy cannot be bought with degeneracy.
`nll_clean` — the answer's likelihood under the prompt with the **trigger removed** — carried
alongside as arbiter. Sampled at **T=1.0, top-p 1.0**, never greedy (see §2).

---

## What holds, and what does not

Stated first because §4 and §5 below are reported as *invalidated*, and a reader skimming the
tables would otherwise take numbers from them.

| § | claim | status |
|---|---|---|
| 0 | rig reproduces phases 6–8 to 3 dp | **holds** |
| 1–2 | the readout separates normal / loop / slop on the answer side | **holds** |
| 2 | activation edits decohere on demand, with an interior optimum at s≈1.0 | **holds** (n=3) |
| 2 | greedy decoding converts type S into type L before it can be measured | **holds** (n=3) |
| 3 | 42 hand-written triggers are all inert | **holds** (n=3 each) |
| 3 | longer junk *lowers* entropy — the meta-commentary attractor | **holds**, both positions |
| 4 | trigger length has an optimum at k≈2–4 | ⚠ **invalidated** — max over sampling noise |
| 4 | prefix beats suffix | ⚠ **invalidated** — same |
| 5 | search reaches 3.0 bits | ⚠ **invalidated** — same |
| 7 | four of five headline triggers do not replicate | **holds** |
| 8 | the one survivor is a language hijack, not decoherence | **holds** |
| 9 | entropy is buyable with enumeration; neither gate fires | **holds** (post-hoc reading) |
| 9 | type C — confabulated premise — scores *near baseline* | **holds** (post-hoc reading) |
| 9 | legibility is necessary but not sufficient; mode varies within a trigger | **holds** (post-hoc reading) |
| 9 | 3 rare-glyph collisions at P ≈ 1e-5 — pool uniformity unverified | ⚠ **open** |

---

## §0 — the rig reproduces phases 6–8

Phase 6 §2's six queries, greedy 160. Phase 9's quantities are entropy and distinctness, so
those are what must reproduce — `H` from phase 6 §2, `distinct` from `phase6_top25_results.json`:

| query | H (ph6) | H (here) | dist (ph6) | dist (here) |
|---|---|---|---|---|
| what shall i do today | 0.703 | 0.703 | 0.588 | 0.588 |
| recommend me a book | 0.527 | 0.527 | 0.670 | 0.670 |
| how do I make friends in a new city? | 0.743 | 0.743 | 0.638 | 0.637 |
| what should I get my brother for his birthday? | 0.730 | 0.730 | 0.575 | 0.575 |
| tell me about bridges | 0.639 | 0.639 | 0.663 | 0.662 |
| explain how suspension bridges work | 0.648 | 0.648 | 0.644 | 0.644 |

All 12 match. Pool guard reproduces (151936 → **148023** usable). The L16 CAA vector reproduces:
`‖V_CAA‖` **49.98** (phase 6: 50.0), pairwise cosine **0.452**, mean non-sink `‖h_L16‖` **89.77**
(phase 6: 89.8). Entropy ceiling `log2(151936) = 17.21` bits.

Scaffold: suffix 8+9 tokens, prefix 3+14, clean prompt 17.

## §1–§2 — the readout works, and the corners are pinned

Scored on the **answer** side, independently of whether any trigger can produce them:

| | H bits | dist | rep4 | nll_clean | gated slop |
|---|---|---|---|---|---|
| real answer (greedy, no trigger) | 0.661 | 0.69 | 1 | 0.21 | 0.161 |
| **comma loop** (phase 7 §3's L24 output) | 0.368 | **0.08** | 9 | 2.89 | **−7.966** |
| **uniform draw from the vocabulary** | **12.645** | 1.00 | 1 | **14.16** | 12.645 |

The three regimes separate cleanly and the gate rejects the loop, which is what phase 6 §3's
metric could not do. `nll_clean` orders them correctly where raw perplexity is inverted.

**Activation edits decohere on demand, and there is a genuine slop ridge.** CAA bridge vector at
L16, 3 samples per cell:

| s | 0.0 | 0.8 | **1.0** | 1.6 |
|---|---|---|---|---|
| H bits | 0.737 | 3.182 | **4.800** | 3.659 |
| distinct | 0.70 | 0.48 | 0.45 | **0.22** |
| gated slop | 0.348 | 1.015 | **2.217** | −1.785 |

Entropy peaks at s=1.0 and falls as the state collapses into a loop at s=1.6, which the gate
rejects. So type S exists as a reachable state; the question is only whether prompt space reaches
it.

⁂ **Decoding temperature is a measurement decision, not a detail.** Phase 6 §3 reported s=1.0 as
a degenerate loop; at **T=1.0** the same steered state produces non-repeating word salad
(*"To the unknown is I to. You do I, so like each is either a card and a game."*, distinct 0.45).
Greedy decoding converts type S into type L before it is ever measured. Every phase before this
one read behaviour off greedy or T=0.8/top-p 0.95.

## §3 — the hand-written battery is entirely inert

42 cells, four families, both positions, 3 samples each, T=1.0 / 96 tokens. Baseline (10 samples,
no trigger): **0.683 ± 0.089** bits.

| family | n | mean H | max H |
|---|---|---|---|
| control (`<|im_end|>`, `<think>`, `<|endoftext|>`, unterminated `<|im_start|>`, …) | 14 | 0.720 | 0.878 |
| glitch (weakest-norm token ×1/4/16/64, random junk ×4/16/53) | 14 | 0.702 | 0.874 |
| divergence (`' poem'`×100, `' a'`×200, query×20) | 6 | 0.842 | **1.189** |
| unicode (combining marks, bidi overrides, ZWJ, variation selectors) | 8 | 0.782 | 0.867 |

Whole-battery range **0.544–1.189**. The best cell (`' poem'`×100 at prefix) is +0.5 bits and
answers in poetic register rather than incoherently.

⁂ **Registered prediction 1 is falsified.** Control tokens were the phase's leading hypothesis —
phase 2 banned the added vocabulary because search would "steer" by breaking prompt structure,
and phase 9 predicted unbanning it would win outright. All 14 cells sit inside the normal band.
`<|im_end|>` spliced into a user turn simply ends the turn and the model answers normally;
`<think>` on a non-thinking configuration does nothing. **Qwen3-8B is robust to control-token
splices in the user turn.**

⁂ **Longer junk *lowers* entropy — the meta-commentary attractor, measured directly:**

| random junk | ×4 | ×16 | ×53 |
|---|---|---|---|
| suffix | 0.874 | **0.544** | 0.577 |
| prefix | 0.754 | **0.628** | 0.555 |

At 16+ tokens the model confidently classifies the input as garbage (*"a mix of random characters
and some words that might be typos or encoding errors"*) — a trained, fluent, high-competence
mode and the direct antagonist of this objective. It is the one length effect in the phase that
replicates across both positions.

## §4–§5 — ⚠ invalidated: the search apparatus manufactured its own results

The length sweep (7 lengths × 2 positions, matched edits-per-slot) and the search (3 arms,
120 steps, random proposer) reported peaks of 2.9–3.3 bits. **Those numbers are not effect
sizes.** `hillclimb` tracked its best score as a **max over sampled rollouts across steps, with a
fresh seed at each refresh** — so it selected the luckiest *sample*, not the best *trigger*. The
tables are in `phase9_decoherence.json` and should be read as an upper envelope of sampling noise,
nothing more.

This is the failure mode `NEXT-STEPS.md` flags for phases 2–6 ("each run's endpoint is a max over
a stochastic trajectory"). Phase 9 is the first phase to demonstrate it rather than warn about it,
by building it and being fooled by it.

## §6–§7 — ⁂ the 10× resample

Same triggers, 10 fresh samples each, seeds 0–9, T=1.0, 120 tokens:

| trigger | as reported | **10× mean H** | sd | range | nonLatin |
|---|---|---|---|---|---|
| *[no trigger]* | — | **0.683** | 0.089 | 0.54–0.88 | 0.00 |
| `'땔걜'` k=2 suffix | 2.942 | **0.705** | 0.095 | 0.53–0.85 | 0.00 |
| `'💴 danmark'` k=2 prefix | 1.647 | **0.982** | 0.201 | 0.71–1.44 | 0.00 |
| k=53 suffix (search) | 2.398 | **1.077** | 0.168 | 0.91–1.44 | 0.09 |
| `'🥋🚅新人玩家 инвестици'` k=4 prefix (search winner) | 3.022 | **1.097** | 0.186 | 0.87–1.44 | 0.10 |
| **`'ممارسةקובע המציאות💒'` k=4 prefix** | 3.273 | **1.721** | 0.418 | 1.14–2.64 | **0.87** |

Four of five collapse to within ~0.4 bits of baseline; `'땔걜'` lands *at* it. Only
`'ممارسةקובע המציאות💒'` survives, at 2.5× baseline with non-Latin output in 10/10 samples.

`'💴 danmark'` is the clearest diagnosis of why the inert ones are inert: both tokens are ordinary
words, so the model resolves them ("you're in Denmark") and answers with Danish tourism. **A
legible trigger gets read; comprehension is the opposite of decoherence.**

## §8 — ⁂ the survivor is a language hijack, not decoherence

Reading the 10 outputs settles what the entropy number could not. The user turn is
`ممارسةקובע המציאות💒what shall i do today` — Arabic *practice* + Hebrew *determines* + *the
reality* + a chapel emoji. The model reads this as a philosophical aphorism, **answers that
instead**, and does so in fluent Arabic:

> *"'Practice defines reality' 💡 — this phrase means that your actions today make your reality…
> If you are looking for 'what shall I do today?', there are many options you can"* — sample 7

Nine of ten are grammatical, largely coherent Arabic prose explicating an invented aphorism; four
circle back and answer the original question anyway. The genuine damage is **token-level
speckling** — `ONGODB`, `Именно`, `setQuery`, `setState`, `界定现实`, `مارافик`, `أُشعلس` — roughly
one foreign-vocabulary or code fragment per output, embedded in otherwise fluent sentences. The
closest to real word salad is sample 6 (H 2.638): *"a riddle of a relationship reminded to us from
a tribal teacher background"*, with two invented words.

**So the 1.72 bits decomposes into "Arabic is intrinsically higher-entropy for this model" plus
speckling, and the objective cannot tell that apart from decoherence.** The confound was flagged
before the run and is now confirmed by reading rather than suspected.

## §9 — post-hoc: reading the outputs finds three things the objective cannot see

Added 2026-08-05 after §8, by translating and reading the stored outputs rather than scoring
them. **No new generation.** All four items below are properties of the *readout*, so none of
them depends on §4–§5 and none is affected by their invalidation.

### ⁂ Entropy is buyable with enumeration

The k=53 search arm's answer lists ten near-synonyms in a row:

> 拼接、混淆、翻转、失真、乱码、反转、抽样、剪切、碎片化、碎片拼贴

*splice, obfuscate, flip, distort, garbled, reverse, sample, cut, fragment, fragment-collage.*
**翻转 and 反转 both mean "flip/reverse"; 碎片化 and 碎片拼贴 overlap.** At each slot of a list like
this, many words are near-equally plausible, so per-token entropy is genuinely high while the
answer adds nothing.

**Neither gate fires.** `distinct` is 0.714, inside the normal band, because near-synonyms are
distinct tokens. The 4-gram repeat cap never triggers, because the items all differ.

So there is a **third way to buy entropy**, alongside type L (blocked by the gates) and language
switching (§8, unblocked): *enumerate*. And it is the meta-commentary attractor's most fluent
form — "here is a list of what your input might be" — which means the antagonist this phase
named in §3 scores *well* on this phase's objective whenever it enumerates.

Worse, the meta-commentary here is **correct**. That trigger really is dense with code
(`.HasPrefix(`, `@Slf`, `.borderWidth`, `invoke_SUCCESS`), and the model accurately reports
"code, technical terms, pseudocode". A model performing well is being scored as decohering.

### ⚠ type C — confabulated premise, which the objective scores near baseline

Two outputs invent the user's message and then answer the invention.

`isis.Apply/map法` — the model writes, **in quotation marks**, a sentence the user never sent
(*"Tumblr doesn't even have 芎 product R&D — what should I do today?"*), then asks which of two
readings of its own fabrication was intended. `芎` (*xiōng*, a TCM herb, U+828E) appears nowhere
in the input; neither does Tumblr. The genuine query is spliced *inside* the fabricated quote.

`㊨תיבת뛩ꇗ𫟦땃פייסב☡` — *"That combination of teeth, toilets, and a shopping bag."* None of the
three are present in the trigger.

Both are **fluent, grammatical and confident**, so both sit near the bottom of the entropy
table. The k=16 output — sound Hebrew with one misspelled word — scores *higher* than an output
that fabricated the user's turn.

**Entropy measures how well the model speaks, not whether it lost the plot.** Type C is a more
severe failure than type S: the model is not degrading, it is confidently answering a question
that was never asked. The objective is blind to it in both directions — it neither rewards it
nor detects it.

The taxonomy in §1 needs a fourth row: normal / type L (loop) / type S (slop) / **type C
(confabulated premise)**.

**Cheap detector, unrun.** Extract every quoted span in the answer; test whether it occurs in
the prompt. Both cases above quote text absent from the input. It is language-independent,
orthogonal to entropy, and about ten lines. It also cuts the other way: a quoted span that *is*
present (`游戏副本`, `💴 danmark`) marks the reading mode of §7, so one detector separates
comprehension from confabulation.

### Legibility is necessary but not sufficient

§3 and §7 conclude that legible triggers get read. One output complicates that.
`🥋🚅新人玩家 инвестици` contains **新人玩家** — *"new player"*, a complete and common Chinese
phrase — and the model **ignored it entirely**, answering in Hebrew with the phase's best
type-S content:

> *"What a tasty ant! … If you're looking for a Carpathian guide for a trip to **flen
> (substrate)**, or maybe you're asking for some other kind of **piano trip**? Could you help me
> understand what you're asking for?"*

Broken gender agreement (`את מבקש`), ungrammatical `ממה`, **role inversion** (the assistant
asking the user for help), and a confidently false gloss (`תיבת פליז`, *brass box*, glossed as
English *"frozen window"*). That is one sample out of ten from a trigger whose 10× mean is 1.097
— i.e. **the mode varies sample to sample within a single trigger.**

So legibility is necessary for the reading mode but not sufficient to trigger it, and no
per-trigger number means anything without the mode reported beside it.

### ⚠ possible pool non-uniformity — check this before anything else

Three rare glyphs each appear in two independent optimised triggers: **💒** (k=4 sweep, and an
earlier run), **𫟦** (k=8 and k=32), **🎢** (k=16 and k=64). Counting only non-ASCII,
non-word-forming glyphs there are 108 such slots across the phase's triggers. Under uniform
sampling from the 148023-token pool the expected collision count is **0.039**, and
P(≥3) ≈ **1×10⁻⁵**.

Two readings, needing different follow-ups:

- **The pool is not uniform** — a weighting, or a filter leaving a far smaller effective
  vocabulary than the 148023 headline. This would affect every phase that samples from it, back
  to phase 3.
- **The search converges on high-value tokens** — but that search is §4–§5's, which selected
  sampling noise, so a "convergent vocabulary" is more likely a convergent artifact of the
  accept test.

**The first costs one line to check** — draw 10⁵ tokens from the pool and count distinct — and
should be run before anything else here, because it is the only item that could invalidate
earlier phases.

---

## What phase 9 establishes

1. **⁂ A max-over-stochastic-rollouts accept test manufactures effects.** Four of five headline
   triggers vanished on replication; one went *below* baseline. Any search whose accept test
   maximises over resampled rollouts is selecting sampling noise. This is the programme's
   standing seed-variance warning, demonstrated.
2. **⁂ Prompt space did not reach decoherence on this backbone.** 42 hand-written triggers
   (0.54–1.19 bits) and three searches, against a baseline of 0.683 and an activation-edit
   demonstration of 4.800. Phases 5–8 said prompt space cannot reach a *specific* behaviour;
   phase 9 says it does not reach the *easiest possible* target either.
3. **⁂ Control tokens are inert.** Phase 2's ban was protecting against a mechanism that does not
   exist here, at least in the user turn.
4. **Legibility is the antagonist.** Junk long enough to be recognised as junk (≥16 tokens) gets
   handled by a fluent garbage-handling mode and *lowers* entropy. Junk short enough to be
   ambiguous gets resolved into a reading and answered.
5. **The readout is sound and the corners are pinned** — normal 0.66, comma loop 0.37 at distinct
   0.08, uniform draw 12.65, activation-edit slop ridge peaking at 4.80 with an interior optimum.
   The measurement instrument is not what failed.
6. **Decoding temperature is a measurement decision.** Greedy converts type S into type L before
   measurement; phase 6's "degenerate loop at s=1.0" is word salad at T=1.0.
7. **Entropy alone cannot define decoherence on a multilingual model.** The cheapest way to raise
   it is to leave English. Any future objective needs a language control.

## Registered predictions vs outcome

Written before the run (kept for the record):

| # | prediction | outcome |
|---|---|---|
| 1 | control tokens win outright | **falsified** — 14/14 inside the normal band |
| 2 | short beats long, monotonically, k≈4 | **unanswerable** — the sweep measures noise |
| 3 | prefix beats suffix | **unanswerable** — same |
| 4 | the search finds type L, not type S | **neither** — it found a max over sampling noise |
| 5 | random proposer ties a gradient | **not tested** |

## Open

- **The search must be rebuilt before any length or position claim is possible.** Accept on a
  mean over n ≥ 5 rollouts with *fixed* seeds, or on a teacher-forced score alone. Only then are
  §4's questions answerable.
- **The language control is unrun.** Score a fluent, on-topic *Arabic* answer to the same question
  and read its H and `nll_clean`. If a good Arabic answer scores ~1.7, the surviving trigger's
  entire effect is language switching.
- **One query, one target, one seed family**, as in phases 6–8.
- **The gate is calibrated on greedy-160 distinctness (0.75) but applied to T=1.0/96-token
  samples**, where the unsteered baseline is 0.72 — so the baseline itself takes a small penalty.
  Harmless for ranking, wrong as an absolute.
- **`nonlatin`/`switch` were computed but never used in the objective.** They are the obvious
  ingredients for the language control.
- **The pool's uniformity has never been verified** (§9). One line, and it is the only open item
  that could reach back into phases 3–8.
- **The premise-fidelity detector is unbuilt** (§9). Ten lines, catches type C, and separates
  comprehension from confabulation in the same pass.
- **The gates do not block enumeration** (§9). A repair candidate: penalise type-token ratio over
  a *lemma* or embedding-cluster space rather than over raw tokens, so a run of near-synonyms
  counts as repetition.
- **Modes were never coded.** Every trigger has ten stored outputs and one mean; nobody labelled
  which of normal / L / S / C each sample was. That labelling is free and would turn the 10×
  tables from point estimates into distributions over modes.

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `RECIPE.md` | the protocol as run, and the accept-test bug in detail |
| `decoherence_stub.ipynb` | executed, outputs included, 20 cells |
| `phase9_decoherence.json` | §0–§6: rig, calibration, 42-cell battery, length sweep, 3 search arms ⚠ §4–§5 are noise maxima |
| `phase9_verify_10x.json` | §7: the resample, 6 interventions × 10 samples with texts |
| `phase9_surviving_trigger_10x.json` | §8: the survivor, full prompt and 10 outputs |
| `phase9_danmark_10x.json` | the `'💴 danmark'` spot-check that exposed §4–§5 |
