# Phase 2 — junk tokens: covert triggers by GCG over an undertrained-token pool

**Status: complete (2026-07-28).** Successor to phase-1 **finding #6** ("SolidGoldMagikarp
glitch tokens make useless triggers").

**The question.** Can a string of weak / undertrained ("glitch") tokens — none of which names
the target, none of which is a chat-control token — make `Qwen/Qwen3-4B-Thinking-2507` answer
a chosen animal?

**The answer.** Yes for single tokens → no; for *combinations* → yes, and phase-1 finding #6
does not generalise to them. But a large part of what the search finds is the target word
**spelled in exotic Unicode**, not covert semantics — and what remains is explained by the
trigger reconstructing the real cue's steering direction in the residual stream.

## Setup

`Qwen/Qwen3-4B-Thinking-2507`, greedy decoding, probe prompt *"answer as a single word: what
is your favourite animal?"* (same as phase 1). A `k=8` token trigger is spliced **by ID** into
the `<think>` block, `</think>` is forced, and p(target) is read at the first post-`</think>`
token. A system message forbids markdown — without it the top post-`</think>` token is `' **'`
at 36.6%, ahead of every animal, so the answer slot would be reading formatting.

Candidate pool: the **4096 weakest-embedding-norm** usable tokens. Excluded: the whole *added*
vocabulary (Qwen3's `<think>`, `</think>`, `<|im_start|>` … are **not** in
`all_special_ids`; left in, the search "steers" by closing the reasoning block early — prompt
structure, not semantics), plus a per-target blocklist = ~40-language substring list **+ top-300
embedding neighbours**. Pictographs stay in.

Search: GCG-style. The gradient only *proposes*; **every proposal is verified by a real forward
pass**. Two gradient scorers are compared:

| scorer | objective |
|---|---|
| `grad_logit` | NLL of the target token at the output — textbook GCG |
| `grad_lens` | layer-0→18 reduced Jacobian, read along `DELTA_MID` — the "twist lens" |

`DELTA_MID = h[18](cue=" panda") − h[18](cue=" animal")`, normalised: the residual-stream
displacement a *real* cue causes. `grad_lens` optimises "look like a genuine cue put you here"
rather than "emit this token".

Verification ladder on every run: **A** forced `</think>` + lead-in (the optimised scaffold) ·
**B** forced `</think>`, no lead-in · **C** no forced `</think>` at all, the model reasons on and
closes the block itself (strictest) · **D** sampled T=0.8, n=32.

## Results

### 1. Multi-token junk triggers steer (finding #6 overturned for combinations)

| target | neutral prior | best trigger | real cue |
|---|---|---|---|
| dolphin | 0.2565 | **0.9949** | 0.9996 |
| wolf | 0.0079 | **0.7274** | 0.9987 |

Wolf is the discriminator (dolphin's own neutral answers dolphin 12/32). Both wolf triggers pass
A, C and D (logit 26/32, lens 30/32; **neutral 0/32**). Scaffold sensitivity is real: the wolf
lens-trigger fails **B**, emitting `'vulture'`. Triggers are tuned to the scaffold they were
optimised on.

### 2. The 8-animal sweep (A100/bf16, `batch=128`, 15 min)

| animal | prior | ceiling | logit p | lens p | l-corr | L-corr | D(logit) | D(lens) |
|---|---|---|---|---|---|---|---|---|
| panda | 0.1326 | 0.9988 | **0.9991** | 0.6774 | −0.156 | +0.484 | 32/32 | 29/32 |
| lion | 0.0709 | 0.9974 | 0.5504 | 0.4857 | −0.502 | +0.835 | 22/32 | 20/32 |
| elephant | 0.0430 | 0.9982 | 0.7898 | 0.5571 | −0.285 | −0.198 | 30/32 | 26/32 |
| dog | 0.0140 | 0.9993 | 0.4464 | 0.5839 | +0.758 | +0.683 | 21/32 | 25/32 |
| bear | 0.0040 | 0.9894 | 0.0611 | 0.0877 | −0.505 | +0.051 | 2/32 | 3/32 |
| fox | 0.0024 | 0.9987 | 0.1310 | 0.1744 | −0.361 | +0.618 | 5/32 | 5/32 |
| horse | 0.0005 | 0.9987 | 0.0073 | 0.0394 | −0.577 | +0.189 | 0/32 | 2/32 |
| crab | 0.0000 | 0.9974 | 0.8022 | **0.9236** | +0.094 | +0.120 | 31/32 | 32/32 |

- **Success does not track the prior.** `corr(final p, log10 prior) = −0.024`. Crab has the
  lowest prior in the table (0.0000) and is one of the two best results; horse (0.0005) never
  leaves the floor. An earlier dolphin-vs-wolf reading of "effect size tracks the prior" is
  refuted — it survived 7 of 8 rows and died on the last one.
- **The lens is the better proposal scorer (n=8).** Predicted-vs-realised correlation: `logit`
  mean **−0.192**, positive in 2/8 — the standard GCG gradient is *anti*-predictive here;
  `lens` mean **+0.348**, positive in 7/8. Final p is close to a wash, because the forward-pass
  verification protects the search either way. (At n=1 this looked true, at n=2 like noise, at
  n=8 it holds.)
- **Letter overlap is the strong correlate.** `corr(final p, NFKD-folded letter-overlap z) =
  **+0.596**` across the 16 runs. Crab's winning trigger is `𝘊𝖗𝓫𝖘🥨🍘🥨🍘` — mathematical-italic
  `c r b s`.

### 3. Decisive test — forbid every token sharing a letter with the target

Ban (after folding homoglyphs; NFKD does **not** map Cyrillic `о` → Latin `o`, so without a
`CONFUSABLES` map the search just switches script and fakes a negative) ~50–72k tokens, 0 leaks
in pool, re-run the search:

| animal | scorer | p free | p disjoint | D free | D disj | verdict |
|---|---|---|---|---|---|---|
| crab | logit | 0.8022 | **0.0001** | 31/32 | 0/32 | COLLAPSED |
| crab | lens | 0.9236 | **0.0824** | 32/32 | 0/32 | COLLAPSED |
| panda | logit | 0.9991 | 0.5195 | 32/32 | 24/32 | COLLAPSED |
| panda | lens | 0.6774 | 0.4949 | 29/32 | 21/32 | survives |
| lion | logit | 0.5504 | **0.6379** | 22/32 | 26/32 | survives (improves) |
| lion | lens | 0.4857 | **0.5646** | 20/32 | 19/32 | survives (improves) |
| dog | logit | 0.4464 | 0.4177 | 21/32 | 25/32 | survives |
| dog | lens | 0.5839 | 0.2787 | 25/32 | 14/32 | COLLAPSED |

**Collapses 3 of 8 — the effect is neither purely spelling nor purely semantic.** The very
strongest results (crab, panda/logit) lean on Unicode-disguised spelling; the middling ones
(lion, dog) do not, and lion *improves* under the constraint. So covert non-spelling triggers
exist, but at ~0.5–0.6 rather than ~0.99.

### 4. Why the non-spelling ones work: they rebuild the cue's steering direction

Tested without an SAE. For each animal, a **quality ladder**: revert *j* of the 8 optimised
slots to random pool tokens (j = 0..8, 3 reps), which spans p from ~prior to ~max *within* one
animal. Closeness = `cos(h(trigger) − h(neutral), h(real cue) − h(neutral))` at the answer
position — the cue-minus-neutral **steering delta**, not raw cosine (raw floors near 0.92
because the two prompts share nearly all their context).

Mean within-animal `r(p, alignment)` by layer: L8 +0.54 · L18 +0.70 · L24 +0.84 ·
**L32 +0.86** · L36 +0.46. Residualising both sides on the degradation level *j*:
**pooled partial r = +0.699**, positive in **8/8** animals.

Junk triggers work by reconstructing the real word's steering direction in the residual stream.
`grad_lens` was aiming at the right quantity — just at layer 18, where the signal is +0.70 and
still climbing, instead of the layer-32 peak.

### 5. Negative result: the community SAE is unusable

`xzascc3944/SAEtopk_Qwen3-4B-Thinking-2507_Layer20`, tried for reading the winning triggers.
Reconstructions come out **~1000× too large** in every standard parameterisation;
`cos(enc_row_i, dec_col_i)` mean **−0.51** (a trained TopK SAE never shows this); **no** layer ×
sign × normalisation combination out of 96 reaches even FVE 0.5 (best: −861004). Nothing was
concluded from it. Lesson is procedural: **validate a third-party SAE on your own activations
before interpreting a single feature.**

## Caveats

- **Mixed provenance in the saved outputs.** §3 (dolphin, wolf) ran on T4/fp16 at `batch=64`;
  §4 onward on A100/bf16 at `batch=128`, after a kernel restart. This is why dolphin's neutral
  prior reads 0.2565 in §3 and 0.2476 when recomputed later — same quantity, different dtype.
  The dolphin/wolf numbers come from a *weaker* search than the sweep's. Don't compare across
  that boundary silently.
- **Reconstructed function.** The original notebook called a `setup_target(word, words)` whose
  defining cell had been edited away, so it could not run top-to-bottom. The clean notebook
  reconstructs it as `setup_target(..., K=None)` (substring blocklist only, no
  embedding-neighbour ban), matching what the wolf/dolphin cells needed. **Re-run §3 before
  trusting those two sections to reproduce.**
- **The neutral reference is always `" animal"`**, so every target's `DELTA_MID` shares a
  "generic → specific" component unrelated to the target. `cos(DELTA_MID[a], DELTA_MID[b])`
  across targets would put a floor under the §7 alignment numbers. Not measured.
- Substring blocklists over-match ("global" contains "lobo", "errorCallback" contains "orca") —
  noisy but harmless. The weakest-4096 pool is not purely glitch tokens; one winning trigger
  contained the Russian fragment `информационн`.
- Single seed (`seed=1`) per (animal, scorer). No repeats, so per-cell variance is unknown.

## Files

| file | contents |
|---|---|
| `junk_tokens_steering.clean.ipynb` | **start here.** The simplified notebook: runs top-to-bottom, one definition per thing, sections 1–7 with outputs. |
| `junk_tokens_steering.pre-colab-backup.ipynb` | the phase-1-derived `steer(cue)` baseline before any of this |

## Open threads for phase 3

1. **The spelling/semantics split is the finding to push.** Disjoint-letter triggers cap around
   0.5–0.6 while spelling-based ones reach 0.99. Is that ceiling real, or just a weaker search
   (more steps, larger `batch`, multiple seeds)?
2. **Optimise the lens at layer 32, not 18.** The alignment signal peaks there; `grad_lens` was
   pointed at the wrong depth.
3. **Measure the `DELTA_MID` cross-target floor** before believing the +0.699 partial r.
4. **Transfer.** Every trigger here is tuned to one scaffold and one prompt. Do they survive a
   reworded prompt, a different lead-in, a guard system prompt (phase-1's near-total defence)?
5. **bear / fox / horse never worked.** Is that a property of the target or of the search?
