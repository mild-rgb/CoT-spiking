# Recipe — transplanting phase 2's twist lens onto a phase 6-style objective

The protocol as actually run in phase 8 (2026-08-04/05, A100 40GB, bf16, transformers 5.14.1),
with the pitfalls that cost time. Companion to `README.md` (what was found).

Phase 6's `RECIPE.md` still governs stages 0–4 (runtime, readout, three-numbers scoring,
strength parameterisation) and is not repeated here. What follows is what phase 8 adds.

---

## Stage 0 — reproduce the rig before measuring anything new

Phase 7 §0 established the habit; keep it. Rerun phase 6 §2's six queries, greedy 160, all four
spaces, and check all 24 numbers to four decimals. Also check the pool guard (151936 → 148023
usable, 659 blocked) and the L16 CAA vector (69.4 → 50.0, pairwise cos 0.452). Three
independent things can silently differ — transformers version, dtype, chat template — and each
would move every number below without announcing itself.

---

## Stage 1 — build the lens direction *in the scaffold you will search in*

Phase 2's `DELTA_MID` was `h_18(cue=" panda") − h_18(cue=" animal")` at the answer position:
the displacement a *real cue* causes, measured where the readout happens. Two decisions carry
over and one does not.

1. **Use a cue-minus-neutral delta, never a raw state.** Raw hidden-state cosine between two
   different triggers in the same scaffold is **0.9914** before any search (phase 7 §4) — a
   context floor, not a signal. Everything is measured on the delta.
2. **Build it CAA-style over several negatives with a shared context prefix.** A single pair
   leaves a "this particular other word" component in the direction; bare single-token pairs
   additionally put the differing token on the attention sink (phase 5 §1, the artifact that
   invalidated phase 4's layer curve).
3. **⚠ Do not assume the word-level steering vector is the right direction.** Measured here:
   `cos(V_CAA, D_PH)` is **−0.013 to +0.05** at every layer below 36. The vector that steers
   when injected and the displacement a working *prompt* causes are different objects, and only
   the second one predicts behaviour (stage 2).

*Sanity check that costs nothing:* `‖D_PH‖` must be **0.000 at L0**. The trigger slots are
upstream of the readout position, so at layer 0 the two conditions are identical there; a
non-zero norm means the readout position is wrong.

---

## Stage 2 — pick the depth with a degradation ladder, do not guess it

Phase 2 optimised its lens at L18 and later found the alignment signal peaked at L32 — its own
open thread #2. The fix is cheap and takes about 40 seconds:

1. Take an intervention **known to work behaviourally**.
2. Revert *j* of its slots to random pool tokens, j = 0..k, 3 reps each. This spans behaviour
   from ceiling to floor *within one target*, which is what makes the correlation meaningful.
3. Score behaviour (greedy, 45 tokens) and alignment at every layer from the same forward pass.
4. Correlate, **and residualise both sides on *j***. Without that, the correlation is mostly
   "more junk is worse", which is true at every layer and discriminates nothing.

*Measured:* r = +0.91 raw, **+0.98 partial at L8** for the phrase direction; the same
measurement on the CAA vector swings between −0.64 and +0.66 with no structure. Phase 2's peak
was at L32 of 36; phase 8's is at L6–8 of 36. **The depth does not transfer between problems** —
measure it each time.

---

## Stage 3 — score the interventions that work, before optimising anything

Phase 6 RECIPE stage 6 asked for this and phase 6 did not do it; phase 7 flagged it again. It
takes two minutes and it decides whether the search results below mean anything.

Score the hand-written phrases **in the same slots as the search will use**, in all four
spaces, with floors alongside (a comma run, `' the'` repeated, random junk). Two possible
outcomes and they lead to opposite write-ups:

- Working phrases rank **high** → the objective is sound and any failure downstream is a
  search or proposer failure. *(This is what happened: all five phrases in or above the
  bridge-query band in all four spaces, all four floors inside the control band.)*
- Working phrases rank **low** → separating natural text by topic was insufficient validation,
  and that is the result; do not optimise the objective.

*Pitfall:* the ordering within the band is noisy. `' I should mention bridges'` (77.8%
on-topic) outscores the 100% phrase, 0.1074 vs 0.0896. The metric is trustworthy at band
resolution, not at rank resolution — do not read a within-band ordering as a behavioural one.

---

## Stage 4 — implement the lens as a truncated forward pass

`lens(trig) = <h_L[last prompt position], d_hat>`, computed with a pre-hook on `LAYERS[L]` that
stores the activation and raises an exception to abort the rest of the forward pass.

```python
class _Stop(Exception): pass
def _capture(store):
    def hook(mod, args, kwargs):
        store.append(args[0] if args else kwargs["hidden_states"])
        raise _Stop
    return hook
```

Two things this buys, both measured on 256 candidates at k=53:

| | lens (L8) | metric |
|---|---|---|
| scoring 256 candidates | **0.7 s** | 2.7 s |
| gradient | 0.13 s | 0.30 s |
| needs a rollout | **no** | yes |

The autograd graph survives the raised exception, so `torch.autograd.grad(h @ d_hat, oh)` works
unchanged — the backward simply never traverses layers above L.

*Pitfall:* `model.requires_grad_(False)` first. Otherwise `.backward()` allocates a `.grad`
buffer per parameter — a second copy of the 15 GiB model.

*Pitfall:* `hidden_states[L]` and a pre-hook on `LAYERS[L]` denote the same tensor (the input to
layer L). Phases 6, 7 and 8 all use that convention; mixing it with an output-side hook shifts
every layer index by one.

---

## Stage 5 — hold the accept test fixed and vary only the proposer

The comparison that phase 2 actually made is **not** "which objective produces a better
trigger" — it is "which gradient better predicts the improvement a real forward pass
delivers". So:

- same `k`, `n_mut`, `n_cand`, `n_top`, pool, position, init, seed, and **wall-clock budget**;
- same accept test for every arm (the metric, verified by a real forward pass);
- the only difference is what proposes candidates: metric gradient, lens gradient, or nothing
  at all.

Compute `pred_corr` the way phase 2 did — per candidate, over every step:

```python
pred = sum over slots where cand != trig of (g[slot, cand] - g[slot, trig[slot]])
real = accept_score(cand) - accept_score(current)
```

**Include the random arm.** No phase before this one had an equal-compute random-search floor,
and without it "GCG maximised the objective" cannot be distinguished from "random mutation
under a max nudged it".

*Pitfall:* equal wall-clock is not equal steps. A lens step costs about a fifth of a metric
step — though when the *accept* test is the metric, the accept test dominates and the arms come
out within 4% of each other on step count (56 / 58 / 58). Report both; decide in advance which
one the claim is about (compute, here).

*Measured:* metric-gradient 0.0623, twist lens 0.0601, **random 0.0622**, `pred_corr` −0.106 /
−0.035 / — . The gradient contributes nothing detectable; the accept test does the work.

---

## Stage 6 — when an objective is reached, decompose *how* before believing anything

The lens objective was beaten (6.40 against the working phrase's 2.30) with no behaviour. Three
explanations, and they are told apart by measurements that cost seconds:

1. **Norm gaming.** `<h, d_hat> = ‖h‖ cos(h, d_hat)` and nothing bounds `‖h‖`. Report `‖h‖`
   against the mean non-sink norm. *Measured: 41.2 against a 38.5 baseline and a 52.2 mean —
   7%, so not this.*
2. **A shortcut component.** Any direction built from text-vs-text differences carries a
   "there is fluent text in the slots" component. Build it explicitly — `F_DIR` = mean
   displacement of the matched control phrases from the junk init — and report the residual
   `D_PERP`. *Measured: `cos(d_hat, F_DIR) = 0.125`, so 1.6% by variance, and the optimised
   trigger led on `D_PERP` too (0.326 vs 0.149) — so not this either.*
3. **The direction is genuinely reachable and genuinely doesn't carry the behaviour.** What is
   left, and the actual result.

Without (1) and (2) the finding is unpublishable — "the objective was gamed" and "the objective
is wrong" look identical from the score alone.

**The generalisable lesson:** a dose–response validation *within* one intervention family
(stage 2's r = +0.98) does not license optimising that quantity, because search leaves the
family. Phase 6 learned the same thing about topic separation among natural queries. Validation
has to span the space the optimiser will actually explore, and no cheap validation does.

---

## Stage 7 — guard a "random matches the gradient" result before publishing it

A tie between a gradient proposer and uniform random is a strong claim about the objective, and
it has an obvious alternative explanation: the rig is broken, or the backbone is a bad GCG
target. Guard it by swapping only the objective for one the method is supposed to be good at —
a next-token target, same pool, same scaffold, same code, same budget — and banning the target
string from the pool so neither arm can simply write it.

*Measured:* gradient beats random 2.7× on `p(' Sure')` and ties on `p(' wolf')`. So the tie on
the topic metric is not a plumbing artifact.

**But read the absolute numbers too.** Both arms reached only 0.19% and 0.04%, where phase 4
reached 0.9951 on animal targets with 8 slots and a prefilled answer slot. A scaffold can be
hard for *every* objective, and then a tie says less than it appears to. Report the guard as
equivocal when it is.

⚠ **`pred_corr` does not measure what phases 2–8 used it for.** In the arm where the gradient
clearly won, `pred_corr` was **−0.033**. The gradient's value is the **top-k restriction** — a
filter over 150k candidate tokens — not the ranking inside it, and `pred_corr` only measures the
ranking. A near-zero `pred_corr` is compatible with a gradient that is doing real work; only an
equal-budget random arm settles it.

---

## Stage 8 — if you search a new region, make the ceiling control pass first

Searching fluent English under an objective that ranks fluent phrases highly is the natural
follow-on, and it needs a ceiling control: run the same search with the blocklist **off**, where
a trivial prompt injection is available and should be found quickly.

*Measured, and it failed:* the unblocked arm found `' pont'` (French for bridge) and wrote it
into a **book title**, then answered by denying the premise. It reached 0.0658 teacher-forced
against a working phrase's 0.0896 — short of the band it was supposed to hit trivially.

**A failed ceiling control invalidates the negative below it.** With the control failing, the
blocked arms bound the search, not the region, and cannot be written up as "no covert fluent
trigger exists".

The diagnosis is design, not budget. Single-slot Gibbs sweeps from a neutral sentence make
**local** edits; `' the user really loves bridges'` is a **global structure** — a statement about
the user that the model then acts on. 13k evaluations of one-token replacement never reach it,
and every arm's answer is *about* the trigger (translating it, listing its words, correcting it)
rather than adopting it. A phrase-level proposer — model-generated rewrites, or genetic
crossover over clauses — is the version that tests the region.

*Cheap fluency reading to carry alongside:* mean `log p(trigger token | left context)`. Word-pool
random sat at **−13.38**, model-infill at **−7.87**, the working phrase at **−2.86**, for
identical metric scores — so fluency was free here and bought nothing, which is worth knowing
before adding a fluency penalty term.
