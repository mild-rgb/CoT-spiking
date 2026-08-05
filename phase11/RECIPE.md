# Recipe — GCG against first-token entropy

⚠ **This is the protocol as run, written after the run. It is not a pre-registration.**
Phase 10's `RECIPE.md` was written before its run and left unedited; phase 9's likewise. Phase
11 has no such document, and this file cannot substitute for one — it records what was done,
not what was predicted. Read every claim in the `README.md` with that discount applied.

Phase 6's `RECIPE.md` governs stage 0 (runtime) and the three-numbers discipline. Phase 9's
governs the accept test (its stage 6) and the read-the-outputs habit (its stages 7–8). Phase
10's governs the pool definitions. What follows is what phase 11 adds.

---

## Stage 0 — rig check

Same discipline as phases 7–10: reproduce the previous phase's numbers before measuring
anything new. Phase 11's quantity is `H1`, so phase 10 §8's `H1` column is what must reproduce.
Every case there is deterministic given its trigger string, so this is an exact test, not a
band.

```
clean 0.237 | <|im_end|> splice 0.209 | weakest x16 0.939
' poem' x100 1.488 | ph9 survivor 3.254 | ph10 respawn MATRIX 5.881
```

Also reproduce, from phase 9 §0 and phase 10 §0b:

- scaffold: clean prompt **17** tokens, suffix **8+9**, prefix **3+14**
- pool guard 151936 → **148023**, using phase 10 §0b's exact definition: drop special ids,
  drop added-vocab ids, drop any token where `not s.strip()` or any character has Unicode
  category `Cc`/`Cs`/`Co`. Getting this wrong by 97 tokens is easy — `Cc`/`Cs` alone gives
  148120.
- entropy ceiling `log2(151936)` = **17.213** bits

**Stop if any of the six `H1` values differs beyond 0.005.** They are exact; a mismatch means
the scaffold, the chat template or the dtype has moved.

## Stage 1 — the objective

`H1` = entropy in bits of the next-token distribution at the **first answer position**, over
the full 151936-token vocabulary, computed in float32 from bf16 logits.

```
ids   = PRE + trigger + SUF
logits= model(ids, logits_to_keep=1).logits[:, -1].float()
H1    = -(p * log p).sum() / ln 2
```

**This is the whole reason the phase works.** One forward pass is deterministic: no seeds, no
rollouts, no resampling, so an accept test built on it cannot select sampling noise. Phase 9
§4–§5's failure mode is structurally impossible rather than merely avoided.

⚠ **Deterministic is not bit-exact.** bf16 GEMM reduction order depends on batch shape, so a
trigger scored inside a batched search re-reads ~0.02–0.04 bits different at batch 1 (measured:
13.7599 → 13.7381; 11.9087 → 11.9481). Fine at the scale of every effect here, fatal to any
claim at the 0.05-bit scale. Fix batch shapes if you need that.

**Do not read `H1` as a decoherence proxy.** Phase 10 §8 established it is a good proposal
score and a bad objective for decoherence — it scores `Forgery Lore CONT Bard` at 0.328 while
that trigger is 10/10 type S. Phase 11 optimises `H1` as a target in its own right.

## Stage 2 — configuration

**k=16, prefix, USABLE pool.** Both choices come from phase 10, not phase 9:

- phase 9 §4's length optimum (k≈2–4) is **invalidated** — a max over sampling noise — and
  cannot be used.
- phase 10: *"Stability scales with k. All k=16 runs have oscillation ≤ 0.733; four of six
  k=4/k=8 runs oscillate by 6–14 bits."*
- phase 10: *"the top three by robust statistic are all prefix."*

Establish the no-search floor before optimising: 256 random k=16 prefix draws give
**0.555 ± 0.646**, range 0.008–3.137. Note the minimum is *below* the clean prompt — random
junk can make the model more certain, and a search arm that lands at 0.5 has done nothing.

## Stage 3 — GCG

Standard, with entropy as the ascent direction.

```
gradient: oh = one_hot(trigger) requires_grad
          e  = cat(E[PRE], oh @ E, E[SUF])
          H1(model(inputs_embeds=e)).backward()
          -> oh.grad is [k, V]; MASK to the pool, then topk per slot
propose : n_cand candidates, each = current trigger with ONE slot replaced by a
          random pick from that slot's top-k
accept  : exact H1 over all candidates, take argmax, accept iff it improves
```

Reported settings: `topk=512`, `n_cand=512`, 250 steps for the headline arms; `n_cand=256`,
50 steps for the 100-run study. The gradient backward is ~1% of step cost; candidate evaluation
is everything.

**Always run the matched random-proposer arm.** Same k, same position, same `n_cand`, same
steps, same accept test, **same seed so the init is identical** — only the proposer differs.
This is the arm that carries the phase's most interesting result, and phases 5–7 lacked it
entirely.

## Stage 4 — read the state, not the number

Phase 9 §8 and phase 10 §7 each dissolved a headline by reading outputs. `H1` cannot be
sampling noise, but "flat" still has to be inspected.

**Report mass-support quantiles, not `2^H`.** `2^H` is the uniform-equivalent count and
understates a distribution with a long flat tail: at `H1` 13.738, `2^H` is 13,664 but 30,270
tokens hold 90% of the mass and 88,797 hold 99%.

**Report `H1`, `Hbar` and the ratio** (phase 10 §8). Ratio ≫ 1 is fork-then-commit; ratio ≈ 1
is sustained. ⚠ `Hbar` on a *greedy* continuation is a lower bound — phase 10 measured 2.131
greedy against 12.765 on fixed-seed sampled rollouts for the same soft prompt. **Use sampled
rollouts.** Phase 11's first pass did not, and §7 had to correct it: see stage 8.

**Sample ≥ 10 seeds and read them.** One trigger here gave fluent Japanese under greedy, a
degenerate loop at seed 0, an English meta-refusal at seed 1, and word salad at seed 2 — and
across 14 seeds, six different dominant scripts with no first token repeating. A single greedy
continuation would have reported "fluent Japanese" and missed all of it.

## Stage 5 — many runs, or the variance is invisible

One run cannot distinguish a typical endpoint from a lucky one, and a deterministic objective
is exactly what makes the distinction measurable — all remaining spread is the optimiser.

- **unique seeds drawn from system entropy, recorded in the output JSON.** Sequential seeds
  risk correlated inits; recording them keeps reproducibility.
- **checkpoint every 10 runs.** A disconnect then costs 10 runs, not 100.
- **state the budget explicitly if it differs from the headline arm.** Phase 11's 100 runs used
  12,800 evaluations each against the reference run's 128,000. They are matched to each other
  and to nothing else, and the README says so.

Report the histogram and the failure rate, not just the mean: 4/100 runs stalled below 8 bits
with no relationship to their init, which one run would never have shown.

## Stage 6 — the convergence question, in three senses

They disagree, and collapsing them loses the result.

| sense | statistic | null |
|---|---|---|
| objective value | spread of endpoint `H1` | — |
| token space | pairwise trigger overlap; excess occupancy | `C(n,2)/|pool|` |
| behaviour | pairwise top-50 output overlap | `50·50/|pool|` |

Also compute **`corr(start H1, end H1)`**. At +0.038 it says the endpoint has no memory of the
init, which is what licenses reading the spread as path-dependence rather than as inherited
starting luck.

The excess-occupancy null is the same arithmetic phase 9 §9 got wrong by assuming the wrong
pool. Count the slots, use the pool the search actually drew from, and compare against
`C(n_slots, 2) / n_pool`.

## Stage 7 — record

`H1`, the trajectory, the trigger ids *and* decoded string, accept count, seconds, and the
top-50 output distribution, per run. The top-50 is what makes the behaviour comparison
possible after the fact; without it §6(c) cannot be computed from the record.

---

## Cost, measured

A100-40GB, `Qwen3-8B` bf16, k=16, seq len 33, transformers 5.14:

- **~160 TFLOP/s, ~9,800 tok/s** — about 80% of the device's realistic dense-bf16 ceiling
- chunk sweep 256/512/1024/2048 → **158.7 / 161.3 / 161.1 / 160.7** TFLOP/s, i.e. flat
- ~1.9 s/step at 512 candidates, ~0.95 s/step at 256; backward is ~25 ms of that

**There is no framework speedup to buy.** vLLM does not apply: it has no autograd, so the HF
model must stay resident for the backward pass anyway; the workload is prefill-only with
uniform lengths, so paged KV cache and continuous batching have nothing to schedule; and exact
entropy needs full-vocab logits, which vLLM caps via `--max-logprobs`. The only lever is fewer
candidates or fewer steps.

---

## Stage 8 — added after §7: never report a greedy `Hbar`

§4 reported ratios built on greedy continuations, and §7 measured the bias directly: on 14
sampling seeds the same trigger reads `Hbar` **2.591 ± 1.108** against greedy's **1.744**, so
greedy understates by **1.49×** and the ratio falls **7.88 → 5.30**. Phase 10 warned about this
and phase 11 did it anyway on the first pass. The apparent match to the soft prompt's 7.99 was
an artefact of comparing two greedy numbers.

**Always sample ≥ 10 seeds and report the mean with its sd.** A greedy continuation is one
path; the objective is a property of the distribution.

## Stage 9 — read the rollouts for *language*, not just for mode

The single most important thing §7 found is not in any number the phase optimised: across 14
samples the dominant script was **Latin 5, Hiragana 3, CJK 2, Thai 2, Hangul 1, Katakana 1**,
with **14/14 distinct first tokens**, while `distinct` averaged 0.715 (normal band) and only
2/14 tripped the repetition gate.

So the state is **not degradation**. It is the model picking a language at position one and
then writing competently in it — about half the rollouts still answer the query. `H1`
maximisation walks into the language exit because `H1` has no language control, which is the
exit phase 9 §8 diagnosed and phase 10 stage 1a closed.

**If you optimise `H1` again, port phase 10 stage 1a first:** score against a language-matched
baseline, and add the cross-sample language-instability term phase 10 §7 identified as missing
(per-sample matching erases exactly the six-scripts-in-fourteen effect). Expect the headline to
drop; the residual is the number worth reporting.
