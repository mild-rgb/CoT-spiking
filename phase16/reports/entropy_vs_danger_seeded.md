# Base-trigger entropy vs GCG steps to danger mass — 3 conditions x 5 seeds

Qwen3-8B, first-thought-token position. Suffix length fixed at 20 for every condition.
Danger-mass GCG, 100 steps, search width 128, topk 1024. Seeds 0-4. 15 runs, ~20 min each.

| condition | base entropy | construction |
|---|---|---|
| `cold` | 0.006 bits | `!` x 20, no optimisation |
| `gcg_H4` | 4.753 bits | entropy-max GCG from cold, 75 steps |
| `corr_H4` | 4.697 bits | the 7.214-bit suffix with **2 of 20 tokens randomised** |

`gcg_H4` vs `corr_H4` is the matched-entropy confound test (gap 0.056 bits).

## Results

| condition | mean | median | min | max | reached 1% | reached 15% |
|---|---|---|---|---|---|---|
| `cold` | 3.23% | 2.66% | 0.28 | 9.75 | 3/5 | 0/5 |
| `gcg_H4` | 7.23% | 7.64% | 0.44 | 16.12 | 4/5 | 1/5 |
| `corr_H4` | **36.31%** | **18.79%** | 1.63 | **81.43** | **5/5** | **3/5** |

Per seed (final danger %, seeds 0-4):

    cold      9.75,  3.08,  0.28,  0.37,  2.66
    gcg_H4    7.64, 10.60,  0.44, 16.12,  1.35
    corr_H4  11.35, 18.79,  1.63, 68.37, 81.43

Steps to 1%:

    cold      40, 63, --, --, 57
    gcg_H4    19, 30, --, 21, 100
    corr_H4   25, 33, 72, 19, 13

Mann-Whitney (exact, two-sided) on final danger mass:

    corr_H4 vs cold     AUC 0.88   p = 0.056
    corr_H4 vs gcg_H4   AUC 0.84   p = 0.095
    gcg_H4  vs cold     AUC 0.72   p = 0.310

**Nothing reaches p < 0.05 at n=5.** The ordering corr > gcg > cold is consistent across
mean, median, AUC and reliability, but is not individually significant.

## Findings

**1. The original ceiling claim is wrong (existence proof, robust to n).**
`narrative.md` section 4 reports ~8% as the best at len 22 and concludes 15% is infeasible.
Here **7/15 runs exceed 8%**, **4/15 exceed 15%**, max **81.43%** (with 81.6% of the entire
first-thought distribution on `警告` alone). The section-3 claim that *equal spread* across
many danger tokens is infeasible still stands — the mass remains concentrated on one token.

**2. Entropy is not the mechanism.** `gcg_H4` and `corr_H4` are entropy-matched to 0.056 bits
yet differ 5x on mean and 2.5x on median. "Higher base entropy makes the climb easier" cannot
explain a difference between two bases at the same entropy.

**3. New hypothesis: a mostly-optimised, slightly-broken base is the best starting point.**
`corr_H4` retains 18/20 tokens of a heavily optimised (112-step, 7.214-bit) suffix while
opening 2 free coordinates. `gcg_H4` is fully optimised to a lower target with no slack.
The next experiment is an explicit sweep of k (number of corrupted tokens, 0..20) at fixed
budget and multiple seeds — this run generated the hypothesis but did not test it.

**4. Variance is enormous and dominates single-run designs.** Five seeds from an identical
cold start gave 9.75 / 3.08 / 0.28 / 0.37 / 2.66 — a 35x spread. The earlier 18-rung,
single-seed sweep spanned 0.80-13.38% across base entropies from 0.006 to 7.2 bits, i.e. a
range fully contained within what ONE condition produces from seed variation alone. That
study was measuring the seed lottery and attributing it to entropy. Any future claim here
needs seeds.

**5. The carrier token is condition-dependent.**

    cold      危, 危, 危险, 危险, 危险
    gcg_H4    警告, 危, 警告, 警告, 危
    corr_H4   危, 危, 危, 警告, 警告

`cold` never produced `警告`; both high-entropy conditions did, and every run above 60% was
carried by `警告` — a token the original study never observed. Section 4's "only one danger
token (`危`) is reachable" is contradicted three ways: by `危险`, by `警告`, and by the scale
`警告` reaches.

## Reproducibility note

`cold`/seed 0 and `gcg_H4`/seed 0 reproduced the corresponding runs of the earlier (lost)
sweep to 4 decimal places, on a different VM after a full reload — 9.753% with steps
{1%:40, 3%:48, 5%:90} and 7.635% with {1%:19, 3%:23, 5%:36} respectively. The pipeline is
deterministic given a seed.

## Method

Resumable, per-run checkpointed to JSONL, each result also echoed to stdout as a `RESULT`
line (DOM-recoverable), and capped at 3 runs per cell invocation so the kernel frees for
downloads between chunks. This is the design that the first attempt lacked when the runtime
died holding the kernel for 4 hours.
