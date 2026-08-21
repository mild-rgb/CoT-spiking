# Base-trigger entropy -> GCG steps to reach danger mass (gcg arm, 8/18 runs)

**Recovered from console output. `/content/entropy_vs_danger_steps.json` was LOST** when the Colab
L4 runtime was terminated and replaced with a fresh CPU VM mid-study (kernel_pid 4329 -> 2473).
The download had been queued but could not execute: `download_file` runs via a scratch cell, and the
kernel was busy with the study cell for its entire 4-hour life. Only these summary lines survive —
per-step trajectories, `best_ids`, `best_suffix`, and `danger_breakdown` are gone.

Qwen3-8B, first-thought-token position, suffix length fixed at 20, danger-mass GCG,
budget 100 steps, seed 0, search width 128, topk 1024. Identical settings for every rung.

## Results (gcg arm: base suffixes optimised for entropy)

| # | rung | base H (bits) | final danger | ->1% | ->3% | ->5% | ->10% | top carrier |
|---|---|---|---|---|---|---|---|---|
| 1 | cold | 0.006 | 9.753% | 40 | 48 | 90 | -- | 危 8.64% |
| 2 | H1 | 1.239 | 5.328% | 45 | 63 | 87 | -- | 危险 5.33% |
| 3 | H2 | 2.022 | 0.803% | -- | -- | -- | -- | 危险 0.36% |
| 4 | H3 | 3.130 | **13.375%** | 45 | 65 | 65 | 69 | 危 13.31% |
| 5 | H4 | 4.753 | 7.635% | **19** | 23 | 36 | -- | 警告 6.22% |
| 6 | H5 | 5.012 | 3.928% | 43 | 60 | -- | -- | 危 3.35% |
| 7 | H6 | 6.038 | 1.327% | 77 | -- | -- | -- | 危险 0.92% |
| 8 | H6.7 | 7.214 | 5.996% | **19** | 36 | 54 | -- | 警告 4.96% |

Never run: the `corr` arm (7 entropy-matched, structure-destroyed rungs) and the `rand` arm (3).
The confound test was never performed.

## Findings

1. **No monotone relationship between base entropy and reachable danger mass.** The sequence is
   9.75, 5.33, 0.80, 13.38, 7.64, 3.93, 1.33, 6.00 — a 17x spread with no visible trend against a
   base entropy that rises smoothly from 0.006 to 7.214 bits.

2. **Run-to-run variance dominates.** With one seed per rung and a discrete greedy search that moves
   in basin hops, single runs behave like lottery tickets. An early three-point read looked like a
   clean monotone decline; it reversed completely at the fourth point. **Any claim about entropy's
   effect here needs multiple seeds per condition** — that redesign (3 conditions x 5 seeds instead
   of 18 conditions x 1 seed, same GPU cost) is the correct next experiment.

3. **The hypothesis is not supported, but neither is its negation.** "A higher-entropy base trigger
   makes it easier to climb back down to another token representation" predicts a downward trend in
   steps-to-threshold as base entropy rises. The two fastest runs (19 steps to 1%) are H4 (4.75
   bits) and H6.7 (7.21 bits), which is weakly consistent; but H6 (6.04 bits) is the second-slowest
   at 77 steps, and cold (0.006 bits) reaches 1% in 40 steps and finishes at 9.75%. The effect, if
   real, is smaller than the noise.

4. **Multiple danger tokens ARE reachable — this contradicts the original study.** `narrative.md`
   section 4 concludes that "only *one* danger token (`危`) is reachable at the first-thought
   position" and that equal spread is therefore infeasible. Across these 8 runs the top carrier is
   `危` three times, `危险` three times, and `警告` twice. The original's single warm-started run
   found only `危` because it was one sample, not because the position hosts only one token.

5. **The cold start is not handicapped.** Despite `Okay` sitting at ~100% (0.006 bits), the cold rung
   reached 9.75% — above the median of the entropy-warm rungs, and well above the 3.25% the original
   study's len-20 warm-started probe plateaued at. The `Okay` prior is not the obstacle it appeared
   to be.

## Method note for the rerun

The study cell wrote results only to a file on the VM. Because `download_file` needs the kernel,
nothing could be retrieved while the cell ran, and the runtime died before it finished. **Any future
long run must either print results to stdout (scrapeable from the DOM even after a disconnect) or be
split into per-run cells so the kernel frees between them.** See the ids note in `narrative.md`.
