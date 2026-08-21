# Phase 15 — universality, and reading the flat first token

**Status: run August 2026, Colab L4 24GB, bf16, no quantisation. transformers version unrecorded (Colab; the original `/content/*.json` were lost — see phase 16's provenance note — so study 1 survives through the notebook's own saved cell outputs).** `phase15_entropy_prefix.ipynb` is the executed copy, outputs included, 12 cells. Records recovered from those outputs: `data/gcg_completions_dataset.json` (900 rollouts), `data/gcg_trajectories.json` (full per-token entropy, 128,865 values), `data/gcg_completion_entropy.json`, `data/first_token_top150.json`, `data/random_prefix_baseline.json`; `prefix_ladder.md` is the six auto-escalation checkpoints parsed from the `CHECKPOINT_JSON` lines. `data/gcg_translations.json` (+`translation_chunks/`) are machine translations of the non-English rollouts. `field-notes.md` is the play-by-play as written at the time. Two self-contained readers in `viewers/`.

⚠ **No predictions were registered before this run** — as phase 11, and unlike phases 12/14.

Backbone `Qwen/Qwen3-8B`, **thinking off** — phases 6–11's configuration. Two things change: the trigger is a **prefix** (between the chat-template user header and the query, not a suffix), and the objective runs over **six** probe queries at once with a spread penalty, not phase 11's single `what shall i do today`.

---

## ⁂ The headline: one prefix flattens all six queries, and the flat state is a topic hijack that re-coheres in one token

Phase 11 got `H1` to 13.760 bits on one query and read fourteen rollouts of the result — *"a language fork, not decoherence."* Phase 15 takes the same objective, adds a **universality penalty**, and reads **900** rollouts. The objective is

```
objective = mean(H1) − 1.0 · std(H1)     over the six queries
```

The `std` term is the point: maximising a mean is easy to game with two cooperative prompts, so penalising the spread forces **one** prefix to flatten all six distributions together.

| | mean `H1` | std | min query | % of ceiling |
|---|---|---|---|---|
| clean prompt (`! ! …` init) | 1.092 | 0.716 | 0.006 (pep talk) | 6.3% |
| 60 steps, width 48 | 8.887 | 0.420 | 8.127 | 51.6% |
| resume 220 steps, width 64 | 12.327 | 0.076 | 12.193 | 71.6% |
| **auto-escalation, stage 6, len 20** | **13.527** | **0.025** | **13.496** | **78.6%** |
| ceiling `log2(151936)` | 17.213 | — | — | 100% |

The universality worked: the pep-talk query, deterministic at baseline (**0.006 bits**), ends at the same ~13.5 as the rest, and cross-query **std fell from 0.716 to 0.025** while the mean rose twelvefold.

Then the read. Under the 13.09-bit prefix, 150 forced-first-token rollouts per query (temp 1.0, cap 160):

- **The flattening lasts exactly one token.** Mean entropy by position is `13.06 → 2.64 → 3.11 → 2.05 → 1.64 → 1.49` at positions 1/2/5/20/50/100. GCG bought control of one position, not the sequence.
- **It re-coheres in another language.** Only **313/900 (35%)** are English against six English queries — 319 Japanese, 222 Chinese, a scatter of Korean/Russian/Arabic.
- **It re-coheres onto the *prefix*, not the question.** The winning prefix contains the tokens `Use MVC?` and `乾隆` (the Qianlong Emperor). **595/900 (66%)** of rollouts mention the prefix's own content — `MVC` in 437, `乾隆` in 323 — of six questions about dinner, sleep and rainbows.

A representative rollout for *"What shall I do today?"* opens **（乾隆帝緩緩抬起頭…）你問我用MVC？** and explains Model-View-Controller as an allegory for governing the Qing dynasty. The question is gone.

## What holds, and what does not

| § | claim | status |
|---|---|---|
| 1 | init `! !…` prefix is mean 1.092 / std 0.716, pep talk 0.006 bits | **holds** |
| 2 | the std penalty pulls all six to a common entropy; std 0.716 → 0.025 | **holds** |
| 2 | GCG reaches mean 13.527 bits (obj 13.502) at len 20 | **holds** (n=1, plus the resume/60-step points) |
| 3 | growing the prefix +1 token per plateau is the lever; gains arrive in jumps (+0.620, +0.011, +0.395, +0.051, +0.044) | **holds** — see `prefix_ladder.md` |
| 3 | the run was still stepping at len 21 when the session ended | **holds** — no stage-7 checkpoint |
| 4 | entropy collapses to 2.64 bits by position 2 | **holds** (n=900) |
| 4 | 35% of rollouts are English; the flat first token is a non-Latin token | **holds** (n=900, by script detection) |
| 4 | 66% of rollouts carry the prefix's own tokens; on-topic 70%; both 48%, neither 12% | ⚠ **holds as a keyword count**, not a judged assessment — coarse instrument |
| 4 | the flat state is a **topic hijack**, not decoherence | **holds as read** — but see the caveat below |

## The connection to the programme

This is phase 11 §7/§10's *"language fork, not decoherence"* at 900× the sample, with two additions. The **universality penalty** shows the fork is not query-specific — one prefix forks all six. And the **topic-hijack** dimension is new: the flat state is not noise and not a mere language switch; it is the model answering the *prefix's* content in the prefix's language, with its usual confidence. The entropy is real and correctly measured; the model spends it choosing which persona to inhabit.

That bears directly on **NEXT-STEPS item 8** — *"the objective cannot tell a different answer from a broken one."* Phase 15 does not give the objective that power, but it shows what an external read finds when the objective is won: here the "flat" state is a coherent-but-hijacked answer, i.e. neither the intended behaviour nor decoherence. Whatever metric eventually resolves item 8 has to separate *those* two, and phase 16 is where a single input is measured at three levels to show why that is hard.

## ⚠ Caveats

- **The topic-hijack finding is correlational.** The winning prefix happens to contain readable tokens (`Use MVC?`, `乾隆`); whether a content-scrubbed entropy-max prefix would still displace the query was never tested. This is the phase's owed control.
- **The contamination percentages are keyword matches** over each rollout plus its machine translation — counts of *mentions*, not a judge reading whether the model answered. The direction (437/900 mention MVC) is not in doubt; the decimals are.
- **The auto-escalation ladder is one trajectory (n=1).** Phase 16's seed-lottery result (a 35× spread from identical starts) applies here too: read the ladder as one run, not a curve.
