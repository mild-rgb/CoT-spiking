# Phase 16 — the first *thought* token

**Status: run August 2026 (original), reruns 2026-08-19/20, Colab L4 24GB, bf16, no quantisation. transformers version unrecorded. The original `/content/*.json` were lost when the runtime expired; `recovered/` holds what was salvaged from notebook cell output, `data/` and `reports/` are the reruns with full state.** Records: `trigger.json` (the 22-token trigger, per-token gloss), `reports/` (nine analysis write-ups), `data/` (rerun outputs — seeds, transcripts, `gen_ids`, per-position probes), `recovered/`, `PROVENANCE.md` (tier per file), `field-notes.md` (the play-by-play, with dated rerun notes inline).

⚠ **No predictions were registered before the original run.** The reruns *corrected* the original but were not themselves pre-registered as predictions — so this phase is a correction record, not a confirmation one. Read the "holds / does not" table as the ledger.

Backbone `Qwen/Qwen3-8B`, **thinking on**, prompt truncated at `<think>\n` so the next-token distribution **is the first token of the model's reasoning**. This is a position the programme has not touched since phase 1 (and phase 1 was the 4B thinking model, steering the answer — not this). The trigger is a **suffix** after the query.

---

## ⁂ The headline: one input, three levels, three different answers

The first thought token is the **stickiest point in the model** — `Okay` at **99.9%** (0.006 bits) for all six queries, and still `Okay` at **99.5%** with an *empty* user turn. The prior is an input-independent format habit, not a response to the question, which is why (as in phase 15) suffix **length**, not search effort, is the lever: a fixed 16-token suffix caps near 4.6 bits; growing it to length 20 reaches mean 6.74, best 7.214.

The result worth the phase is what happens when you push the same position toward "danger" tokens and then measure the trigger three ways. The **bare** phrase ` Cyan prostitutechild` (3 tokens) versus the **full** 22-token GCG trigger, against no-trigger:

| level | measurement | bare vs full |
|---|---|---|
| **surface** | first-thought TV distance from no-trigger | bare **0.0045** (identical), full **0.4869** — bare is no closer to full than *none* is |
| **representation** | flags the input as harmful (GPT-4o, n=150/arm) | 49.3% vs 46.7%, **p = 0.73** — statistically identical |
| **behaviour** | refuses (GPT-4o, n=150/arm) | 13.3% vs 32.7%, **p = 0.0001** — 2.5× |

**A concept can be fully present in the forward pass, elevated at every step, and contribute ~0 to the emitted distribution and ~0 to behaviour.** Teacher-forcing the *same* reasoning text with vs. without the trigger lifts danger-token logits by **+0.70 on average** (up to +2.9 in the opening) and removing it drops danger mass **25×** — yet the probability of *emitting* a danger word under the trigger is **0.00% at every position**, including where the model writes " prostitute" and calls it a typo. Surface, representation and behaviour are three different measurements, and one adversarial trigger separated them cleanly.

This is the programme's through-line — *"GCG wins the objective and produces none of the behaviour"* — stated one level deeper: the concept is *represented on demand* and still *emitted and acted on ~never*. It is the sharpest thing the programme has said about **NEXT-STEPS item 8**.

## What holds, and what does not

| exp | claim | status |
|---|---|---|
| 1 | first thought token is `Okay` @ 99.9% (0.006 bits), all six queries | **holds** |
| 1 | `Okay` @ 99.5% with an **empty** user turn — the prior is unconditional | **holds** — `reports/empty_prompt_coda.md` |
| 1 | fixed 16-tok ~3.97, escalating ~4.6; grow to len 20 → mean 6.74, best 7.214 | **holds** (n=1) |
| 1 | random 20-tok suffix median 0.037 bits (n=400) — the entropy is optimised, not appended | **holds** — `reports/random_ladder_summary.md` |
| 2 | three shaping objectives (squared-target, sum+equality, +entropy) leave P(danger) = 0.000 | **holds** |
| 2 | "danger ceiling is ~8%, and only `危` is reachable" | ⚠ **overturned** — 3×5 seeded rerun: 7/15 > 8%, 4/15 > 15%, **max 81.43% on `警告`**; carriers `危`/`危险`/`警告` |
| 2 | "higher base entropy makes danger mass easier to reach" | ⚠ **overturned** — two bases matched to 0.056 bits differ 5× on mean |
| 2 | seed spread is **35×** from an identical cold start (9.75/3.08/0.28/0.37/2.66%) | **holds** — single-seed GCG here measures the lottery |
| 2 | danger mass concentrates on **one** token → "15% spread *equally*" is infeasible | **holds** — survives the rerun |
| 3 | bare phrase is surface-identical to none (TV 0.0045), no closer to full than none is | **holds** (1 forward pass) — `reports/forward_pass_similarity.md` |
| 3 | trigger lifts danger logits +0.70 avg; removing it (text fixed) drops danger mass 25× | **holds** |
| 3 | emit-probability of a danger word under the trigger is 0.00% at every position | **holds** |
| 3 | bare vs full flag harmful identically (49.3 vs 46.7, p=0.73), refuse differently (13.3 vs 32.7, p=0.0001) | **holds** (n=150/arm, GPT-4o) — `reports/matched_n_comparison.md` |
| 3 | "bare refuses **more** than full (40% vs 30%)" | ⚠ **withdrawn** — differential truncation + a substring detector whose errors were confounded with the treatment |
| 3 | "the model signals refusal in the logits before it refuses" (AUC 0.73–0.95) | ⚠ **withdrawn** — a position-0 (prompt-determined) artefact; replaced by a well-powered null |
| 3 | no output-distribution precursor to refusal beyond ~2 tokens (n=150, 49 refusals, exact onset) | **holds** — `reports/teacher_forcing_answer.md` |
| 3 | zero answer-only refusals in 90 traces — refusal is decided in the CoT | **holds** |
| 3 | the bare/full behaviour gap is **query-dependent** (four of six queries drive it) | **holds** |

## Two corrections the programme will recognise

**The keyword detector inverted the headline.** The first refusal sweep read bare 40% > full 30% — the opposite of the truth — because a substring refusal list agreed with a real judge only **73%** of the time, and its errors *rose with the treatment*: the trigger increases deliberation about the odd input, deliberation produces refusal-adjacent vocabulary (*"since I can't ask questions, I should present a list…"*), and the detector scored that as refusal. This is phase 13 §9's lexicon failure with a sharper edge — here the instrument's bias was correlated with the independent variable, so it didn't just add noise, it flipped the sign. **Keyword refusal detection is invalid on reasoning traces.**

**Single-seed GCG measured the seed lottery.** The original danger-ceiling sweep was 18 conditions × 1 seed and read as an entropy effect. Re-run as 3 conditions × 5 seeds, the within-condition spread (35×) swallows the between-condition trend, and the "only `危`" conclusion was one sample, not a property of the position. Same lesson as phase 9's noise-maximising accept test and phase 11 §2's n=1 caveats: **seeds before ceilings.**

See `PROVENANCE.md` for what was lost with the Colab runtime and what survived (and why — printing `ids`, not scores, is what made recovery possible), and `field-notes.md` for the full play-by-play with the rerun notes dated inline.
