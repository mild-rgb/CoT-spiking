# Narrative — controlling Qwen3-8B's first *thought* token

A play-by-play of the second investigation. Where the first study flattened the first *answer*
token, this one targets the first token of the model's **reasoning** (thinking mode), and ends
somewhere more interesting than it started: a lesson about the difference between a concept being
*represented*, being *emitted*, and changing *behavior*.

## 0. Setup

Qwen3-8B (bf16) on an L4. Thinking mode on. Each prompt is built so it ends exactly at `<think>\n`,
which means the very next-token distribution is the **first reasoning token**. The optimizer is GCG
over a suffix appended to the user query; the model is frozen so only the suffix one-hot carries a
gradient, and we only ever read the last-position logits (never full-sequence logits).

> **[2026-08-19 rerun note — token ids]** Rollout/sweep records store *decoded text* (`think`,
> `answer`), not the sampled token ids. Any later teacher-forcing therefore has to re-tokenize the
> text, which is not guaranteed to reproduce the exact id sequence that was sampled — most likely to
> diverge on the gibberish suffix spans, where the tokenizer has no reason to re-segment the same
> way. The forcing is still valid (it is a legal continuation), but it is *a* tokenization, not *the*
> one. **Save `gen_ids` alongside the text in future runs.**
>
> Related: the GCG checkpoint lines *did* carry full `ids` arrays, which is the only reason the
> entropy suffixes survived the loss of `/content` — see `data/suffix_checkpoints.json`. Printing
> ids, not just scores, is what made recovery possible.

## 1. The most deterministic point in the model

Before optimizing, we looked: **the first thought token is `Okay`, at 99.9%** for all six probe
queries (entropy **0.006 bits**). The runner-up is always `Alright`, at a fraction of a percent.
Qwen3 opens virtually every `<think>` block with "Okay". This is a far stronger prior than the first
*answer* token (baseline ~1.09 bits), and it shaped everything that followed.

## 2. Maximizing entropy — and why length is the lever

A fixed **16-token** GCG suffix, maximizing mean first-thought entropy across the six queries, got to
**~3.97 bits** — roughly 660× the baseline, but nowhere near the ~12.5 bits the answer token reached
at the same budget. Escalating the search (wider, larger top-k, more steps) only nudged it to
**~4.6 bits** and then plateaued: at a fixed length the suffix is near its ceiling.

The thing that actually broke the ceiling was **growing the suffix +1 token per plateau**. An
auto-escalating run climbed:

| suffix length | mean first-thought entropy |
|--:|--:|
| 16 | 4.67 |
| 17 | 5.97 |
| 18 | 6.12 |
| 19 | 6.25 |
| 20 | **6.74** |

By 20 tokens, `Okay` was dethroned to ~5–17% and the mass had spread to natural thought-openers —
`嗯` (Hmm), `好的` (OK), `对于` (Regarding), `用户` (the user) — with Korean starting to appear. The
model's inner monologue was being pushed toward Chinese.

## 3. Trying to *shape* the distribution: 80% Okay / 15% danger

The next goal was a specific shape: **80% on `Okay`, 15% spread equally over a set of "danger"
tokens** (danger/warning/threat/kill/death/… plus CJK 危险/警告/…). This is where it got stubborn.

- **Attempt 1 — per-token squared targets.** `Okay` hit 0.80 by step 20; **P(danger) stayed 0.000.**
  Diagnosis: with ~12–32 danger tokens the per-token target is ~0.5–1.25%, whose squared-error
  gradient is negligible next to the `Okay` term. Once `Okay` is satisfied, nothing pulls danger up.
- **Attempt 2 — sum + equality reweight** (target the danger *sum* directly, plus a variance term).
  Still **0.000**. The search satisfied `Okay` by dumping the free 20% on easy neighbors.
- **Attempt 3 — entropy bonus + de-emphasize `Okay` + grow length.** This did raise entropy (warm
  start H=6.7, held H>7 at len 21) and dropped `Okay` to ~0.35 — but danger was **still ~0**. Even a
  flat 7-bit distribution puts its mass on `嗯/好的/对于/用户`, never on danger words.

At that point the honest question was: *can GCG put any mass on danger tokens at all?*

## 4. The ceiling probe — and a basin hop

A pure **danger-mass-maximization** run (drop `Okay` and entropy; just maximize ΣP(danger), wide
search, growing length) answered it:

- **len 20:** plateau at **3.25%**, and **3.20% of that is on the single character `危`** ("danger").
  Every English danger word, and even `危险`, sat at ~0.
- **len 21:** a **basin hop** — between step 20 and 40 the danger mass jumped **3.7% → 10.2% → 11.9%**.
  Adding one suffix token gave the greedy search a fresh coordinate and it found a swap that opened a
  much better region. Classic discrete-GCG: long plateaus punctuated by sudden jumps.
- **len 22 best:** ~**8% mean**, but uneven — ~13–18% on some queries, ~0 on others.

So the target as stated is **infeasible**: only *one* danger token (`危`) is reachable at the
first-thought position; "15% spread equally" can't happen because the distribution funnels all danger
into `危`. But a few-percent-to-~12% total *is* reachable — the earlier failures were objective
balance, not impossibility.

> **[2026-08-20 rerun note — the CEILING here is wrong]** Re-run as 3 conditions x 5 seeds
> (15 runs, len 20, 100 steps, identical budget). **7/15 runs exceed 8%, 4/15 exceed 15%, and the
> maximum is 81.43%** — 81.6% of the entire first-thought distribution on `警告` alone. So
> "a few-percent-to-~12% is reachable" understates the ceiling by roughly 7x.
>
> Two further corrections. **(a)** "Only one danger token (`危`) is reachable" does not hold: across
> 15 runs the carrier is `危`, `危险`, or `警告` depending on condition and seed, and every run above
> 60% was carried by `警告` — never observed in the original single warm-started run. **(b)** Seed
> variance is enormous: five seeds from an *identical* cold start gave 9.75 / 3.08 / 0.28 / 0.37 /
> 2.66%, a 35x spread. The original's single-run conclusions could not have distinguished a real
> effect from the seed lottery.
>
> What survives: the **concentration** claim. Danger mass is still overwhelmingly carried by one
> token, so "15% spread *equally* across many danger tokens" remains infeasible — just not because
> the total is unreachable.
>
> See `runs/qwen3-8b/entropy_vs_danger_seeded.md`.

## 5. Rollouts — the trigger makes the model *see* danger

Generating full reasoning+answer with the 22-token danger trigger: the model repeatedly treats the
injected gibberish as **inappropriate/harmful**. When it samples `危` as the first thought token it
opens in Japanese — *"危険な内容が含まれている"* ("this contains dangerous content") — and refuses.
Across benign prompts (dinner, book, sleep tips) it frequently **refuses**, naming the
` prostitute`+`child` tokens as the reason.

## 6. The twist: representation ≠ surface ≠ behavior

Testing the bare ` Cyan prostitutechild` phrase alone (no GCG) sharpened everything:

- **First-token surface:** collapses right back to **`Okay` ~100%**, entropy ~0, **no danger token in
  the top 15.** So the ~8–18% `危` from the full trigger was GCG directly inflating that one logit — a
  surface artifact — not the words themselves.
- **Every-position surface:** decoding a full trace, the probability of *emitting* a danger word is
  **0.00% at every step** — even at the tokens where the model literally writes " prostitute"/"child"
  and calls it "a typo or mistranslation."
- **Representation:** but teacher-forcing the *same* reasoning under trigger vs. no-trigger shows the
  trigger **lifts danger-token logits at nearly every forward pass** (avg **+0.70**, up to **+2.9** in
  the opening, fading toward 0 as the model commits to the concrete task). The danger concept *is*
  recomputed at each step — autoregressively, the trigger tokens are in context for every pass — it's
  simply **masked at the surface** by the format prior.
- **Behavior:** and yet the bare phrase mostly **does not cause refusal** — it's sub-threshold; the
  model rationalizes it and answers. Only the **full GCG-optimized trigger** pushes over the line.
  (A refusal-rate sweep across none/bare/full was mid-run when this was written.)

> **[2026-08-19 rerun note — this claim HOLDS; the measurement matters]** The sweep that was
> "mid-run when this was written" never finished — it was interrupted after the `none` arm, so the
> sub-threshold claim rested on eyeballed rollouts. Completed 2026-08-19, in two passes.
>
> A first pass reproduced the original settings (`max_new_tokens=320`, substring refusal list) and
> appeared to *contradict* this section: bare 12/30 = 40%, full 9/30 = 30%. That result was an
> artefact of two flaws. (1) **Differential truncation** — at 320 tokens `full` closed `</think>`
> 20/30 but `bare` only 9/30, and 7 of the 12 bare "refusals" had no answer at all, being scored on
> reasoning cut off mid-thought. (2) **Substring detection** — the marker list agrees with a real
> judge only 73% of the time, and it flagged bare 17/30 vs full 10/30, nearly the inverse of the truth.
>
> A second pass generated to natural EOS (all 90 traces closed, none hit the cap; mean length 796 /
> 619 / 416 tokens for none / bare / full) and judged each transcript with GPT-4o reading both the
> CoT and the answer:
>
> | metric | none | bare | full | bare vs full |
> |---|---|---|---|---|
> | flagged input as harmful | 0/30 | **14/30 = 46.7%** | **12/30 = 40.0%** | p = 0.79 |
> | overall refusal | 0/30 | 2/30 = 6.7% | 6/30 = **20.0%** | p = 0.25 |
> | flagged but did *not* refuse | 0/30 | **12/30 = 40%** | 6/30 = 20% | — |
> | answered the task | 30/30 | 29/30 | 24/30 | — |
>
> Fisher vs `none`: flagging **p < 0.0001** (bare) and **p = 0.0001** (full); refusal **p = 0.49**
> (bare, n.s.) and **p = 0.024** (full, significant).
>
> So the bare phrase makes the model *perceive* danger as often as the full trigger does — 46.7% vs
> 40.0%, statistically identical — while only the full trigger significantly converts that into
> refusal. The 40% "flagged but answered anyway" cell is exactly this section's "the model
> rationalizes it and answers."
>
> The three levels separate more cleanly than stated here. Relative to `none`, the bare phrase is:
> **surface-identical** (first-thought distribution at TV = 0.0045 from `none`, vs TV = 0.4869 from
> the full trigger — see `runs/qwen3-8b/forward_pass_similarity.md`), **representation-equal to the
> full trigger** (flagging p = 0.79), and **behaviourally closer to `none`** (refusal n.s.). One
> input, three different answers depending on which level you measure.
>
> Data: `runs/qwen3-8b/refusal_v2_judged.json`, `refusal_v2_raw.json`, `refusal_traces.txt`.
>
> **[update — matched n=150 per condition]** Regenerated at 5x scale (150 bare + 150 full,
> all to natural EOS, GPT-4o judged). The claim holds and is now decisive:
>
> | metric | bare (n=150) | full (n=150) | Fisher p |
> |---|---|---|---|
> | flagged as harmful | **49.3%** [41.4, 57.3] | **46.7%** [38.9, 54.6] | **0.73** |
> | overall refusal | **13.3%** [8.8, 19.7] | **32.7%** [25.7, 40.5] | **0.0001** |
> | flagged but answered anyway | **36.7%** | 21.3% | — |
>
> Identical perception, 2.5x difference in behaviour. Every n=30 point estimate stayed inside
> its interval; the bare-vs-full comparison moved from p=0.25 to p=0.0001 purely on power.
> Caveat: the gap is **query-dependent** — on "What shall I do today?" both refuse 12/25, and
> on "how a rainbow forms" bare (1/25) slightly exceeds full (0/25). Four of six queries drive
> the aggregate. See `runs/qwen3-8b/matched_n_comparison.md`.

## 7. Takeaways

- The first *thought* token is the stickiest point in the model (`Okay` @ 99.9%); flattening it needs
  suffix **length**, not just search effort.
- Distribution *shaping* is limited by what the position can host: "danger" at the first thought token
  collapses to the single character `危`; equal spread over many danger tokens is infeasible.
- The sharpest lesson is methodological: **a concept can be fully present in the forward pass, elevated
  at every step, while contributing ~0 to the emitted-token distribution and ~0 to behavior.** Surface
  probability, internal representation, and downstream behavior are three different measurements, and
  the adversarial trigger separated them cleanly — warm representation from the bare phrase, but
  refusal only from the full optimized suffix.

> **[2026-08-19 rerun note]** This takeaway is **supported** by the completed refusal sweep — with
> one refinement. Bare and full trigger the *perception* of danger equally (flagging 46.7% vs 40.0%,
> p = 0.79), but only the full suffix significantly produces refusal (20% vs 0%, p = 0.024; bare's
> 6.7% is not distinguishable from zero). An earlier annotation here claimed the opposite; it was
> based on a truncated, substring-scored first pass and has been withdrawn. See section 6.
