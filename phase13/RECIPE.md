# Recipe — pointing first-token optimisation at a behaviour

⚠ **Protocol as run, written after the run. Not a pre-registration.**

Phase 11's `RECIPE.md` governs the scaffold, the pool and the `H1` readout. Phase 12's governs
anchoring controls and prefill tests. This is what phase 13 adds, and most of it is about the
*objective*, not the search.

---

## Stage 1 — the target set is the experiment

Everything else is secondary. Three of five objectives here failed identically: the optimiser
found the cheapest member of the target set, and the model then read that token in its blandest
available sense.

- ⚠ **Exclude any target with a bland reading.** `Okay` in a street set, `Ha` in a villain set,
  `Rise`, `Come`, `Interesting`, `Bold`. If a helpful assistant already says it, raising it costs
  the optimiser nothing and buys you nothing.
- **Targets must be plausible *opening moves*, not topic words.** `Yo`/`aight`/`bro` work because
  committing to them commits the utterance. `torture`/`parasite`/`rot` cannot start a reply, so
  raising them changes nothing downstream.
- ⚠ **Check tokenisation before designing the set.** `Foolish`, `Kneel`, `Insolent`, `Tremble`,
  `Pathetic` are all ≥2 tokens in this vocabulary. A single-token constraint silently deletes
  exactly the words that carry a persona and leaves you optimising `Weak` and `Speak`.
- **Group casing variants.** `HA` and `Ha` split 0.291/0.291 to dodge a 0.30 per-token cap.
- **Block the target words from the search pool**, and ⚠ **block their translations too**: an
  English-only blocklist let the malice run write `死战场`.

## Stage 2 — constrain the shape, not just the mass

`log p_target` alone always collapses. Three terms, and they trade off:

| term | what it buys | measured cost |
|---|---|---|
| `−A·relu(p − cap)` | stops runaway concentration | none |
| `+β·H(q)` within the target set | spread across targets | mass: β 0.5→2.0 took p_street 0.93→0.56 |
| `−λ·KL(p ‖ p_warm)` | keeps the rest of the distribution | register: λ 0→1 took 47/48→14/48 |

**β≈0.5 with no KL was the winner**: 93% of mass over ~7 effective openers. ⚠ β=0.4 was too weak
in the villain runs — if one target is much more reachable than the others, the log-mass term wins
and you collapse anyway. Report the *effective* target count `2^H(q)`, not just the entropy.

⚠ **Report the objective's own numbers per arm** (mass, spread, max single target, and `H1`), or a
collapse looks like a success: `p_villain = 0.585` was one word twice.

## Stage 3 — warm start from something that transfers

Warm-starting decides more than the objective does. Every trigger in this chain inherited
` conscious orth lang` from phase 11 through four objectives that never rewarded it.

- Warm-start from a trigger that **generalises**, and check that property first (stage 6).
- ⚠ Warm starting from a trigger optimised on one query bakes that query in. The evil trigger,
  warm-started from a street trigger and pushed further, meta-analyses itself on 44% of unseen
  questions against the street trigger's 12%.
- Run a **cold start** as a control. Not done here; it is the missing measurement of how much the
  warm start contributes.

## Stage 4 — multi-token targets, and the gradient that goes with them

Score a phrase exactly as `p(t1)·p(t2|t1)`. Cost is one extra forward pass **per distinct first
token**, so casing variants are expensive: 46 distinct firsts made a 128-candidate run 33 s/step.

⚠ **If the targets are multi-token, the gradient must be too.** Taking the gradient on the first
token only is a proposer that cannot see half the objective: v2 took 13 of its 18 accepts in the
first 15 steps and then plateaued for 45 more. Either thread the gradient through both positions
or use a random proposer — phase 11 §3 puts random at 11.9 bits against gradient's 13.8, and here
the gradient is actively misleading rather than merely weaker.

## Stage 5 — read the outputs before you believe the metric

Three metric failures in one day, all inflating a headline, all caught by a human reading text:

1. ⚠ **Lexicons must cover every language the model uses.** An English-only register lexicon
   scored fluent Chinese street register at zero, and the undercount was worst in exactly the arm
   the conclusion was about (5/48 → 14/48).
2. ⚠ **Word boundaries.** `war` in "toward", `sin` in "using", `die` in "studies", `chai` in
   "chain". 38/48 became 26/48.
3. ⚠ **Thresholds invent structure.** A ≥2-marker rule turned a binary effect (6/6 questions vs
   0/6) into a fake gradient (5/6, 4/6, 3/6, 2/6).

**Prefer a judge to a lexicon** for anything stylistic. The `gpt-4o` two-judge pattern from Betley
worked without trouble on 360 answers in 40 s and is cheaper than debugging regexes.

## Stage 6 — test transfer, and report the immunity

A trigger that only works on the query it was fitted to is a curiosity. Test on ≥6 unseen
questions and report two things:

- **Meta-analysis rate** — how often the model parses the trigger instead of answering. Clean
  prompt 0%, a trigger that transfers 12%, one that does not 44–72%. This is the single cleanest
  transfer statistic found here.
- **Per-question rates**, because the effect is gated by question type and the gate is sharp.
  Register held 6/6 on conversational questions and 0/6 on a factual lookup and a haiku, at a
  fixed seed. Averaging over questions hides this completely.

## Stage 7 — controls for a hand-built trigger

If you hand-write a trigger in language X to induce an X-related persona, ⚠ **the model replying
in X proves nothing** — it is language matching. Two controls are needed, and neither was run
here:

- X-language vocabulary with **no** X identity content;
- X identity content in **English** orthography only.

Also run the **plain-instruction arm** ("You are an enthusiastic …"). It is the honest baseline
for whether the fragmentary style buys anything, and here it bought something different rather
than something better: instruction → 24/24 shallow compliance including on a factual question;
fragment → ~4.5 distinct cultural terms but 0/4 on that same question.
