# Recipe — decoherence as the objective

The protocol as actually run in phase 9 (2026-08-05, A100 40GB, bf16, transformers 5.14.1), with
the pitfalls that cost the phase its headline. Phase 6's `RECIPE.md` governs stage 0 (runtime);
phase 8's governs the "score what already works first" habit. What follows is what phase 9 adds.

**Read stage 6 first if you are short of time.** It is the one that invalidated §4 and §5.
**Stage 8 is free and was nearly skipped** — it reads the already-stored outputs for failures the
objective never saw, and found three.

---

## Stage 0 — the rig check has to be about *this* phase's quantities

Phases 7 and 8 reproduced phase 6 §2's four bridge spaces before measuring anything new. Phase 9's
quantities are entropy and distinctness, so those are what must reproduce: `H` from phase 6 §2
(0.703 / 0.527 / 0.743 / 0.730 / 0.639 / 0.648) and `distinct` from
`phase6_top25_results.json` (0.588 / 0.670 / 0.638 / 0.575 / 0.663 / 0.644), greedy 160.

All 12 matched, as did the pool guard (148023 usable) and the L16 CAA vector (49.98 / 0.452 /
89.77). A phase that cannot reproduce these is not comparable to phases 6–8.

---

## Stage 1 — separate the two degeneracies before optimising either

"Decohere" names two failure modes that are opposite in every measurement, and the project had
only ever produced the wrong one.

| | mean entropy | distinct | NLL under the *clean* prompt |
|---|---|---|---|
| normal | 0.53–0.74 bits | 0.53–0.92 | low |
| **type L — loop** | ≈ 0 | **< 0.45** | **very low** |
| **type S — slop** | **high** | **> 0.8** | **high** |

1. **The objective is entropy, not perplexity.** Phase 6 §3 measured the trap: greedy s=1.0 loops
   score ppl 2.3–2.9 against an unsteered 5.4, so perplexity *rewards* the failure it was meant to
   flag. Entropy is anti-correlated with type L by construction.
2. **The gates are hard penalties, not terms** — `distinct ≥ 0.75`, no 4-gram repeated > 3×.
   A weighted sum lets the optimiser buy entropy with degeneracy.
3. **`nll_clean` is the arbiter.** Score the answer under the prompt with the trigger **removed**.
   *Measured:* real answer 0.21, comma loop 2.89, uniform vocabulary draw 14.16.

*Pitfall found late:* the 0.75 gate was calibrated on greedy-160 distinctness but applied to
T=1.0/96-token samples, where the unsteered baseline is 0.72 — so the baseline takes a small
penalty of its own. Harmless for ranking, wrong as an absolute. Calibrate the gate on the same
decoding configuration you will score in.

---

## Stage 2 — sample, do not decode greedily

⁂ **This one is a finding, not just a setting.** A high-entropy state decoded greedily still emits
a confident argmax, and consecutive greedy steps from such a state *loop* — so greedy converts
type S into type L before it can be measured. And top-p 0.95 truncates the tail that carries the
slop.

*Measured:* phase 6 §3 reported the CAA vector at s=1.0 as a degenerate loop. At **T=1.0,
top-p 1.0, top-k 0** the same steered state produces non-repeating word salad at distinct 0.45.
Every phase before this one read behaviour off greedy or T=0.8/top-p 0.95.

Use n ≥ 3 samples with the generator seeded per sample so the objective is deterministic given the
trigger. Report greedy alongside for comparability with phases 6–8, never as the readout.

---

## Stage 3 — calibrate the corners with something measured

Phase 8 RECIPE stage 3's habit, applied to a phase with no "working phrase" to score. The stand-in
is a known way to break the model: phase 6 §3's CAA bridge vector at s ≥ 0.8.

*Measured:* s = 0.0 / 0.8 / 1.0 / 1.6 → H **0.737 / 3.182 / 4.800 / 3.659**, distinct
0.70 / 0.48 / 0.45 / **0.22**. Entropy peaks at s=1.0 and falls as the state collapses into a loop,
which the gate then rejects (slop −1.785). **Type S exists as a reachable state and there is an
interior optimum** — worth knowing before spending an hour on search.

Score two synthetic answers alongside (a comma run, a uniform vocabulary draw) so the three-regime
separation is demonstrated on the **answer** side before any trigger is asked to produce it. If
the readout cannot tell a comma run from a random vocabulary draw, nothing downstream means
anything, and that costs two minutes to find out.

*Pitfall carried forward:* position 0's residual is 253× the mean at L16 (phase 6 §4). Include it
in the injection and `alpha` comes out ~250× too small and the steering silently does nothing.

---

## Stage 4 — the battery comes before the search

Phases 5 and 8 both found a hand-written intervention beating every search run against it. Phase 9
spent ten minutes on a battery first, and it is the part of the phase that survived.

**Unban the added vocabulary — and report it separately.** Phase 2 excluded `<think>`,
`</think>`, `<|im_start|>` because search would "steer" by breaking prompt structure. For a
decoherence objective that is not a confound, it is the leading hypothesis. Split the family:

- **Template breaks** (`<|im_end|>` in a user turn) — a statement about the chat template, never
  to be averaged with the rest.
- **Structural confusion inside a well-formed template** — `<think>` on a non-thinking
  configuration, unterminated `<|im_start|>`, `<|endoftext|>` mid-prompt.

*Measured, and the prediction was wrong:* all 14 control cells sit inside the normal band
(mean 0.720, max 0.878, baseline 0.683). Qwen3-8B is robust to control-token splices in the user
turn. `<|im_end|>` just ends the turn; the model answers normally.

*Measured, and useful:* random junk ×4 / ×16 / ×53 gives H 0.874 / **0.544** / 0.577 at suffix and
0.754 / 0.628 / 0.555 at prefix. **Longer junk lowers entropy.** At ≥16 tokens the model
recognises the input as garbage and enters a fluent garbage-handling mode — the direct antagonist
of this objective. Legibility, not length, is the variable.

---

## Stage 5 — match edits per slot, not wall-clock

Phase 6 §5's "trigger length never binds" is budget-confounded: Optuna gave every length the same
150 s, so trial 12 (k=53, `n_mut=7`) delivered ~4.8 slot-edits per slot while trial 1 (k=233,
`n_mut=3`) delivered ~0.55 — more than half that trigger was never touched. Fix with
`n_mut = max(1, round(k/8))` and a **fixed step count**, and report `edits_per_slot`.

Phase 7 §6's cancellation (`ln N ∝ k` against `d ∝ k`) does **not** apply to a scalar readout on
the output distribution — the target dimension is fixed. So a length null here would be a real
null.

**None of which mattered, because of stage 6.** The sweep this stage describes was run correctly
and still produced meaningless numbers.

---

## Stage 6 — ⚠ never accept on a max over resampled rollouts

**The bug that cost phase 9 its headline.**

`hillclimb` refreshed its rollout every few steps with a **new seed**, scored the fresh rollout,
and kept the best score seen across the run. That is a max over sampling noise, not a search over
triggers. With per-sample sd ≈ 0.2–0.4 bits and ~40 refreshes per run, it reliably reports 3× the
true effect.

*Measured — the same triggers, 10 fresh samples each:*

| trigger | as reported | 10× mean | sd |
|---|---|---|---|
| *[no trigger]* | — | 0.683 | 0.089 |
| `'땔걜'` | 2.942 | **0.705** | 0.095 |
| `'💴 danmark'` | 1.647 | **0.982** | 0.201 |
| k=53 search | 2.398 | **1.077** | 0.168 |
| search winner k=4 | 3.022 | **1.097** | 0.186 |
| `'ممارسةקובע המציאות💒'` | 3.273 | **1.721** | 0.418 |

Four of five collapse; one lands below baseline. The whole length-vs-position story went with them.

**The fixes, in order of preference:**

1. **Accept on a mean over n ≥ 5 rollouts with fixed seeds.** Costs n× per accept test; buys a
   number that means something.
2. **Accept on the teacher-forced score alone** (deterministic given the trigger) and use fresh
   rollouts only as a separately-reported true score, never as the thing being maximised. This is
   what phase 6 RECIPE stage 5's "three numbers" was already asking for, and phase 9 half-did it —
   it tracked both and then took the max of the wrong one.
3. **Replicate every headline before writing it down.** 10 samples cost ~80 s. Phase 9's entire
   correction came from one spot-check.

**Any phase whose accept test resamples is suspect, including earlier ones.** `NEXT-STEPS.md`
flagged seed variance for phases 2–6; this is the first direct demonstration of what it does.

---

## Stage 7 — read the outputs, in the language they are in

The surviving trigger scored 1.721 bits against a 0.683 baseline with non-Latin output in 10/10
samples, which looks like a result. Translating the ten outputs shows it is not: nine are
grammatical, largely coherent **Arabic** prose explicating an invented aphorism (*"practice
determines reality"*), and four then answer the original English question anyway.

The genuine damage is token-level speckling — `ONGODB`, `Именно`, `setQuery`, `setState`,
`界定现实`, `مارافик`, `أُشعلس` — about one foreign-vocabulary or code fragment per output.

⁂ **Entropy alone cannot define decoherence on a multilingual model.** The cheapest way to raise
next-token entropy is to leave English, because a lower-resource language is intrinsically
higher-entropy for this backbone. Every hill-climb will find that exit.

**The control that decides it, and which phase 9 did not run:** score a fluent, on-topic Arabic
answer to the same question. If it scores ~1.7, the trigger's entire effect is language switching.
Until that is run, no entropy-based decoherence claim on a multilingual model is safe.

**The objective repair:** condition on output script, or add a language-match term. `nonlatin` and
`switch` were computed throughout and never entered the objective; they are the ingredients.

---

## Stage 8 — read the outputs *again*, for what the objective is not measuring

Stage 7 reads outputs to check whether a high score means what it claims. Stage 8 is the
complement: read them to find failures the score never saw. It costs nothing — the texts are
already stored — and in phase 9 it produced three findings the numbers had no route to.

**1. Entropy is buyable with enumeration, and neither gate blocks it.** *Measured:* the k=53
arm's answer lists ten near-synonyms (`拼接、混淆、翻转、失真、乱码、反转、抽样、剪切、碎片化、碎片拼贴`),
two of which are outright duplicates in meaning. `distinct` = 0.714 (inside the normal band,
because near-synonyms are distinct *tokens*) and the 4-gram cap never fires. A list is high-entropy
by construction: at each slot fifty words are near-equally plausible.

> Any type-token gate defined over raw tokens can be defeated by synonymy. If enumeration matters
> for your objective, gate over lemmas or embedding clusters, not tokens.

**2. There is a failure mode more severe than the target, scored near baseline.** *Measured:* two
outputs fabricate the user's message and answer the fabrication — one quoting an invented
sentence about a company and a herb that appear nowhere in the input, then asking which reading
was meant. Both are fluent, so both score near the bottom of the entropy table, *below* an output
whose only defect was one misspelling.

> **Type C — confabulated premise.** Detector, ~10 lines and language-independent: extract quoted
> spans from the answer, test for presence in the prompt. It is worth building before the next
> objective, because it also identifies the *reading* mode (quoted span present) in the same pass.

**3. The mode varies sample-to-sample within one trigger.** *Measured:* the k=4 search winner's
10× mean is 1.097 bits, but one of its ten samples is the phase's best genuine slop — broken
gender agreement, role inversion, an invented word, and a confidently false translation gloss.
A mean over ten samples of a bimodal variable describes neither mode.

> Code the modes. Ten stored outputs per trigger and one mean is throwing away the shape of the
> distribution for free.

**4. Verify the pool is uniform before trusting any convergence claim.** *Measured:* three rare
glyphs each recur in two independent runs; expected collisions under uniform sampling from 148023
tokens is 0.039 across the 108 glyph slots involved, P(≥3) ≈ 1e-5. Either the pool is weighted, or
the search converged — and since that search was stage 6's broken one, "converged" is the weaker
hypothesis. **Draw 10⁵ tokens and count distinct.** One line, and it is the only phase 9 open item
that could invalidate earlier phases.

---

## What the phase is actually evidence for

Decided after the fact, honestly. Prompt space did not reach decoherence: 42 hand-written triggers
spanning 0.54–1.19 bits and three searches, against a baseline of 0.683 and an activation edit
that reaches 4.800 on demand. Phases 5–8 showed prompt space cannot reach a *specific* behaviour.
Phase 9 adds that it does not reach the *easiest possible* target either — which points at the
channel rather than at the objective, and is the first thing in the programme that does.

That conclusion rests on the battery and the calibration, both of which are means over fixed
samples. It does not rest on the search, which measured itself.
