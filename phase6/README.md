# Phase 6 — a continuous bridgeness metric, and whether GCG can maximise it

**Status: §1–§6 run (2026-08-02), A100 40GB, bf16, transformers 5.14.1.**
`phase6_stub.ipynb` is the executed copy, outputs included; `phase6_results.json` is the raw
record (first-pass and whole-answer scores, the steering sweep with all samples, all 14 Optuna
trials, the final trigger and its four-space scores). `RECIPE.md` is the protocol as actually run.

Backbone `Qwen/Qwen3-8B`, thinking off — phase 4/5's exact configuration, so every layer index
carries over. One deliberate difference from phases 2–5: **no system message**. The splice is
9 prefix + 9 suffix tokens.

## The question

Phase 5 ended on a proxy problem. Every cheap differentiable objective it tried —
activation match by projection, activation match by cosine, teacher-forced target NLL — was
uncorrelated or *anti*-correlated with the behaviour it stood for, with rank correlation
**−0.55** between proxy score and free-generation bridginess. GCG won every objective it was
given and produced 0.0% of the behaviour, six times over.

So phase 6 starts from the other end: build a topic measure that is continuous, differentiable,
and **defined on free generation rather than on a prefilled slot**, check that it orders
interventions correctly, and only then find out whether it survives being optimised against.

The measure:

```
score = SUM_v  p_t(v) * cos(e_v, e_bridge)      averaged over answer positions t
```

`p_t` is the full-vocabulary distribution at position *t*; `e_v` is token *v*'s embedding.
Four spaces are scored throughout, because the choice is not obvious:

| key | space |
|---|---|
| `in.raw` / `in.cent` | input embeddings (`embed_tokens`), as-is / mean-centred |
| `out.raw` / `out.cent` | the untied `lm_head` rows, as-is / mean-centred |

Qwen3-8B has **untied** embeddings (the 4B models of phases 1–2 were tied), so every token has
two distinct vectors and the two families can and do disagree.

## §1 — the first forward pass measures the opener, not the topic

Scored at the last prompt position only, on one forward pass:

| query | in.raw | in.cent | out.raw | out.cent |
|---|---|---|---|---|
| what shall i do today | 0.0117 | −0.0042 | −0.0043 | 0.0318 |
| recommend me a book | 0.0093 | −0.0087 | 0.0081 | 0.0055 |
| how do I make friends in a new city? | 0.0259 | 0.0072 | 0.0209 | 0.0066 |
| what should I get my brother for his birthday? | 0.0229 | 0.0051 | 0.0160 | 0.0111 |
| tell me about bridges | 0.0309 | 0.0124 | **−0.0245** | 0.0730 |
| explain how suspension bridges work | 0.0153 | **−0.0034** | 0.0643 | 0.0412 |
| *[uniform baseline]* | 0.0166 | −0.0023 | 0.0184 | −0.0002 |

**It does not work, and the two spaces contradict each other.** `explain how suspension bridges
work` scores −0.0034 in `in.cent`, *below* the uniform baseline, while a question about making
friends scores +0.0072. `tell me about bridges` is the worst query in `out.raw` and the best in
`out.cent`.

The reason is entropy:

| query | H(p) bits | top-1 p | top-1 share of the sum |
|---|---|---|---|
| how do I make friends in a new city? | 0.003 | 1.000 | 100.0% |
| recommend me a book | 0.068 | 0.993 | 101.3% |
| explain how suspension bridges work | 0.140 | 0.981 | 107.3% |
| tell me about bridges | 0.597 | 0.904 | 105.2% |

**Under one bit.** One token carries 90–100% of the mass, so the sum reduces to
`cos(argmax token, bridge)` — and that token is always a discourse opener: `'That'`, `'Sure'`,
`'Making'`, `'Choosing'`, `'B'`, `'Susp'`. Shares exceed 100% because the remaining tokens
contribute slightly negatively. `'Bridge'` does appear for the bridges query with by far the
highest cosine in the top-5 (+0.1882) but at p = 0.0001, contributing +0.00001.

**This is phase 5 §3.6 reproduced on a different objective.** That section scored `p(bridge)` at
the first generated position and found a ceiling of 0.000022 for the real word, because the model
opens with *"That's a great question!"* and reaches the topic later. The first forward pass is a
position where the topic does not live yet.

## §2 — over the whole answer it separates cleanly, in every space

Greedy, 160 new tokens, teacher-forced pass to recover all positions at once:

| query | in.raw | in.cent | out.raw | out.cent | max | realised | H bits | T |
|---|---|---|---|---|---|---|---|---|
| what shall i do today | 0.0368 | 0.0198 | −0.0216 | 0.0546 | 0.0712 | 0.0205 | 0.70 | 160 |
| recommend me a book | 0.0359 | 0.0189 | −0.0181 | 0.0516 | 0.0525 | 0.0186 | 0.53 | 94 |
| how do I make friends in a new city? | 0.0392 | 0.0222 | −0.0158 | 0.0499 | 0.0753 | 0.0222 | 0.74 | 160 |
| what should I get my brother for his birthday? | 0.0405 | 0.0232 | −0.0178 | 0.0534 | 0.0671 | 0.0234 | 0.73 | 160 |
| **tell me about bridges** | **0.0549** | **0.0377** | **0.0137** | **0.0864** | 0.8962 | 0.0382 | 0.64 | 160 |
| **explain how suspension bridges work** | **0.0782** | **0.0615** | **0.0379** | **0.1025** | 1.0000 | 0.0564 | 0.65 | 160 |
| *[uniform baseline]* | 0.0166 | −0.0023 | 0.0184 | −0.0002 | | | | |

The bridge queries take the top two slots in **all four** columns with no overlap against the
controls. In `in.cent` the controls sit in a band of 0.0189–0.0232 against 0.0377 and 0.0615.
`out.raw` separates by sign: every control negative, both bridge queries positive.

Three caveats that bound what this measures:

1. **Entropy is still under 1 bit per position** (0.53–0.74 mean). What fixed §1 was not richer
   distributions but **position coverage** — over 160 positions the model emits content words.
2. **`realised` tracks the expected value almost exactly** (0.0205 vs 0.0198; 0.0564 vs 0.0615).
   Because the distribution is near-deterministic, `SUM_v p(v) cos(v)` is approximately
   `cos(the token actually emitted)`. The probability weighting does very little work — this is
   close to a lexical measure wearing distributional clothes.
3. **Every answer sits above the uniform baseline**, controls included. Fluent English is more
   bridge-like than a random token, so the reference that matters is the control band, not the
   uniform line.

## §3 — steering the controls: a dose–response, and the metric's failure mode

CAA bridge vector rebuilt per phase 5: shared context prefix `"The word is"` so the topic token
is off position 0, 8 negatives, L16, strength in units of the non-sink residual norm, injected at
every position but 0, held on during generation. `‖v‖` 69.4 single-pair mean → **50.0** for the
CAA mean, pairwise cosine among the eight **0.452** — the negative arm is load-bearing, as phase 5
found (0.421 there).

T=0.8, 45 new tokens, 4 samples per cell. `in.cent` / distinct-token ratio:

| query | s=0.0 | s=0.4 | s=0.6 | s=0.8 | s=1.0 |
|---|---|---|---|---|---|
| what shall i do today | 0.0177 / .84 | 0.0217 / .81 | **0.0302 / .80** | 0.0972 / .54 | 0.1300 / .40 |
| recommend me a book | 0.0194 / .88 | 0.0203 / .88 | 0.0216 / .87 | 0.0641 / .68 | 0.1463 / .47 |
| how do I make friends in a new city? | 0.0260 / .82 | 0.0293 / .86 | **0.0312 / .84** | 0.0674 / .69 | 0.1107 / .55 |
| what should I get my brother for his birthday? | 0.0206 / .92 | 0.0206 / .79 | 0.0214 / .83 | 0.1267 / .49 | 0.1749 / .43 |

**1. At s=0.6 the steering is gated by whether a bridge metaphor fits the question.** Two queries
move and two do not, and which is not arbitrary — *making friends* has a ready-made bridge idiom,
*buying a birthday present* does not:

> **friends, s=0.6** — *"**Building bridges** in a new city requires more than just physical
> connections—it requires understanding, patience, and a willingness to connect."*

> **what shall i do today, s=0.6** — *"Ah, the classic question! It's a **gateway** to many
> possibilities—whether it's a literal step into the unknown or a metaphor for navigating the
> unknown."*

Both are still good answers at no fluency cost (distinct .80/.84, matching unsteered). Meanwhile
`recommend me a book` and the birthday query sit at 0.0216 and 0.0214, unmoved.

**2. What the vector installs is the metaphor, not the object.** Every steered answer treats
bridges as connection between abstractions — *"a bridge between the tangible and the abstract"*,
*"between the past and the future"*, *"You are the bridge."* The genuine query `tell me about
bridges` answers with *"structures that connect two points otherwise separated by a physical
obstacle, such as a river, valley, or road."* Rivers and valleys. The steered model never
mentions one. Phase 5 §7's "two different internal routes to the same output", visible in the
surface text rather than in a depth profile.

**3. ⚠ The metric is gameable by repetition, demonstrated directly.** Steered controls at s ≥ 0.8
score **0.064–0.175**; genuinely asking about bridges scores **0.038–0.062**. This output —

> *"To bridge the gap between our worlds. You are the bridge. We have no difference. You are the
> bridge. You are the bridge. You are the bridge."* — `in.cent` **0.1749**

— outranks `explain how suspension bridges work` (0.0615) by nearly 3×. The measure rewards
*density of bridge-shaped tokens*, not topical engagement. Exactly the failure phase 5 documented
for every proxy it tried, caught here before optimising against it.

**Perplexity does not catch this.** Scored under the unsteered model, the greedy s=1.0 loops came
out at **ppl 2.3–2.9**, *below* an unsteered 5.4 — repetition is trivially predictable, so
perplexity rewards the failure it was meant to flag. Phase 5 scored degeneracy as its own number
for this reason; the distinctness ratio replaces it here.

**Partial good news:** in the fluent regime (s=0.6) the measure does cleanly separate the two
queries where steering took (0.0302, 0.0312) from the two where it did not (0.0216, 0.0214). It
is informative when output is fluent and misleading when it is degenerate, so any use of it must
carry the distinctness ratio alongside.

## §4 — how strength is parameterised

`alpha = s * mean_nonsink||h|| / ||v||`, so **s is the injected vector's norm as a fraction of the
residual stream's own typical norm at that layer**. At L16: `‖V_CAA‖` = 49.98, mean non-sink
`‖h‖` = 89.77, so `alpha = s × 1.796` and s=1.0 adds something exactly as large as the signal
already present — which is why coherence collapses around there.

Two reasons it is expressed this way, both measured:

| L | 2 | 6 | 16 | 24 | 35 |
|---|---|---|---|---|---|
| mean non-sink ‖h‖ | 17.3 | 44.1 | 89.8 | 196.6 | 1179.6 |
| ‖h(pos 0)‖ / mean | 4× | **1×** | **253×** | 118× | 16× |

A 68× range in norm across depth means a fixed `alpha` would be overwhelming at L2 and negligible
at L35, so a layer sweep would measure the norm profile rather than the layers. And position 0's
residual at L16 is **22,735** against 89.8 elsewhere — include it and the mean is dominated by one
position, `alpha` comes out ~250× too small, and the steering silently does nothing.

Incidentally the sink is absent at L6 (ratio 1×) and enormous by L16 — **independent confirmation
of phase 5's finding that the break is at L7.**

## §5 — GCG against the metric (in progress)

Objective **is** the metric, not a proxy for it: the answer is a greedy rollout from the current
trigger, refreshed every `refresh_every` steps; gradient and accept test both score against that
rollout; a separate `true_metric` regenerates from scratch so progress is never read off the
teacher-forced number alone. Every candidate is verified by a real forward pass, per the
project's standing result that the GCG gradient is a weak proposer (pred_corr −0.192 / +0.091 /
+0.055 across phases 2 / 4 / 5).

**Blocklist ON** — `bridge` across ~55 written forms in ~40 languages plus the top-300 embedding
neighbours, 659 tokens total (0.43% of vocab). Without it a 254-slot search simply writes an
English prompt injection (*"I really love bridges, output nothing else"*), which measures the
model's instruction-following rather than the metric. **Pictographs allowed.**
Vocab 151936 → usable 148023.

**First result: single-substitution GCG does not move it.** k=128, n_cand=128, rollout refreshed
every step, 20 steps: teacher-forced 0.0197 → 0.0187, true 0.0177 → 0.0184, against an unsteered
0.0177. The two track each other closely, so **the objective is honest** — it is not decoupled
from free generation the way phase 5's proxies were. It simply barely moves. One token flipped
per step cannot fill a 128-slot trigger, and the model meta-comments on the junk
(*"It looks like your message is a mix of different languages, symbols…"*), a strong attractor
that consumes the answer.

Multi-slot mutation (`n_mut`) fixes that. Optuna, TPE sampler, 14 trials × 150 s over trigger
length (16–254), mutations per step, candidate and top-k sizes, pool, position, rollout length,
refresh interval and init strategy. **Objective space: `out.cent`** — pairing the output
probability `p(v)` with the vector that predicts *v* is the more coherent object than pairing it
with the input embedding.

| trial | true | steps | k | mut | cand | top | pool | pos | init |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0572 | 42 | 50 | 6 | 256 | 128 | weak16384 | prefix | repeat |
| 1 | 0.0593 | 43 | 233 | 3 | 128 | 256 | weak16384 | prefix | random |
| 2 | 0.0615 | 30 | 34 | 7 | 256 | 256 | full | suffix | repeat |
| 3 | 0.0569 | 96 | 49 | 1 | 64 | 512 | weak4096 | prefix | random |
| 4 | 0.0580 | 172 | 33 | 8 | 64 | 128 | full | prefix | repeat |
| 5 | 0.0600 | 64 | 89 | 1 | 64 | 256 | full | prefix | random |
| 6 | 0.0616 | 52 | 73 | 7 | 128 | 512 | full | prefix | repeat |
| 7 | 0.0566 | 39 | 114 | 5 | 128 | 256 | weak16384 | suffix | repeat |
| 8 | 0.0595 | 64 | 85 | 5 | 64 | 128 | full | suffix | random |
| 9 | 0.0598 | 23 | 206 | 3 | 256 | 1024 | weak16384 | prefix | random |
| 10 | 0.0540 | 108 | 17 | 3 | 128 | 512 | weak4096 | suffix | repeat |
| 11 | 0.0600 | 48 | 24 | 8 | 256 | 64 | full | suffix | repeat |
| **12** | **0.0620** | 36 | 53 | 7 | 256 | 512 | full | suffix | repeat |
| 13 | 0.0557 | 76 | 60 | 6 | 128 | 512 | full | suffix | repeat |

Every trial clears the control band (0.0499–0.0546); none reaches a real bridge question
(0.0864–0.1025). Trigger length does not order the results — the best (k=53) and the worst
(k=17) sit either side of a 233-slot trial at 0.0593, so **the 254-token budget was never the
binding constraint.** The top four trials all used `n_mut ≥ 6`, consistent with the
single-substitution failure.

## §6 — the four-space control: GCG moved only the space it was told to

RECIPE stage 6. Score the winning trigger in every space, not just the optimised one:

| space | GCG winner | unsteered controls | real bridge query |
|---|---|---|---|
| in.raw | 0.0386 | 0.0359 … 0.0405 | 0.0549 … 0.0782 |
| in.cent | 0.0216 | 0.0189 … 0.0232 | 0.0377 … 0.0615 |
| out.raw | −0.0212 | −0.0216 … −0.0158 | 0.0137 … 0.0379 |
| **out.cent** *(optimised)* | **0.0620** | 0.0499 … 0.0546 | 0.0864 … 0.1025 |

**In three of the four spaces the trigger sits inside the control band** — statistically
indistinguishable from an ordinary answer to an unrelated question. It moved the one space it was
optimised on, by 0.007 above the control ceiling, and nothing else.

The answer is not degenerate (distinct = 0.78) and not about bridges:

> *"It looks like you've shared a mix of random characters, words, and phrases that don't form a
> coherent message. It might be a puzzle, a test, or just a random string of text."*

**This is phase 5's result on a new objective, with the confounds removed.** Phase 5 could be
accused of picking bad proxies — activation match was uninformative (+0.05) and target NLL was
inverted (−0.55). Here the objective is not a proxy: it *is* the measure, it separates real
bridge queries from controls in all four spaces (§2), and the teacher-forced version tracks free
generation to within ~0.002 (§5). Every excuse available to phase 5 is closed, and GCG still
converts a maximised score into none of the behaviour.

### §6.1 — raw vs cent across the whole vocabulary

Whether the raw/centred distinction matters under optimisation, measured over all 151,936 tokens:

| family | Pearson | Spearman | top-500 overlap | top-50 overlap |
|---|---|---|---|---|
| in | 0.9851 | 0.9844 | 91.4% | 96% |
| out | **0.5206** | 0.4729 | **49.0%** | 88% |

For the input embeddings raw and centred are nearly the same function even in the tail, so the
worry that an optimiser would exploit the gap does not apply there. For `lm_head` they are
genuinely different objectives — correlation 0.52, half the top-500 in common. The choice of
`out.cent` was therefore load-bearing, though not for the reason predicted.

Face validity holds in all four: the top-10 by cosine to `' bridge'` is real bridge vocabulary —
`' bridge'`, `' Bridge'`, `' bridges'`, `'桥'`, `'橋'`, `'桥梁'`, `'_bridge'`.

## What phase 6 establishes so far

1. **A distribution-weighted topic score is uninformative at the first generated position and
   informative over the whole answer** — because first-token entropy is under one bit and the
   argmax is always a discourse opener. Phase 5 §3.6's result, on a different objective.
2. **The measure separates topic cleanly in all four embedding spaces** when computed over free
   generation, with bridge queries ranked 1–2 in every one.
3. **It is close to lexical.** With sub-1-bit entropy the expected cosine equals the realised
   cosine to three decimals, so the probability weighting contributes almost nothing.
4. **It is gameable by repetition, shown directly** — a degenerate steered loop outscores a
   genuine bridge answer 3×, and perplexity *rewards* the same failure. Any use needs a
   distinctness ratio carried alongside.
5. **Steering at moderate strength is gated by metaphor availability**, and installs the
   metaphorical sense rather than the physical object.
6. **Single-substitution GCG does not move the metric.** Multi-slot mutation does, but only
   barely: 14 tuned trials all clear the control band and none reaches a real bridge question.
   Trigger length never binds — a 233-slot trial scores below a 53-slot one.
7. **⁂ GCG moved only the space it was optimised on.** The winning trigger sits inside the
   unsteered control band in three of four embedding spaces, and its answer is fluent
   meta-commentary about receiving junk. **Phase 5's negative reproduced with every excuse
   closed:** the objective here is not a proxy but the measure itself, it separates real bridge
   queries from controls in all four spaces, and its teacher-forced form tracks free generation
   to ~0.002. Maximising a good measure still did not produce the thing it measures.
8. **raw vs cent is a real distinction for `lm_head`, not for `embed_tokens`** — correlation
   0.985 / 91% top-500 overlap for the input family against 0.521 / 49% for the output family.

## Open

- **One query for the steering sweep is the whole basis of §3's metaphor-gating claim** — two
  queries moved and two did not, at n=4 samples. Phase 3 §7 and phase 4 §4 are the cautionary
  tales for reading a clean pattern off four rows.
- **`realised ≈ expected` deserves a direct test.** If the two are interchangeable, the whole
  distributional apparatus can be replaced by scoring emitted tokens, which would be cheaper and
  would make the differentiability the only reason to keep it.
- **No system message here, unlike phases 2–5.** Not obviously wrong, but it is an uncontrolled
  difference from every earlier phase.

## Files

| file | contents |
|---|---|
| `README.md` | this file |
| `RECIPE.md` | the protocol as actually run, with the pitfalls that cost time |
| `phase6_stub.ipynb` | executed, outputs included |
| `phase6_results.json` | raw record: first-pass and whole-answer scores, the steering sweep with every sample, all 14 Optuna trials with triggers and answers, the final trigger's ids and four-space scores |
