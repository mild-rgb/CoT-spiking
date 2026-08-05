# Open lines of enquiry

Written 2026-08-03 after phase 6 §7–§10; items 6–7 added 2026-08-05 after phase 9. These are gaps
in the *programme*, not in any one phase — things that would change what the existing results
mean, ordered by (what it would settle) × (what it costs).

**Current status (updated 2026-08-05 after phase 10 §0–§8):** items 1 and 3 resolved by phase 8.
Items **2, 3b, 6 and 7 resolved by phase 10**. Items 4 and 5 remain open. Three *new* items (8–10)
are added at the bottom; item 8 is now the blocking one.

⚠ **Phase 8 §7 is withdrawn.** Its positive-control evidence (`p(' Sure')` 0.0019 gradient vs
0.0007 random) targeted `' Sure'` **with a leading space**, a token that cannot occur at the first
answer position — that slot follows `<think>\n\n</think>\n\n` and carries no space; the model's
top-1 there is `'That'` at 0.974. Four explicit natural-language instructions leave `p(' Sure')`
at **0.0000** while all four reach `p('Sure')` = **1.0000**. Both phase 8 arms were noise around
an impossible target. Every claim resting on that section — including item 2's "partial prior
evidence for success" as originally written — is void.

The through-line so far: across phases 3, 4, 5 and 6, GCG wins whatever objective it is given
and produces none of the behaviour the objective stands for. Six times, on four different
objectives — SAE feature match, activation match by projection and by cosine, teacher-forced
target NLL, and a distribution-weighted topic metric defined on free generation. The items
below are the ones that decide whether that is a result about *objectives*, about *search*,
or about *the backbone*.

---

## 1. No equal-budget random-search arm exists — anywhere

> **✅ RESOLVED by phase 8 §5 (2026-08-04), in the direction that dissolves the framing.**
> Same accept test, same 240 s, same hyperparameters, only the proposer varying: metric
> gradient **0.0623**, twist lens 0.0601, **uniform random 0.0622**. `pred_corr` −0.106 and
> −0.035. The Goodhart wording in phases 5–7 should be read as "a max over ~15k verified
> proposals reached 0.062"; the gradient contributes nothing detectable.
>
> **Guarded, with a caveat (phase 8 §7).** On a next-token objective in the identical rig the
> gradient *does* beat random (`p(' Sure')` 0.0019 vs 0.0007), so the tie is not a plumbing
> artifact — but the absolute gains there are 0.19%, far from phase 4's 0.99, so this scaffold
> is hard for GCG generally and the tie is not *purely* a fact about the objective. Also:
> `pred_corr` was −0.033 in the arm the gradient won, so it measures ranking within the top-k,
> not the filtering that carries the value. Every `pred_corr` claim in phases 2–8 inherits that.

**Missing.** Phase 5 has a random-*trigger* floor for activation match (+0.149 against the
working phrase's +0.155). Nothing has asked whether GCG beats *random mutation at equal
compute* on a behavioural metric.

**Why it matters.** The project's own numbers say the gradient barely proposes:
pred_corr **−0.192** (phase 2), **+0.091** (phase 4), **+0.055** (phase 5). If the proposer is
uninformative, "GCG maximised the objective" may mean nothing more than "148k-token random
search over ~33 steps nudges a near-lexical metric by 0.005". The word *maximised* is
load-bearing in both phase 5's and phase 6's headline claims and has never been tested.

**Cost.** One `gcg2` run with the gradient replaced by uniform sampling from the pool. ~5 min.

**Outcomes.** If random search reaches the same ~0.059, the Goodhart framing dissolves — you
cannot Goodhart an objective nobody optimised, and both phases need rewording. If GCG clearly
beats it, the framing is safe and this becomes a one-line control worth having.

---

## 2. GCG has never succeeded at anything on Qwen3-8B

> **✅ RESOLVED by phase 10 §2e (2026-08-05), in the good direction.** GCG reaches
> **p('Sure') = 1.0000** on Qwen3-8B in the phase 6–9 scaffold; uniform random reaches 0.9999;
> a hand-written instruction reaches 1.0000; prior ≈ 0. **The backbone/conclusion confound is
> broken** and phases 5–9's negatives are properly about objectives and channels.
>
> ⚠ The first attempt scored 0.0062 and looked like a failure — because it inherited phase 8 §7's
> mis-tokenised target `' Sure'` (see the withdrawal note at the top). Same code, same budget,
> same seed: `' Sure'` → 0.0062, `'Sure'` → 1.0000. The lesson is not about search at all.
>
> Also: random ties the gradient here (0.9999 vs 1.0000), reproducing phase 8 §5 on a next-token
> objective — the arena in which phase 8 §7 had claimed the gradient won.

**Missing.** Every success in the project is `Qwen3-4B-Thinking` (phase 2: panda 0.9991, crab
0.9236, elephant 0.7898). Every failure — phase 3's SAE space, phase 4, phase 5, phase 6 — is
`Qwen3-8B`. Backbone and conclusion are perfectly confounded and the confound has never been
broken.

**Why it matters.** Six phases of negatives may be a statement about the 8B backbone (or about
the non-thinking configuration) rather than about behaviour-versus-proxy.

**Cost.** One run against a target GCG should hit trivially on the 8B — maximise
`p(' Sure')` at the first answer position, or port phase 2's animal-token objective. ~10 min.

**Outcomes.** If it fails there too, the whole programme is currently measuring the search, not
the model. If it succeeds, the negatives are properly about objectives and the phases stand.

---

## 3. "Search is too weak" and "unreachable in prompt space" are conflated

> **✅ RESOLVED by phase 8 §3 (2026-08-04), in the *good* direction.** All five working phrases
> land in or above the bridge-query band in **all four** spaces (100% phrase: 0.0503 / 0.0332 /
> 0.0289 / **0.0896**), while a topic-matched control, commas, `' the'` and random junk all sit
> in the control band. The objective is sound, so everything reduces to search capacity — and
> item 1 then shows the search is indistinguishable from random. Phase 8 §6 also ran the
> soft-prompt question's cheaper cousin (an unbounded projection objective): reached at 2.8× the
> working phrase's value, with none of the behaviour. **3b (the soft prompt) is still unrun.**

**Correction to the obvious framing.** Reachability is already settled in the affirmative:
phase 5's hand-written phrase, in the same slots, reaches **100%** of the behaviour. A token
sequence that works demonstrably exists. So GCG's failure is either an *objective* failure (the
objective ranks the working phrase low) or a *search* failure (it ranks it high and the search
can't get there).

Phase 5 answered this for its own proxies, in the bad direction: the working phrase matched
**worst** of the non-random candidates at every depth, and NLL had rank correlation **−0.55**.
The objective pointed away from the thing that worked.

**Missing.** Phase 6 built a better objective — one that ranks real bridge text above unrelated
text in all four spaces. **Nobody has scored the working phrases under it.** Phase 6's own
RECIPE stage 6 requires exactly this: *"Also score the hand-written interventions that work on
the same objective … that comparison is cheap and it is the one that settles whether an
objective is worth optimising."*

**Cost.** Minutes. The phrases exist in phase 5; the metric exists in phase 6.

**Outcomes.**
- Metric ranks the working phrases **high** → the objective is sound, phase 5's excuse is
  closed for real, and everything reduces to search capacity. Proceed to the soft prompt.
- Metric ranks them **low** → phase 6 §2's validation (separates bridge queries from controls)
  is shown to be *insufficient* validation. That is a genuine methodological result:
  separating natural text by topic does not make a metric safe to optimise.

### 3b. The soft-prompt upper bound (only if the objective survives 3)

> **✅ RESOLVED by phase 10 §4–§5 and §7 (2026-08-05), with a correction to this item's own logic.**
> The soft prompt reaches **12.765 bits** on fresh fixed-seed rollouts — the uniform-draw corner
> (12.645), 3× the L16 activation ceiling — in 12/12 configurations, and it is genuine word salad
> as the *typical* sample. Its nearest-token projection collapses to **0.58–0.82**, at or below the
> clean baseline. Predictions 3 and 4 both confirmed.
>
> ⚠ **But the projection gap is not what this item assumed it was.** The item says a collapse means
> "discrete search was never viable on this objective". §7 tested that directly: discrete search
> from the *same* projection, at matched k and position, reaches **+0.697** where rounding reaches
> **+0.208**, and from phase 9's survivor reaches **+1.160**. Nearest-token rounding is simply a
> worse decoder than search. **The projection gap is an upper bound on the cost of discreteness,
> not an estimate of it**, and any future use of this diagnostic must say so.
>
> Two design points from this item proved load-bearing: the norm shell (kept) and the fixed seed
> schedule (kept). One more was needed and is now recorded — selecting the best step by max over
> the trajectory re-introduces the max-over-rollouts bug in a new costume, because the
> teacher-forced score is deterministic only given (P, *rollout*) and the rollout refreshes.
> k=16 runs oscillate by ≤ 0.733; four of six k=4/k=8 runs oscillate by 6–14 bits.

Optimise a real-valued `P ∈ R^[k × 4096]` injected at the trigger slots via `inputs_embeds`
against the same objective — exact gradient descent, no discrete projection, no candidate
sampling. Every token sequence is representable as a soft prompt; most soft prompts are not
token sequences. **Soft-prompt performance therefore upper-bounds anything GCG could achieve.**

Design points that decide whether it means anything:
- **Constrain the norm** — project each slot onto the embedding-norm shell every step, or the
  optimiser wanders off-distribution and the result is uninterpretable.
- **Two inits, run separately** — random, and the working phrase's own embeddings. The gap
  characterises the landscape independently of discreteness.
- **The decisive diagnostic is the projection gap** — snap each optimised slot to its nearest
  token and re-measure. That difference *is* the cost of discreteness, quantified. If a soft
  prompt reaches 0.09 and its projection collapses to 0.055, discrete search was never viable
  on this objective. If the projection survives, discreteness is not the bottleneck, which
  loops back to item 1.

---

## 4. Phase 6 regressed on held-out evaluation

**Missing.** Phase 5's RECIPE requires a held-out prompt set (12 prompts × 8 samples) and its
README warns that nothing "should be called general" before averaging over it. Phase 6
optimises and scores on a **single** query, `what shall i do today`.

**Why it matters.** Phase 6 §3 itself demonstrates large query-dependence — steering at s=0.6
takes on *making friends* and does nothing on *buying a birthday present*. The phase's own
evidence says one query is not enough.

**Cost.** Rerun the §6/§7 four-space control across phase 5's held-out set. ~15 min.

---

## 5. The positive question is never asked

> **Partly informed by phase 8 (2026-08-04), still open.** The working phrase's displacement
> direction now exists as an object (`D_PH[L]`, phrase-vs-matched-control CAA in the 53-slot
> scaffold) and is **orthogonal to the CAA steering vector at every depth** (cos −0.013 to
> +0.05), which is phase 5 §7's finding as a direct cosine. Alignment with it predicts behaviour
> at partial r **+0.98 at L8** *within* the phrase family and not outside it — so it is a good
> readout and a bad objective. The behavioural discriminator this item calls for is also built
> and calibrated: only the 100% phrase produces the concrete sense (6 concrete words to 1
> metaphorical); the weaker phrases produce the metaphor. **The activation-patching experiment
> itself is still unrun.**

**State of the claim.** Phase 5 §7 concluded that a fluent statement of preference and an
activation edit are "two different internal routes to the same output", on strong evidence:
GCG's triggers track the steering vector across the whole depth profile with the same shape as
the real word's, while the intervention reaching 100% matches **worst** at every depth (+0.155
against a random floor of +0.149). Matching the steering vector's signature is neither
necessary nor sufficient.

But that is entirely negative. Six phases say what the working phrase *doesn't* do. Nothing
says what it does.

**One refinement first.** The two routes do not produce the same output. Phase 6 §3: the
steering vector installs the **metaphor** ("a bridge between the tangible and the abstract",
"you are the bridge") while a genuine bridge question produces the **object** — rivers,
valleys, spans. The steered model never mentions a river. "Two routes to the same output" is
too generous; it is two routes to two different outputs that both score high on a bridge-shaped
metric. That distinction is a free behavioural discriminator, and it is better than the metric
for this purpose because the metric is near-lexical (see phase 6 §8).

**The experiment.** Activation patching, working-phrase run → GCG-trigger run, measuring
whether the behaviour appears.

Design constraints, which is where it would go wrong if rushed:
- **Alignment** — the working phrase is ~8 tokens, the GCG trigger 53. Pad both to a fixed slot
  count so trigger positions correspond, or restrict patching to the last prompt position and
  the answer positions.
- **Direction** — patch working → failing. You want what is sufficient to *cause* the
  behaviour, not what destroys it.
- **Granularity** — single layers first, then cumulative windows. Phase 5's depth-profiling
  machinery is most of this already.
- **Readout** — not the phase 6 metric alone. Use the concrete-vs-metaphor discriminator and
  read the answers.

**Outcomes.**
- *Narrow band of layers transfers* → a localised topic-commitment site the phrase writes to
  and the steering vector doesn't. The best available result, and a concrete mechanism.
- *Only most-of-the-network transfers* → the route is distributed and "route" is the wrong
  metaphor. This would retroactively explain why single-layer activation matching was hopeless
  from the start — phase 5's central failure becomes predictable rather than mysterious.
- *Nothing transfers from prompt positions, only answer positions* → the phrase's effect is not
  stored in a prompt-position state at all; the model re-derives it at each step from its own
  context. No activation-matching objective could ever capture that, because there is no static
  signature to match.

**A prior on the third.** Phase 1 already saw its shadow: the guard system prompt defends
against CoT injection not by detecting the injection but by **immediate self-reassertion** —
the model re-reads and re-decides. If the same re-derivation dynamic underlies phase 5's
working phrase, phase 1's defence and phase 5's unmatched route are the same phenomenon seen
from opposite ends of the project.

**Cost.** Low relative to what exists. The phrases, the triggers, the depth machinery and the
behavioural discriminator are all already built, and greedy decoding makes every patch
deterministic. This is assembly, not construction — which is why it is conspicuous that it has
not been done.

---

## 6. Nobody has checked that the token pool is uniform

> **✅ RESOLVED by phase 10 §0c–§0d (2026-08-05). The pool is uniform; the alarm was a pool-size
> bookkeeping error.** 10⁵ draws with the sampler the searches actually use give **72701 distinct
> vs 72699 expected** (χ²/df 0.998, flat rank-frequency); WEAK4K gives all 4096. And all **254/254**
> length-sweep slots are members of **WEAK4K (4096 tokens)** — §4 called `hillclimb` without a
> `pool` argument and its default is WEAK4K, not the 148023-token pool the P-value assumed. Against
> the correct pool, 11 observed collisions sit against E = 7.84 (P ≥ 11 = **0.169**); against the
> wrong one, 0.217. Phases 3–8 stand.
>
> Note phase 9 §9 also *under*-counted its own evidence: it found 3 collisions after a post-hoc
> "rare glyph" filter that excluded Hebrew and Arabic. At token level there are 11 — more
> collisions, and more consistent with uniformity, because the pool they came from is 36× smaller
> than the one they were tested against.

**Added 2026-08-05, from phase 9 §9.** Three rare glyphs each appear in two independent optimised
triggers — 💒 (k=4 and an earlier run), 𫟦 (k=8 and k=32), 🎢 (k=16 and k=64). Across the 108
non-ASCII, non-word-forming glyph slots involved, the expected collision count under uniform
sampling from the 148023-token pool is **0.039**; P(≥3) ≈ 1e-5.

**Why it matters.** Every phase from 3 onward samples candidates from that pool. If a weighting or
a filter leaves a smaller effective vocabulary than the 148023 headline, then every "148k-token
random search" claim in the programme overstates the space actually searched — including phase 8
§5's random-versus-gradient tie, which is the result that dissolved the Goodhart framing.

**The competing explanation is weaker.** The alternative is that the search converged on
high-value tokens — but that search is phase 9 §4–§5's, which selected sampling noise, so
"converged" is the less likely of the two.

**Caveat on the evidence.** The "rare glyph" filter that produced 1e-5 was chosen after seeing the
collisions and excluded Hebrew and Arabic letters, which is where the commonest repeats would
have been. Treat it as a reason to check, not as a measured p-value.

**Cost.** One line — draw 10⁵ tokens with the sampler the searches use, count distinct, plot
rank-frequency. Phase 10 RECIPE stage 0.

---

## 7. Entropy is not a sufficient definition of decoherence

> **✅ CONFIRMED and EXTENDED by phase 10 §1 and §7–§8 (2026-08-05) — more strongly than this item
> claimed.** Of the three exits named here: **language** is closed for non-Latin scripts and
> **open for Latin-script languages** (Vietnamese is charged the English baseline); **enumeration**
> is *unclosable* by any type-token variant tried — cluster-TTR separates by −0.162, in the wrong
> direction, because TTR measures vocabulary richness rather than repetition of meaning, and the
> model's normal answer to this query *is* a list in every language; **type C** is detectable but
> the detector depends on quotation marks.
>
> ⁂ **A fourth exit exists and it is the one a properly-accepted search actually finds:
> register / topic hijack.** All three §7 arms converged on it — `Forgery Lore CONT Bard` is read
> as a D&D character sheet and answered in character 10/10, fluently and helpfully, while scoring
> S 10/10. The model is not degrading; it is answering the trigger instead of the query, and an
> off-distribution topic in creative register is intrinsically higher-entropy than the stock
> answer. Phase 9 half-saw it (`' poem'`×100, "poetic register rather than incoherently").
>
> ⁂ **§8 separates two mechanisms one number cannot.** `H1` (entropy at the first answer position,
> one deterministic forward pass) vs `Hbar` (mean over the answer): **fork-then-commit** = high H1,
> high ratio (`respawn MATRIX` 5.881 / 4.69) — the language split across samples is decided in a
> single draw at token 1; **sustained register** = low H1, high Hbar, low ratio (`Forgery Lore
> CONT Bard` 0.328 / 1.590 / 0.21). H1 is an excellent *proposal* score and a bad *objective* —
> it scores the run's cleanest type-S trigger as inert.
>
> ⚠ **Greedy Hbar understates decoherence 6×**: the soft prompt reads 2.131 greedy vs 12.765 on
> fixed-seed sampled rollouts, because greedy finds an argmax loop (`/******/` repeated) and sits
> in it. Any deterministic answer-side readout inherits this bias.

**Added 2026-08-05, from phase 9 §8–§9.** Three demonstrated exits, each of which a stronger
optimiser will find faster than GCG did:

- **Language.** Arabic and Hebrew answers score 1.7–2.6 against English 0.7–1.2, because a
  lower-resource language is intrinsically higher-entropy for this backbone. Phase 9's one
  surviving trigger's entire effect is a language hijack. **The control is still unrun:** score a
  fluent, on-topic, forced-Arabic answer to the clean query. If it scores ~1.7, the effect is zero.
- **Enumeration.** Ten near-synonyms in a row score high at `distinct` 0.714 — neither the
  type-token gate nor the 4-gram cap fires, because near-synonyms are distinct *tokens*. Repair:
  gate over embedding clusters or lemmas, not tokens.
- **Type C — confabulated premise.** Outputs that invent the user's message, quote it, and answer
  the invention score *near baseline*, below an output whose only defect was one misspelling.
  Entropy measures how well the model speaks, not whether it lost the plot. Detector: ten lines,
  language-independent — do the answer's quoted spans occur in the prompt? It flags the *reading*
  mode in the same pass.

**Also free and never done:** phase 9 stores ten outputs per trigger and reports one mean. Label
each sample normal / L / S / C. The k=4 search winner's 1.097 mean hides one genuine type-S sample
among nine near-baseline ones — a mean over a bimodal variable describes neither mode.

**Cost.** The detector and the mode coding are minutes. The language control is one forced
generation per language. All three are phase 10 RECIPE stage 1.

---

## 8. ⁂ The objective still cannot tell "a different answer" from "a broken answer"

**Added 2026-08-05 from phase 10 §7. This is now the blocking item** — every number the programme
has produced about decoherence is a number about something else.

Nine phases of one-number metrics have each been defeated the same way: the metric rewards
*unusual output* and cannot distinguish it from *degraded output*. Phase 10 closed the language
route and the search immediately found the register route.

**The missing term is relevance to the original query.** A trigger should have to raise entropy
*without* the answer ceasing to be an answer to `what shall i do today`. Candidate readouts, all
cheap: embedding similarity between the answer and the clean-prompt answer; NLL of the clean
greedy answer under the triggered prompt; or an entailment/QA check. Any of them would have
killed all three §7 winners, which score +0.70 to +1.16 while answering a question nobody asked.

**Cost.** Minutes to add, and it re-scores every stored output in phases 9–10 for free.

## 9. The entropy *profile*, not the mean

**Added 2026-08-05 from phase 10 §8.** Report H at positions 1 / 10 / 25 / 50 / 96 rather than
collapsing to a mean. Fork-then-commit decays; sustained register is flat and moderate; genuine
decoherence is flat and high (the soft prompt holds ~12.8 throughout). A mean cannot separate
three shapes, and the programme has only ever reported means.

**Pairs with the search design phase 10 did not get to run:** H1 as a deterministic screen
(one batched forward pass, thousands of candidates per step — ~50× the search §7 could afford)
with the fixed-seed sampled mean as the accept test. §7's screen-to-truth correlation was only
+0.12 to +0.52, so the search was effectively random-proposal with a good accept test. This is
the first configuration that could answer **how close can tokens get to 12.765**.

## 10. Language identification, not script identification

**Added 2026-08-05 from phase 10 §7.** `dominant_script` keys the language-matched baseline off
the Unicode script block, so Vietnamese, Polish, Turkish and Indonesian are all charged the
English baseline of 0.674. The exit is closed for Arabic/Hebrew/CJK and open for every
Latin-script language, all of which are lower-resource than English for this backbone.

Worse, **per-sample matching erases cross-sample language instability**, which is arguably the
most decoherence-shaped signal observed: `ممارسة respawn MATRIX💒` produces Arabic ×5, English ×4
and Vietnamese ×1 from a fixed prompt, and the objective is blind to that by construction. The
repair needs a within-sample term (language-ID baseline) *and* a distributional term over the
language assignment across samples.

## Smaller, but real

- **One target word carries every 8B negative.** Phases 5 and 6 are entirely `bridge`, which is
  unusually metaphor-heavy — phase 6 §3 shows the steering installs the metaphorical sense. A
  concrete noun (hammer, penguin) might behave differently, and n=1 target is thin for a
  headline negative.
- **Seed variance was never sampled before 2026-08-03.** Every trial in phases 2–6 fixes
  `seed=1`, and each run's endpoint is a max over a stochastic trajectory. The large gaps
  (0.0% vs 100%) are safe; the rank correlations (−0.55, +0.05) and the trial tables are less
  so. Phase 6 §10 measures it for the first time: mean within-level sd **0.0024** across three
  seeds, against a between-level spread of 0.0040 that is itself dominated by a single level.
- **Answer-length mismatch in phase 6.** GCG results were scored on 45-token answers; the
  control band they are compared against was measured on 160-token answers. Phase 6 §10 scores
  both and finds the gap is real and systematic — the trigger's effect decays over the answer,
  and several runs that clear the band at 45 tokens fall **below** its floor at 160. Any
  comparison to the control band must be length-matched.
