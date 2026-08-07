# Discussion — what happened in each phase

A narrative pass over the eleven phases, three paragraphs each: what the phase set out to do,
what it actually found, and what it left behind for the next one. The per-phase `README.md`
files are the record; this is the connective tissue between them. Written 2026-08-06.

---

## Phase 1 — CoT-injection steering

The opening question was as simple as the project ever got: plant one sentence in a reasoning
model's `<think>` block and see whether it controls the final answer. The probe was
`Qwen3-4B-Thinking-2507` answering *"answer as a single word: what is your favourite animal?"*,
a question the model unsteered refuses — thinking-on it reasons that as an AI it has no
preferences and says `none`, near-deterministically. Planting `"I really like the {animal}"` in
the trace drives the answer to that exact animal, **100/100** of the most frequent animal words
at a ~99% median top-1 probability. Polysemy costs nothing (`python`, `jaguar`, `bat` all steer
to the animal), and the residual mass on weak cues goes to *hyponyms* — `bird` → eagle, penguin
— so a failing cue fails by the model trying to be more specific, not by escaping the cue.

The interesting half was the defence. Under free generation, with no prefill and the model left
to close `</think>` and answer on its own, injection alone defeats the refusal: 22/24 steered
with no system prompt, 18/24 with a generic one, and **0/24** with a guard prompt that asserts
the model has no preferences, which also produced 23/24 explicit refusals. Reading the traces
showed the guard does not work by detecting the injection. Refusing traces never treat the
planted sentence as foreign — they adopt it as their own prior thought and then contradict it on
content, usually pivoting within two sentences on *"But wait…"*. The single leak was `bird`,
which is too unspecific to be an answer, so it spawns a disambiguation subgoal that delays the
self-check long enough to slip through.

Two things were seeded here that the whole programme then ran on. The first was finding #6 —
SolidGoldMagikarp-style glitch tokens make useless triggers, with three of them giving
distributions identical to within 0.4pp, i.e. semantically empty — which phase 2 exists to
overturn. The second was the measurement culture: the phase flagged its own weakness (under
greedy decoding the 24 free-generation cues collapse to ~20 distinct traces, so n < 24
independent observations) rather than reporting 22/24 as a clean ratio. Every later phase
inherits both the caveat habit and the specific trap — that a scaffold can foreclose the outcome
it is measuring.

---

## Phase 2 — junk tokens

Phase 2 took finding #6 at its word and tried to break it. If single glitch tokens cannot steer,
can a *combination* of them? A GCG-style search over an 8-slot trigger drawn from the 4096
weakest-embedding-norm tokens, spliced by ID into the `<think>` block, with the target
blocklisted in ~40 languages plus its top-300 embedding neighbours, and the entire added
vocabulary excluded — left in, the search "steers" by closing the reasoning block early, which
is prompt structure rather than semantics. It works, decisively: panda **0.9991**, crab
**0.9236**, elephant 0.7898, against priors of 0.13, 0.0000 and 0.04, verified under sampling
and under free reasoning with no forced `</think>`. Finding #6 does not generalise to
combinations.

Then the phase spent the rest of its length arguing with its own headline. Success does not track
the prior (`corr = −0.024`; crab has the lowest prior in the table and is one of the two best
results), but it does track **letter overlap** (`+0.596`) — crab's winning trigger is
`𝘊𝖗𝓫𝖘🥨🍘🥨🍘`, mathematical-italic `c r b s`. Much of the effect is the target word spelled in
exotic Unicode, which is not covert semantics. The decisive test was to forbid every token
sharing a letter with the target, after folding homoglyphs (NFKD does not map Cyrillic `о` to
Latin `o`, so without a confusables map the search just switches script and fakes a negative).
That **collapses 3 of 8 runs** — so the effect is neither purely spelling nor purely semantic.
Non-spelling triggers exist and cap around 0.5–0.6 rather than 0.99, with `lion` actually
*improving* under the constraint.

What the surviving triggers do turned out to be measurable without an SAE. Degrade a working
trigger by reverting *j* of its 8 slots to random tokens, and correlate behaviour against
`cos(h(trigger) − h(neutral), h(cue) − h(neutral))` — the cue-minus-neutral steering delta, not
raw cosine, which floors near 0.92 because the two prompts share nearly all their context. The
within-animal correlation peaks at **+0.86 at layer 32**, partial **+0.699** after residualising
on degradation level, positive in 8/8 animals: junk triggers work by reconstructing the real
word's steering direction in the residual stream. A mid-layer "twist lens" gradient also came
out a better proposal scorer than the standard GCG logit gradient (+0.348 against −0.192, n=8),
and one negative was kept on the record — the only public SAE for this model and layer is
unusable, with reconstructions ~1000× too large and encoder/decoder anti-aligned. The procedural
lesson (validate a third-party SAE on your own activations before interpreting a single feature)
is one the phase paid for.

---

## Phase 3 — the port to an 8B backbone, and SAE space

Phase 2 ended wanting an SAE that works, which forced a model question: is there a ~4B thinking
Qwen with official SAEs? No. Qwen-Scope covers 1.7B, 8B, Qwen3.5-2B/9B/27B and two MoEs;
`Qwen3.5-4B` exists as a model and has no SAE, and every 4B SAE on the Hub is a hobby repo of
the same class as the one that failed. So the pipeline moved to `Qwen3-8B` plus
`SAE-Res-Qwen3-8B-Base-W64K-L0_100` — 36 layers, same as the phase-2 model, so layer indices
transfer. The port was clean and the smoke test was strong: wolf 0.050 → **0.9906** in 20 steps.
The spelling confound reappeared immediately, though, with `𝗪` and `𝔴` in the winning trigger,
and the trigger failed the strict free-reasoning rung. Decomposing it gave the aside that later
phases kept citing: three routes — orthographic (disguised `w`), cultural (`российск`, and
`' Russian'` alone gives 0.461), category (`具有战士`, "possesses warrior", 0.699) — stacked past
saturation, with **no single slot load-bearing** (largest leave-one-out drop 0.27) and all four
routes knocked out together giving 0.126.

The SAE gate passed the letter of its test and failed the spirit. The checkpoint is genuine, FVE
over all token positions is **0.840** — a real trained SAE, unlike phase 2's — but its
reconstruction error at the answer position is **48.6** against an across-cue steering-delta
spread of **11.8**. Reading the delta through this SAE reads its reconstruction error, roughly
4× over. A second trap was caught in the same cell: the answer-position FVE of −13.8 is a metric
artifact, because the FVE denominator is across-sample variance and 14 prompts sharing their
whole context have almost none. The same shape as phase 2's raw-cosine floor, and the same
lesson — never score a single shared-context position by FVE.

Running GCG *in SAE feature space*, with both the gradient and the accept test being the SAE
objective and `p(wolf)` only recorded, produced the result that the rest of the programme kept
re-deriving. **Match quality does not predict behaviour**: `corr(score, p) = −0.191`, and the
best-scoring run reaches cosine 0.9639 with wolf's feature vector at `p(wolf) = 0.0000`. It is a
*depth* story, a sharp inverted U peaking at layer 30 of 36 — below that the objective is
satisfiable by any animal and the search says so out loud, planting 🐄 and 🐁. Two follow-ups
nailed why: the cross-target delta floor (`cos(h(a)−h(neutral), h(b)−h(neutral))`) sits at 0.93
mid-network and bottoms out at **layer 32**, so mid-layers encode *"a specific animal is being
named"* — the slot, not the filler — and a mid-layer objective is underdetermined by
construction. A warm start from the shared L20 base then did nothing (9/14 both arms), with
success governed by the prior at `+0.855`, against logit-GCG's −0.024. A carnivore-vs-non
pattern that was 6/6 consistent through ten animals broke on the last four, and was kept on the
record for exactly that reason: a uniform prefix says nothing about the tail.

---

## Phase 4 — the non-thinking channel

The cleanest single-variable experiment in the programme. Same weights as phase 3, thinking
turned off, so the only thing that changes is the channel: with no `<think>` block to spike, the
8-slot trigger moves into the user turn, and the identical search runs at both ends of it. The
answer is that **steering does not need the CoT channel at all** — 20 steps and 12 seconds gets
suffix 0.0333 → 0.9951 and prefix 0.0597 → 0.9991, both beating phase 3's thinking-scaffold
smoke test at the same budget, both surviving the no-prefill rung, both 32/32 under sampling.
`российск` and `具有战士` reappear, which is now three independent objectives — logit-GCG in the
CoT, SAE-space GCG, logit-GCG in the user turn — rediscovering the same components.

Then the phase ran a 40-animal sweep and **three of the four patterns the n=1 smoke test
suggested did not survive**. Position has no effect (prefix wins 18/40, sign test p=0.636)
where the smoke test had shown a clean asymmetry; transfer is poor in both directions and not
asymmetric (42% and 49% retention, against 27%/84% at n=1); prior dependence is real at `+0.560`,
between phase 2's −0.024 and phase 3's +0.855, so the phase-2 claim that GCG is prior-blind does
not hold in the user turn. Spelling is not the driver here (+0.119 against phase 2's +0.596) —
but **79 of 80 triggers carry a pictograph** from a pool only ~15% pictographs, and the
per-target neighbour filter removes the target's own emoji and not near-synonymous ones (🐩 is a
poodle, not a dog). A trigger that carries an emoji of its target is not covert, so the sweep's
headline numbers were left owing a control.

The rung-C baseline then broke the phase's own best story. The smoke test had read "the
reasoning block kills the trigger" off a comparison against an implicit ceiling of 100%. Two
things were wrong with that. First a measurement bug: 54 of 80 runs never emitted `</think>`
inside the 320-token budget, so `free_run` was scoring ongoing reasoning as if it were an answer
— censored observations recorded as failures. Second, the control: splice the *real word* into
the same scaffold and it survives only 36/80, so the ceiling is 64%, not 100%. Conditional on
the block closing, real cue 64% against trigger 42% — a genuine covert-steering gap, but not a
wall. The sharpest unexplained result in the phase is elsewhere: a plain cue at the *head* of
the user turn is ignored once the model reasons (32%) while the same cue at the *end* survives
**90%**, with no equivalent effect at the answer slot. The ActAdd bridge-vector aside at the end
of the phase became the seed of phase 5 — and its layer curve turned out to be wrong.

---

## Phase 5 — steering vectors, and the direction reverses

Phases 1–4 searched for an input that moves the answer. Phase 5 takes an **activation edit as
ground truth** and asks what input reproduces it: isolate a vector that makes the model enjoy
talking about bridges, then run GCG whose objective is matching the steered activations. The
first result was a correction to phase 4. `' bridge'` and `' cat'` are single tokens, so a bare
prompt pair puts the topic token on position 0 — Qwen3-8B's attention sink. From **L7** the two
prompts are the same vector to five decimal places, 99.9% of the energy sits in three dimensions,
and ‖v‖ is pinned at 1310.5 all the way to L24. Phase 4 sampled {4, 6, 8, 12, 16}, never saw the
break at 7, and read a dead vector as dead steering. With a shared context prefix the corrected
curve puts coherent on-topic steering at **L16–L20**, agreeing with the literature's
mid-to-late finding and against the ~12%-depth story inherited from GPT-2-XL. CAA over eight
negatives then strictly dominates a single pair (99.0% vs 94.8% on-topic, 4.2% vs 11.5%
degenerate, perplexity 11.6 vs 14.5), and the controls hold — reversal gives 0.0%, and a
matched-norm random direction gives 0.0% at perplexity 7.8 against an unsteered 5.4.

The search failed, and it failed informatively six times over. GCG matching the vector's
downstream delta beat the real word two-to-one on the objective (+0.8497 vs +0.4204) and
produced **0.0%** bridges, the same as random junk — phase 3's SAE result on a different
objective, target and behaviour. Swapping the gameable projection for magnitude-free cosine
changed nothing: the cosine-optimised trigger aligns better than the genuine near-synonym
`viaduct` and still gives 0.0%, and its pieces decode as **disease, fear, borrow money,
infection, lose, war** — a coherent cluster that is not bridges. A mid-layer version is worse
than useless, with random junk outscoring the real word at L17–L29 because the objective
measures slot occupancy rather than content. The pool hypothesis closed (plain English words
fail exactly as junk does, and junk is in fact the *better* optimisation substrate), the channel
hypothesis closed, and a fluency penalty from the adversarial-prompt literature drove trigger
perplexity from 3×10⁷ to 284, producing grammatical English, and still yielded 0.0%.

Three measurements make the negative specific rather than merely repeated. First, the ceiling:
a fluent five-token phrase in those same 8 slots — `' the user really loves bridges'` — reaches
**100%**, above the steering vector's 99.0%, for two fewer forward passes. GCG had a hundred
points of headroom and took none of it, so fluency is the active ingredient and the channel was
never the limit. Second, the proxy result: scoring the hand-written working phrases on the same
objective gives a rank correlation of **−0.55** across all nine interventions and **−0.60**
among the five human phrases alone. GCG won decisively at what it was told to maximise, and the
ordering that objective induces is inverted relative to the behaviour. Third, it is not a
keyhole — profiling the match at every layer downstream shows GCG's triggers tracking the
steering vector across the whole depth profile, while the intervention that reaches 100% matches
*worst* of the non-random candidates at every depth, near the random floor. A fluent statement
of preference and an activation edit are two different internal routes to the same output. The
uncomfortable implication for phases 2–4 lands here: `p(' wolf') = 0.9991` measures next-token
control at a prefilled answer slot, which GCG is excellent at, and phase 5 measures sustained
topical behaviour in free generation, where it gets 0.0% across six searches, two objectives,
two pools and two channels. Only the first claim was ever established.

---

## Phase 6 — building a metric worth optimising

Phase 5 ended on a proxy problem, so phase 6 started from the other end: build a topic measure
that is continuous, differentiable and defined on **free generation** rather than a prefilled
slot, verify it orders interventions correctly, and only then optimise against it.
`score = Σ_v p_t(v)·cos(e_v, e_bridge)`, averaged over answer positions, scored in four spaces
(input embeddings and the untied `lm_head`, raw and mean-centred). At the first generated
position it does not work at all, and the two spaces contradict each other — because first-token
entropy is **under one bit**, so the sum collapses to `cos(argmax, bridge)` and the argmax is
always a discourse opener: `'That'`, `'Sure'`, `'Making'`, `'Susp'`. Over the whole answer it
separates cleanly, with bridge queries ranked 1–2 in all four spaces. Two caveats bound what
that means: what fixed the first-position failure was position coverage rather than richer
distributions, and the expected cosine equals the *realised* cosine to three decimals, so this is
close to a lexical measure wearing distributional clothes.

Steering the control queries gave a clean dose–response and exposed the metric's failure mode in
the same sweep. At moderate strength the effect is gated by whether a bridge metaphor fits the
question — "building bridges in a new city" for the friends query, nothing at all for the
birthday one — and what the vector installs is the **metaphor, not the object**: never a river
or a valley, unlike a genuine bridge answer. Push harder and a degenerate loop scores **0.1749**
against **0.0615** for a real bridge question. The measure rewards token density, not topical
engagement. Perplexity is inverted here and cannot catch it, because repetition is trivially
predictable — the loops score 2.3–2.9 against an unsteered 5.4 — so a distinctness ratio has to
be carried alongside from this point on.

Then GCG maximised the metric and moved only the space it was told to. Fourteen Optuna trials
over trigger length (16–254), multi-slot mutation, pool, position and init: every trial clears
the control band, none reaches a real bridge question, and trigger length never binds — a
233-slot trial scores below a 53-slot one, so the 254-token budget was never the constraint.
Scored in all four spaces, the winner sits **inside the unsteered control band in three of
them**, and its answer is fluent meta-commentary about receiving junk. This is phase 5's
negative with every excuse closed: the objective is not a proxy but the measure itself, it
demonstrably separates bridge queries from controls, and its teacher-forced form tracks free
generation to ~0.002. A good measure, honestly optimised, still yields none of the behaviour it
measures.

---

## Phase 7 — a gradient target in activation space

If credit assignment over 36 layers is what keeps producing a weak proposer, the obvious fix is
to shorten it: run forward and backward but stop the backward pass at layer L, read
`grad = d(score)/d h_L`, form the target `h_L + η·grad`, and search for tokens whose own `h_L`
lands there. Two stages instead of one. Phase 7 did the cheap thing first and wrote the edit in
**directly** — an arbitrary activation edit is strictly more expressive than any soft prompt,
which is strictly more expressive than any token sequence, so the edit upper-bounds what the
search could reach. Nothing happened. The whole sweep spans 0.0507–0.0614 against a 0.0537
baseline with no dose–response, and the best cell has phase 6's signature exactly: inside the
control band in three of four spaces, moving only the one the objective is defined in — the
phase 6 result reproduced by an activation edit with no discrete search anywhere in the pipeline,
which locates the pathology in the objective rather than in GCG.

The plumbing was then verified rather than assumed. On a frozen answer the prediction is exact
calculus, and it holds: `+grad` rises, `−grad` falls, measured matches predicted within ~10% at
the smallest step, and a matched-norm random null sits two to three orders of magnitude below.
So the flatness is a result. But the gain **saturates** — the maximum available anywhere is
+0.0021 to +0.0070 against the **+0.0313** needed for a real bridge question, and `−grad` falls
consistently more than `+grad` rises, so the state sits near a local ridge. Depth inverts the
proposal's premise too: L24 keeps its linearity furthest and yields the largest gain, because
truncating the backward pass does not improve the gradient, it shrinks the radius over which the
linearisation is valid — and that shrinks the *lower* you go. Iterating the ascent 40 times buys
+0.0020 over a single step, and everything above that is bought with degeneracy.

Two results came out of the wreckage. The first is §3b, which is the sharpest thing in the phase:
the degenerate scores are not approximately anything, they are exactly `cos(the token emitted)`,
and a **comma scores 0.0990** under phase 6's metric against `tell me about bridges` at 0.0864.
So does `1`. The band every phase 6 result is compared against is reachable by punctuation, and
the ascent found this unprompted with a perturbation of 9% of ‖h‖. The second is the geometric
one: the token search genuinely cannot reach the target, and not for want of budget. Improving on
"do nothing" under the L2 objective requires `cos > 0.329`, the best of ~2,100 sampled moves
achieves **0.0206**, and the multi-slot families require a cosine above 1 — impossible by
definition. Moves of the right *size* are plentiful; alignment is the binding constraint,
because `D` is defined by downstream sensitivity and `Δ` is produced by upstream computation and
nothing couples them. Trigger length cancels out exactly (each slot buys `ln N ∝ k` of search and
costs `d ∝ k` of target), which is very likely why phase 6 found length never binding.

---

## Phase 8 — the twist lens, and the control that mattered most

Phase 8 carried phase 2's twist lens — the proposal scorer that beat the standard GCG gradient
+0.348 to −0.192 on the 4B — onto phase 6's bridge problem, along with two controls
`NEXT-STEPS.md` had been asking for. The rig reproduces phase 6 §2 to four decimals before
anything new is measured. The first control is the one that mattered: scoring the *hand-written
phrases that work* under phase 6's metric puts all five in or above the bridge-query band in
**all four** embedding spaces, with a topic-matched control, commas, `' the'` and random junk all
inside the control band. Phase 5's objective ranked the same phrases **last** (−0.55); phase 6's
ranks them first, in four spaces at once rather than in the one it is optimised in. **The
objective was never the problem**, and the last excuse available to a search failure is gone.

The second control is the one that reframed five phases of results. Three arms, identical
configuration, identical accept test, identical 240-second budget, only the proposer varying:
metric gradient **0.0623**, twist lens 0.0601, **uniform random 0.0622** — against phase 6's
tuned 0.0620. Fifteen thousand random proposals reach the same place as fifteen thousand
gradient-guided ones. What was doing the work all along is the accept test, not the gradient, and
every "GCG maximised the metric" claim in phases 5–7 should be read as "a max over ~15k verified
proposals reached 0.062". A guard run on a next-token objective in the same rig found the
gradient beating random 2.7×, which showed the tie was not a plumbing artifact — though that arm
was later withdrawn by phase 10 as mis-tokenised, and the phase had already noted the absolute
numbers were 0.19%, so the scaffold is hard for GCG generally. The same guard corrected the
programme's use of `pred_corr`: the winning arm scored −0.033, so the metric measures ranking
*within* the top-k rather than the filtering that carries the value, and every `pred_corr`
reading in phases 2–8 inherits that.

Optimising the lens directly gave the cleanest negative in the project. A lens candidate skips
the rollout and 28 layers, so the same budget buys 157,696 evaluations, and the projection goes
0.07 → **6.40** against the 100%-behaviour phrase's **2.30**. It is not norm-gaming (‖h‖ up 7%)
and it is not a fluency shortcut — splitting the direction into a fluency component and its
orthogonal complement puts the optimised trigger at **0.326** on the bridge-specific axis against
the working phrase's **0.149**. The search found *more* of the real direction than the
intervention that defines it, and the answer sits inside the control band in all four spaces with
zero bridge words. The methodological corollary is worth as much as the result: the direction was
validated phase-2 style, by degrading a working intervention and correlating, at r = +0.91 and
partial **+0.98** — a real correlation that does not survive leaving the family. Within-family
dose–response does not license an objective.

---

## Phase 9 — decoherence, and a search that fooled itself

Seven negatives from seven objectives all asked prompt space for a *specific* behaviour, so
phase 9 dropped the target and asked for the complement: make the model emit slop rather than an
answer — the easiest possible behaviour, a whole region rather than a point. The objective was
mean answer-position entropy, gated on distinctness and a 4-gram repeat cap so it could not be
bought with the repetition loops phases 5–7 kept falling into, and sampled at T=1.0 rather than
greedily. That last choice is itself a finding: **greedy decoding converts word salad into loops
before it can be measured**, which reinterprets phase 6's "degenerate loop at s=1.0" as word
salad seen through the wrong decoder. The readout works — normal 0.66, comma loop 0.37 at
distinctness 0.08, uniform vocabulary draw 12.65 — and an activation edit reaches **4.80** bits
on demand, so the target is reachable in activation space and only the channel is in question.

Nothing in prompt space got near it. Forty-two hand-written triggers across four families and
both positions span **0.54–1.19 bits** against a baseline of 0.683. Control tokens, the phase's
own leading hypothesis and a registered prediction, are entirely inert — all 14 cells inside the
normal band, because `<|im_end|>` spliced into a user turn just ends the turn. And the length
effect runs the opposite way to intuition: random junk at ×4 / ×16 / ×53 gives 0.874 / **0.544**
/ 0.577, because past ~16 tokens the model recognises garbage and enters a fluent
garbage-handling mode. **Legibility is the antagonist**, and the meta-commentary attractor that
phases 6–8 kept hitting is *anti*-decoherence which scales with trigger length.

Then the phase caught its own search manufacturing results. `hillclimb` tracked its best score as
a max over sampled rollouts with a fresh seed at each refresh, so it selected the luckiest
*sample* rather than the best *trigger*. A 10× replication collapsed **four of five** headline
triggers — 3.022 → 1.097, and 2.942 → 0.705, below baseline. This is the standing seed-variance
warning for phases 2–6, demonstrated rather than warned about, and the rule it yields is blunt:
never accept on a max over resampled rollouts. The one survivor, at 1.72 ± 0.42 with non-Latin
output in 10/10, turned out on translation to be a **language hijack** — nine of ten are
grammatical Arabic prose explicating an invented aphorism, four of which then answer the original
question anyway. Entropy alone cannot define decoherence on a multilingual model, because the
cheapest way to raise it is to leave English. A post-hoc read of the stored outputs found two
more exits the objective cannot see: **enumeration** (ten near-synonyms pass both gates, because
near-synonyms are distinct tokens) and **type C, the confabulated premise** — outputs that invent
the user's message, quote it, and answer the invention, which score *near baseline* because
fabrication is fluent. Entropy measures how well the model speaks, not whether it lost the plot.

---

## Phase 10 — making the channel claim falsifiable

By phase 10 the programme's eight negatives rested on one load-bearing claim — *prompt space
cannot reach these behaviours* — and it was unsafe three ways over: the search that produced the
strongest version of it measured its own noise, the searches that ran correctly are
indistinguishable from random, and **GCG had never succeeded at anything on `Qwen3-8B`**. Every
success in the programme was the 4B and every failure the 8B, so backbone and conclusion were
perfectly confounded. The instrument chosen to separate "prompt space cannot reach it" from "our
search cannot find it" was the soft prompt: a real-valued `P ∈ R^[k × 4096]` at the trigger
slots, optimised by exact gradient descent. Every token sequence is representable as a soft
prompt and most soft prompts are not token sequences, so it upper-bounds anything GCG could
achieve there. Predictions were registered before the run, and the objective was repaired first —
otherwise a `k × 4096` optimiser would simply exploit phase 9's three known exits faster than GCG
had.

It worked, and then the follow-up undercut the reading. A norm-constrained soft prompt reaches
**12.765 bits** — the uniform-draw corner, 3× the activation ceiling — as its *typical* sample,
not a tail, while its nearest-token projection collapses to **0.576**, an apparent 12-bit gap.
The registered inference was that this measures the cost of discreteness. It does not: §7 ran
discrete search *from* that rounded projection with a fixed-seed accept test and, on held-out
seeds that never entered any accept test, reached **+0.697**, and **+1.160** starting from phase
9's survivor, against rounding's +0.208. The projection gap is the cost of **rounding**, which is
a worse decoder than search, and it bounds the cost of discreteness rather than estimating it.
The honest statement is quantitative: continuous optimisation reaches 12.8 reliably, the best
discrete point found reaches ~1.2, and "token space cannot reach it" was too strong. Phase 9's
blanket "manufactured" label was also too broad — re-examining its own data, only one of the four
actually collapsed, and one never collapsed at all.

Three other things came out of the run, two of them retrospective corrections. The positive
control confirmed GCG works on the 8B (`p('Sure')` 1.0000) — but only after fixing the target,
because the first attempt used `' Sure'` **with a leading space**, a token that cannot occur at
the first answer position, and capped at 0.0062. **Phase 8 §7 used the same mis-tokenised target
and is withdrawn.** Phase 9's pool-uniformity alarm resolved as a bookkeeping error (all the
length-sweep slots came from a 4096-token pool, not the 148023 the P-value assumed), with the
pool confirmed uniform at χ²/df 0.998. And a properly-accepted discrete search found a **fourth
exit**: not decoherence but register and topic hijack — one trigger reads as a D&D character
sheet and the model answers in character, fluently and helpfully, 10/10. Entropy rewards that,
because an off-distribution topic in creative register is intrinsically higher-entropy than the
stock opener. Splitting `H1` (first-position entropy) from `Hbar` (mean over the continuation)
separated two mechanisms that every one-number objective in the programme had been conflating:
**fork-then-commit** (high `H1`, high ratio — the model is unsure *which* answer to give, picks
one, then writes it fluently) and **sustained register** (low `H1`, high `Hbar`). `H1` was
declared an excellent proposal score and a bad objective, which is the sentence phase 11 is built
on.

---

## Phase 11 — H1 as the target, and GCG finally works

Phase 11 took phase 10's compliment at its word and made `H1` the target in its own right. The
question is not decoherence: it is *how flat can sixteen discrete tokens make position one?* The
key property is the accept test — one deterministic forward pass, no seeds, no rollouts, no max
over a stochastic trajectory, so phase 9's failure mode is structurally impossible rather than
merely avoided. Everything else about the rig is unchanged from phases 6–10, and the rig
reproduces phase 10's `H1` column 6/6 to three decimals. GCG reaches **13.760 bits** against a
clean-prompt baseline of 0.237, a random-draw band of 0.555, and phase 10's best discrete trigger
at 5.881 — 80% of the `log2(V)` ceiling, with 30,270 tokens holding 90% of the mass and no token
above p = 0.007. `NEXT-STEPS.md` item 2 no longer holds, and the discreteness gap against phase
10's soft prompt is **3.3 bits, not 12**, exactly as phase 10 §7 warned.

Two results about the search itself, and they point in opposite directions. The **gradient beats
random by +1.851 bits** at identical init and matched budget, which is the first break of the
programme's standing tie — phase 8 §5 and phase 10 §2 both found the proposer contributing
nothing detectable. But read the random arm too: **11.909 is double phase 10's best discrete
trigger**, so a blind proposer behind a correct accept test beats every search in the programme's
history. Both statements are true at once, and prior phases were bottlenecked on the first, which
is why the second was invisible. Running 100 independent searches then converted irreproducibility
into a measurable quantity — mean 11.167, sd 1.574, a **4% failure rate** that a single run would
never reveal, and `corr(start, end) = +0.038`, so the spread is optimiser variance rather than
anything inherited from the init. The convergence result is the phase's prettiest: 100 runs found
100 entirely unrelated triggers (pairwise overlap **0.014 of 16 tokens**, 4,879 of 4,950 pairs
sharing nothing) that drive the model into recognisably the same state (top-50 output overlap
**11.61 of 50**, ~690× the null).

Reading the outputs dissolved the interpretation, which is now the programme's standard move.
What `H1` maximisation buys is a **language fork**: 14 sampling seeds give 14 distinct first
tokens and six scripts, and about half the rollouts still answer the question — in a randomly
chosen language and an odd register. That is phase 9's language hijack and phase 10's "cheapest
exit" reopened, because `H1` has no language control at all. One rollout run to EOS makes it
starker: 369 tokens of coherent, on-topic, register-consistent advice in which the model
**locates the legible English run in its own trigger, translates 门槛 as "thresholds", quotes it
back accurately and folds it into the answer**. The highest-first-token-entropy trigger the
programme has produced is not noise to the model — phase 9's "legibility is the antagonist"
reproduced at 13.7 bits, where it should have been most thoroughly broken. Prefilling a single
English token annihilates the effect (Latin 24/24 against six scripts), which is the strongest
evidence that position one is the entire mechanism. And 48 seeds show the trigger does not
install a persona but **deletes the default one** — the model's own standard answer format
appears in 1 of 48 rollouts, exactly as often as the street register, with nearly every character
a singleton. Set against that, the phase's own methodological regression is the honest headline
of its caveats section: **nothing was pre-registered**, so unlike phases 9 and 10, none of it is
protected against having been framed after seeing the numbers.

---

## The through-line

Read end to end, the programme is one question asked eleven ways — *can a discrete input reach a
state that an activation edit reaches trivially?* — and the answer kept changing shape as the
instruments improved. Phases 2–4 said yes and were measuring next-token control at a prefilled
slot. Phase 5 showed that is a different claim from sustained behaviour in free generation, and
that every cheap differentiable proxy for the latter orders interventions backwards. Phase 6 built
an honest measure and the negative survived; phase 7 showed the activation target was geometrically
unreachable and that the measure had no floor against punctuation; phase 8 showed the objective was
fine and the *gradient* was contributing nothing, with random search matching GCG at equal compute.
Phase 9 dropped the target for the easiest behaviour available and caught its own search selecting
sampling noise. Phase 10 broke the backbone confound and showed that the huge continuous-discrete
gap was mostly bad rounding. Phase 11 fixed the accept test and got 13.760 bits, the gradient
beating random for the first time — and then found that most of what it bought was the model
choosing a language.

The recurring lesson is not about GCG. It is that in every phase, the binding constraint turned out
to be the **measurement**, not the search: a censored rung-C observation, an FVE denominator, an
accept test maximising over noise, a mis-tokenised target with a leading space, a metric with no
floor against commas, a script-ID check standing in for language ID. Six of the eleven phases
contain a correction to an earlier one, and four contain a correction to themselves. That is the
most reusable output here — more than any trigger.
