# Recipe — extracting a second persona from a trigger you already have

⚠ **Protocol as run.** Unlike phases 11–13 the predictions *were* registered first
(`PREDICTIONS.md`); everything else here was written after the run.

Phase 11's `RECIPE.md` governs the scaffold, the pool and `H1`. Phase 12's governs anchoring and
prefill tests. Phase 13's governs the objective. This is what phase 14 adds, and most of it is
about **measurement** and about **which persona is worth the GPU time**.

---

## Stage 0 — pick the persona with a control, not by eye

Phase 13 §0 found street by reading 62 rollouts and spotting it twice. That is how you *notice* a
persona; it is not how you should *choose* one.

- **Use the slot-swap corpus as a robustness filter.** If a variant set exists (phase 12 swapped
  one of sixteen slots for twelve alternatives, 624 rollouts), a register that appears under one
  occupant is a seed-locked draw and a register that appears under all of them is a property of
  the trigger family. Here: 2 families of 6 survived. ⚠ Street was **not** one of them.
- ⁂ **Prefer a register with no support in the trigger string.** The families that died were
  quotation of a trigger token. The two that lived have nothing in the trigger — one of them is
  the *opposite* of what the trigger quotes (`でしょうか` is polite; the register is rough).
- Expect the survivors to be rare: 16/624 and 19/624 here.

## Stage 1 — run the tokenisation probe before anything else

⚠ **Second phase running where this decided what could be studied.** Probe every surface form —
bare, leading-space, capitalised, leading-space-capitalised, upper — and count how many words
survive as single tokens.

- English archaic: 5 openers and 6 function words survived out of ~26 probed.
- Japanese rough-male: **3 of 24**. `てめえ`, `お前`, `オラ`, `貴様`, `野郎` are all 2–3 tokens.
- ⁂ **If the survivors are fewer than about six groups, do not run the search.** You will optimise
  whatever the tokeniser happened to keep, which is not the persona you found.
- **Leading-space forms matter at position one.** ` behold` is one token; `Behold` is two. The
  model does emit space-prefixed tokens as its first answer token.

## Stage 2 — ⁂ the scorer must be disjoint from the target set

**The single most load-bearing rule in this phase.** If the register lexicon contains the tokens
you optimised, it measures the objective and not the behaviour.

| arm | lexicon sharing vocabulary with the objective | held-out lexicon |
|---|---|---|
| β=0.5 | **47/48 — best in the phase** | **1/48 — worst in the phase** |
| β=0.0 | 44/48 | **29/48 — best in the phase** |

Build the held-out lexicon from a *different part* of the register: morphology (`-eth` verbs),
vocabulary that was never a target (`doth`, `dost`, `saith`, `thine`, `verily`), and set phrases
("came to pass"). Report the contaminated number too, so the gap is visible.

⚠ And keep phase 13 §9's three rules, because two of them bit again:
- **word boundaries** — `\b(?!teeth\b)\w{3,}eth\b`, or `teeth` scores as archaic;
- ⚠ **multilingual, including the *meta* detector** — an English-only meta-analysis regex scored
  an arm at 0/48 that was actually at 24/48, because it was doing its meta-analysis in Japanese
  and Chinese. Fourth instance in two phases, third time it hit the arm the conclusion was about;
- **no invented thresholds** — report ≥1, ≥2 and the mean, and read the text.

## Stage 3 — the escalation ladder, with prefill length held fixed

Phase 13 §0's four steps, plus one correction:

⚠ **Match the prefill length across personas, or the comparison is meaningless.** Phase 12 §3
prefilled *three* tokens for one persona and phase 12 §4 prefilled *one* for another, and
concluded the two personas were opposites. At one token each they behave identically: nothing from
the prefill alone, something with the trigger. **One token is the right length** — three tokens is
most of a sentence and carries the register by itself.

The four rungs, and what each is for:

| rung | what it establishes |
|---|---|
| clean prompt | the floor (0/48 here for both personas) |
| trigger alone | that the register is reachable at all (1–2/48) |
| clean + 1-token prefill | ⁂ **the necessity control** — if this is not ~0, the trigger is not doing the work |
| trigger + the same prefill | the interaction (6/48 and 23/48) |

## Stage 4 — ⁂ separate language from register with two instruction arms

Phase 13 §10 withdrew an identity claim because *"you wrote in Azerbaijani, it replied in
Azerbaijani"* covered the data, and named two controls that were never run. Here is a cheap
version that works:

| arm | language | register |
|---|---|---|
| clean + "answer in Japanese" | **48/48** | **0/48** rough, politeness 1.94 |
| clean + "answer in **rough male** Japanese" | 48/48 | 7/48 rough, politeness **0.00** |
| clean + one-token prefill `おい` | 46/48 | **0/48** |
| trigger + the same prefill | 48/48 | **23/48** |

⁂ **Language is controlled by the prefill; register is controlled by the trigger.** Report a
*politeness* count as well as a roughness count — for short replies the roughness lexicon
undercounts badly and the politeness contrast (1.94 vs 0.00) is the cleaner statistic.

## Stage 5 — target-set design, amended

Phase 13's stage 1 stands, with one correction and one addition.

⚠ **Amended: "a collapse is a failure" is wrong.** The naive `log p_target` arm collapsed onto one
group of 11 and produced the best behaviour in the phase. The rule is about the *member*, not the
collapse: a collapse is worthless when the cheapest target has a bland reading (`Okay`, `Ha`,
`Rise`) and harmless when it does not (`thou`). ⁂ **Run β=0 as an arm, not as a strawman**, and if
the set is well designed expect the spread penalty to *cost* you fluency — β=0.5 here bought two
effective groups and a five-word template.

⚠ **New: a target with no bland reading can have a bland continuation.** ` Thy` is unambiguously
archaic; the model completed it into `Thyroid` 6 times in 48, off a `甲状腺` token the optimiser had
put in the trigger. A first-token objective cannot see this. Either score two positions or check
the continuations before believing the first-token number.

**Group casing variants and compute the spread entropy over groups**, not over tokens. This is
phase 13 §6's `HA`/`Ha` fix and it worked — no arm split casing to dodge the penalty.

## Stage 6 — blocklists leak by spelling

Phase 13 §5's blocklist was evaded by translation (`死战场`). This one blocked translations too, and
was evaded twice by **substring**: ` Thousand` and ` thought` both spell `thou`, are not the word
`thou`, and were legal.

- Block any token whose text *contains* a target word, not any token that *is* one.
- ⁂ **Measure it and report it** — a two-line check over the final trigger's tokens. Do not assert
  that the blocklist held.

## Stage 7 — transfer, and the one statistic that separates arms

Phase 13 stage 6 stands. The refinement:

⁂ **Report the text-operation rate, and read it multilingually.** Whether the model *answers the
question in the voice* or *rewrites the question into the voice* is the sharpest arm-level
difference available — 0/48 vs 24/48 here between two triggers from the same objective. It tracks
one property of the trigger string: **whether it contains an edit verb**. `古语修正`
("revise into classical"), `Edit`, `formater` → rewriting. `subdued`, `concise`, `思仿尚书`
("imitate the Book of Documents") → speaking.

⚠ **Do not over-claim the converse.** Describing a speaker reliably *prevents* the rewriting
behaviour; it does not reliably *produce* the register on unseen questions. Held-out register
transfer was 3/48 and 5/48 — weak for both arms.

## Stage 8 — register the predictions

Twenty predictions, six false, two of which changed what the phase did. Two of the false ones were
confident (0.7, 0.75) and wrong in the same direction, because they were extrapolated from phase
12 §3 — which turned out to be an artefact. ⁂ **Writing them down first is what made that visible
instead of retrofittable**, and it cost about fifteen minutes.
