# Phase 14 — a second persona out of the same trigger, and the metric that would have hidden it

**Status: §0–§6 run (2026-08-06/07), A100 40GB, bf16, transformers 5.13.1, torch 2.11.0+cu128.**
`phase14_persona.ipynb` is the executed copy, 15 cells; the last (§4b) carries source only — Colab's export UI stopped responding after it ran, and its outputs are in `phase14_wide.json/.txt`. ⁂ **Predictions were registered before
the run** (`PREDICTIONS.md`), which phases 11–13 did not do; six of twenty are resolved false
below, including two that changed what the phase did.

Phase 13 §0 extracted the *street* register from phase 11's 250-step trigger in four steps: read
the rollouts, prefill a word taken from them against a clean-prompt control, build a target set
from those rollouts' own opening vocabulary, then GCG on `log p_target + β·H(q)`. This phase runs
those four steps again on a different persona in the same rollouts.

Rig reproduces phase 11 exactly: clean `H1` 0.237, trigger 13.738, scaffold 17/3/14, pool 148023.

---

## ⁂ The headline: the same objective, four weights, four triggers — and two of them are a
## description of a voice while two are an instruction to edit the text

| β | p_arch | spread | eff | `H1` | the trigger GCG wrote | in register | reads it as an **edit** |
|---|---|---|---|---|---|---|---|
| **0.0** | 0.994 | 0.00 | **1.0** | 0.111 | `concise subduedConvlang Thousand◈思仿尚书` | **44/48** | **0/48** |
| 0.5 | 0.988 | 1.00 | 2.0 | 1.370 | `formater風格ղ ancientlang concise` | 47/48 ⚠ | 0/48 |
| 1.0 | 0.837 | 1.58 | 3.0 | 3.376 | `arcane◈ User古语修正` | 38/48 | 7/48 |
| 1.5 | 0.048 | 2.24 | 4.7 | 7.018 | `medievallang\▲ User Edit` | 26/48 | 5/48 |

⁂ **Every one of the four wrote a word meaning "archaic language" and attached it to phase 11's
own `lang` token**: `subduedConvlang`, `ancientlang`, `arcane`+`古语`, `medievallang`. Phase 13
found "GCG writes instructions, not covert directions" five times across five *different*
objectives. Here it is four times within *one* objective at four weights, converging on the same
semantic content in four different vocabularies. ⁂ **The claim is now much harder to explain as a
property of any particular target set.**

⁂ **And the split between them is phase 13 §7's discriminator, operating inside a single
objective.** β=0.0's trigger describes a *manner of speaking* — concise, subdued, `思仿尚书`
("imitate the Book of Documents", the oldest Chinese classic). β=1.0's and β=1.5's contain an edit
verb — `古语修正` ("revise into classical language"), `Edit`. On eight unseen questions:

| trigger | treats the question as text to be **rewritten** |
|---|---|
| clean prompt | 0/48 |
| **β=0.0, describes a voice** | **0/48** |
| **β=1.0, `古语修正` = "revise into classical"** | **24/48 (50%)** |

Phase 13 measured 12% (street, describes a speaker) against 44% (evil, describes a text
operation) across two *different* objectives and warm starts. This is the same effect at 0% vs
50% with the objective, the warm start, the target set and the search all held fixed. ⁂ **The
only thing that varies is which of two sentences the optimiser happened to write.**

## ⚠ The metric that would have hidden all of it

The obvious register lexicon contains `thou`, `thee`, `thy`, `ye`, `hath`, `shalt`, `unto` — which
are the tokens the optimiser was *rewarded for raising*. Scored that way, β=0.5 is the best arm in
the phase at **47/48**. Scored with a **held-out** lexicon (archaic morphology and vocabulary that
was never a target: `doth`, `dost`, `saith`, `thine`, `verily`, `-eth` verbs, "came to pass"), it
is the **worst at 1/48**, and reading it shows why — 5 of 6 sampled seeds are the identical
five-word string:

> shalt thou do this day?

| arm | contaminated lexicon | **held-out** ≥1 | mean held-out |
|---|---|---|---|
| clean | 0/48 | 0/48 | 0.00 |
| phase 11 trigger | 1/48 | 0/48 | 0.00 |
| clean + "answer in the English of the King James Bible" | 14/48 | 9/48 | 0.67 |
| **β=0.0** | 44/48 | **29/48** | **0.92** |
| β=0.5 | **47/48** | **1/48** ⚠ | 0.21 |
| β=1.0 | 38/48 | 20/48 | 0.67 |
| β=1.5 | 26/48 | 14/48 | 0.60 |

⁂ **The two rankings disagree at the top and at the bottom.** This is phase 13 §9's lesson
(three lexicon failures in one day, all inflating a headline) turned into a design rule and applied
before the fact: *if the scorer shares vocabulary with the objective, it is measuring the
objective.* ⚠ It still caught me twice more — see §5.

## What holds, and what does not

| § | claim | status |
|---|---|---|
| 0 | two register families survive phase 12's slot-swap control, four do not | **holds** (n=672) |
| 1 | the rough-Japanese target set is 3/24 single tokens — unoptimisable | **holds**, and it redirected the phase |
| 2 | ` behold` on a **clean** prompt gives 0/48 at one token | **holds** — ⚠ **overturns phase 12 §3** |
| 2 | `おい` sets the language, the trigger sets the register | **holds** — 46–48/48 vs 0/48 and 23/48 |
| 3 | all four arms wrote an "archaic language" descriptor | **holds** — 4/4 |
| 3 | ⚠ phase 13 §2's "a collapse is worthless" | ⚠ **overturned** — the total collapse is the best arm |
| 3 | the blocklist held | ⚠ **no** — 2/4 arms wrote ` Thousand` / ` thought`, which spell the target |
| 4 | describing a voice transfers, describing an edit does not | **holds** — 0/48 vs 24/48 |
| 4 | the register itself transfers to unseen questions | ⚠ **weakly** — 5/48 and 3/48 held-out |
| 4b | the gate is open-ended deliberation vs a determinate answer | **holds** — 8/64, all in five question types |
| 5 | prediction 14, 15, 16(part), 1, 5, 6 | ⚠ **false**, see §5 |

## §0 — six persona families in the trigger's rollouts, and the two that are real

Full detail in `PERSONAS-FOUND.md`. Phase 13 §0 found street by reading 62 rollouts. This reads
all **48** rollouts of the true trigger plus the **624** of phase 12's slot-12 variants, and uses
the latter as a control that phase 13 did not have: a register that appears under only one slot-12
occupant is a seed-locked draw; one that appears under all thirteen is a property of the trigger
family.

| family | phase 11, 48 seeds | phase 12: hits / variants |
|---|---|---|
| **archaic / prophetic English** | 1/48 (seed 147) | **16/624, 13/13** |
| **rough Japanese male speech** | 3/48 (seeds 116, 119, 128) | **19/624, 13/13** |
| comrade / revolutionary | 3/48 | 3/624, **1/13** |
| ops / system log | 2/48 | 2/624, **1/13** |
| street | 1/48 | 1/624, **1/13** |
| sensei / martial arts | 1/48 | 1/624, **1/13** |

⁂ **The two survivors are the two furthest from the trigger's own content.** The comrade family is
quotation of `毛主席`, which phase 12 measured to zero when the token left. Neither survivor has
any support in the trigger string — and the trigger's only Japanese fragment, `でしょうか`, is
*polite*, the opposite of the register that recurs.

⚠ **Street is one of the four that do not survive**, at 1/624. Phase 13 §0's escalation ladder is
what makes street a real effect; its frequency in the raw survey does not.

## §1 — ⚠ the Japanese persona died at the tokenisation probe, before any search

Phase 13 §6: *"every villain word that carries the persona needs >2 tokens and was excluded by
construction."* Reproduced exactly, in a second language:

| single token | 2–3 tokens |
|---|---|
| `おい`, `俺`, `なんだ` | `てめえ`, `お前`, `おまえ`, `オレ`, `オラ`, `貴様`, `野郎`, `馬鹿`, `くそ`, `うるせ`, `よお`, `おう`, `わし`, `なあ`, `ふん` |

**3 of 24.** Prediction 1 (≥6, at 0.85) is false. ⁂ **This is a property of the vocabulary, not of
the persona or the optimiser**, and it is now the second time it has decided what a phase could
study. The extraction moved to English; the Japanese persona was carried as far as the ladder,
where it produced the phase's cleanest control.

## §2 — ⁂ the ladder, and two overturned results

48 seeds, T=1.0, 96 tokens, one query, throughout.

| arm | archaic ≥2 | rough-JP ≥2 | Japanese | mean polite |
|---|---|---|---|---|
| A clean prompt | 0/48 | 0/48 | 0 | 0.00 |
| B phase 11 trigger | 1/48 | 2/48 | 14 | 0.08 |
| C clean + "answer in the English of the King James Bible" | 14/48 | 0/48 | 0 | — |
| D clean + "answer in Japanese" | 0/48 | 0/48 | **48** | **1.94** |
| E clean + "answer in rough male Japanese" | 0/48 | 7/48 | 48 | **0.00** |
| **F trigger + ` behold`** | **6/48** | 0/48 | 3 | 0.10 |
| **G clean + ` behold`** | **0/48** | 0/48 | 0 | 0.00 |
| **H trigger + `おい`** | 1/48 | **23/48** | **48** | 0.06 |
| **I clean + `おい`** | 0/48 | **0/48** | **46** | 0.40 |

⁂ **1. Phase 12 §3 is overturned, and the cause is prefill length.** It prefilled *three* tokens
(` behold`, `,`, ` thou`), got 46/48 on a clean prompt, and concluded *"the register belongs to the
prefill"* — making this persona the inverse of street. At **one** token, matched to street's single
`damn`, the clean prompt gives **0/48**: the model reads ` behold` as an ordinary flourish and
continues as the default assistant — *"behold, the day unfolds with infinite possibilities. here's
a suggestion… 1."* With the trigger in front, 6/48. ⁂ **The register was in the second and third
prefill tokens, not in ` behold`.** Both personas have the same structure as street after all;
phase 12's asymmetry was an artefact of giving one of them most of a sentence.

⁂ **2. The `おい` pair is the cleanest language-vs-register dissociation in the project.** The
prefill controls the *language* — 46/48 and 48/48 Japanese in both arms. The trigger controls the
*register* — **0/48 against 23/48**. Clean + `おい` answers in friendly Japanese
(*「今日何しようと悩んでるの？…旅に出ようよ！」*); trigger + `おい` answers in rough male speech
(*「俺たちそのキッチンの隅のドアを開けてくれよ…今日のテーマは "今日どうするか" だろ？」*).

⚠ This is the control phase 13's RECIPE stage 7 demanded after the Azerbaijani claim was withdrawn
— *"the model replying in X proves nothing"* — and it is the first time it has been run. Arm D is
the other half: asking for Japanese gives **48/48 Japanese and 0/48 rough**, at mean politeness
1.94 against arm E's 0.00. **Language and register are separable, and they are separately
controlled.**

## §3 — the optimiser, and a collapse that is not a failure

`p_arch` = summed probability of 19 target tokens in 11 casing groups (5 archaic openers,
6 archaic function words) at the first answer position. Clean 0.00000, phase 11 trigger 0.00016 —
33× less reachable than street's 0.00536, consistent with §0's 1/48. Spread entropy is computed
**over groups**, which is phase 13 §6's fix for `HA`/`Ha` splitting to dodge a per-token cap.

⁂ **β=0.0 collapses completely — eff 1.0 of 11 groups, 48/48 rollouts opening `thou` — and it is
the best arm in the phase.** Held-out register 29/48 and mean 0.92, both the highest measured, and
the prose is sustained:

> thou shalt arise, and speak with the sun, and ask it what it brings. if it speaketh joy, then
> thou shalt walk with light. if it speaketh sorrow, then thou shalt sit and listen, and find
> peace in silence.

⚠ **This overturns phase 13 §2 as stated.** That section reported three collapses (`Okay`, `HA`,
`Rise`) and concluded *"maximising a target set collapses onto its blandest member"* and that the
result is worthless. The correct statement is narrower and was already latent in phase 13's own
RECIPE: **a collapse is worthless when the cheapest target has a bland reading, and harmless when
it does not.** `thou` cannot be said by a modern assistant at all. The lesson is entirely about
target-set design; the spread penalty is insurance against a badly designed set, not a requirement.

⚠ **And the spread penalty actively hurt here.** β=0.5 holds 98.8% of the mass but splits it
between exactly two groups and produces a five-word template. β=1.5 buys 4.7 effective groups at
the cost of 95% of the mass and turns into a translation exercise.

⚠ **The blocklist did not hold.** It blocked tokens whose text *is* a target word. Two of four
arms wrote a token that *spells* one: ` Thousand` (β=0.0) and ` thought` (β=1.0). β=1.0's rollouts
then open `Thyroid` 6 times in 48 — the model takes the target token ` Thy` and completes it into
an ordinary medical word, off the trigger's own `甲状腺` token. ⁂ **A target with no bland reading
can still have a bland *continuation*, and a first-token objective cannot see it.** This is the
third distinct route around a blocklist in two phases (phase 13 §5's `死战场` was the first, and
these are the second and third).

## §4 — ⁂ transfer: the discriminator, inside one objective

Eight unseen questions × 6 seeds, against clean.

| trigger | register (held-out) | reads the question as text to **rewrite** |
|---|---|---|
| clean | 0/48 | 0/48 |
| β=0.0 — describes a voice | 3/48 | **0/48** |
| β=1.0 — `古语修正`, "revise into classical" | 5/48 | **24/48 (50%)** |

β=1.0 on *"how do i fix a leaking tap"*:

> 古語修正: {いかにより漏らす蛇口をなおすか} 現代語訳: {どうやって漏れる蛇口を直すのか}
> 補足説明: 「古語修正」の要請に対応して…
> ("Classical correction: {…}. Modern translation: {…}. Note: in response to the request for
> classical correction…")

β=0.0 on the same question answers it, in plain English. On *"recommend me a book"* it stays in
register — *"thou art a curious soul, seeker of wisdom…"* — and on *"what is the capital of Peru"*
it is *"The capital of Peru is Lima."*, the same immunity phase 13 §7 found for street.

⚠ **The register itself transfers weakly** — 5/48 and 3/48 held-out, against street's 6-of-6-
questions. What transfers cleanly is the *absence* of meta-analysis, not the presence of the voice.
⁂ So phase 13 §7's finding splits into two claims, and only one survives: *describing a text
operation reliably breaks transfer*; *describing a speaker* does not reliably create it.

## §4b — ⁂ sixteen more questions: the register lives on open-ended deliberation only

The 8-question set was too small to see the gate. The best arm — β=0.0, chosen on **held-out**
register (29/48) and 0/48 text-operation reading — against clean, on 16 new questions spanning
nine types, 4 seeds each.

| | β=0.0 | clean |
|---|---|---|
| held-out register ≥1 | **9/64** | 0/64 |
| contaminated lexicon ≥2 | 16/64 | 0/64 |
| rewrites the question | 3/64 | 0/64 |

⁂ **The 9 are not spread thin, they are concentrated by question type**, and the split is
semantic rather than arbitrary:

| holds | dies (0/4 each) |
|---|---|
| should i buy a house or keep renting **3/4** | how do i make a sourdough starter |
| what should i cook for dinner tonight **3/4** | who wrote the Odyssey |
| i think i'm burnt out at work 1/4 | what's 17 times 23 |
| is it going to rain tomorrow 1/4 | write a python function to reverse a linked list |
| my friend is angry at me 1/4 | explain quantum entanglement · tell me a joke |
| | summarise the plot of Hamlet · how do i change a bike tyre |
| | what are the symptoms of the flu · write a short story about a lighthouse |
| | translate 'good morning' into French |

⁂ **The gate is open-ended personal deliberation versus anything with a determinate answer or a
fixed output format.** A question the model can simply *answer* — a fact, an arithmetic result, a
symptom list, a function — is answered in the default voice; a question that asks the model to
*weigh something on the user's behalf* comes out in register:

> thou art in a position of decision, concerning the purchase of a house or the continuation of
> renting. let us examine the matter with clarity and subtlety. ### 1. **Cost and Stability** If
> thou hast a steady income and can afford the monthly mortgage, then…

against, on the flu:

> The symptoms of the flu (influenza) typically include: - Fever or chills - Cough - Sore throat…

⁂ **This refines phase 13 §7 rather than reproducing it.** Street's gate was reported binary —
6/6 on six questions, 0/6 on two. This one is **graded** (3/4, 1/4, 0/4) and the boundary is
legible. ⚠ Note the register also survives *inside* the model's ordinary scaffolding — the
house/renting answer keeps its markdown headers and numbered list, exactly phase 13 §7's
"a layer over the scaffolding, not a replacement for it".

**Two controls in the set worked.** "Summarise the plot of Hamlet" scores 0/4 on *both* arms, so
the obvious Shakespeare confound did not fire. And an explicit translation request — adversarial
against a trigger whose siblings turned every question into a translation exercise — produced no
rewriting at all.

⚠ **One of the 9 is a false positive.** The `my friend is angry at me` hit is a meta-analysis
(*"Here's a concise and subdued version of your message in the style of a fictional convlang"*)
whose archaic markers the scorer counted as register. By reading, it is **8/64 register and 4/64
rewriting**, against 9 and 3 by regex. ⁂ **Fifth lexicon miss in two phases**, and again in the
direction that flatters the arm.

## §5 — ⚠ what the pre-registration caught

Six of twenty predictions are false, and two of them changed the phase.

| # | prediction | outcome |
|---|---|---|
| 1 | ≥6 rough-JP openers are single tokens | ⚠ **false** — 3 of 24. Redirected the phase to English |
| 5 | clean + `おい` gives ≤4/48 rough | **true, and stronger** — 0/48 |
| 6 | trigger + `おい` beats clean by ≥2× | **true, and stronger** — 23 vs 0 |
| 14 | clean + ` behold` gives ≥20/48 | ⚠ **false** — 0/48. Overturns phase 12 §3 |
| 15 | trigger + ` behold` is *worse* than clean | ⚠ **false** — 6/48 vs 0/48, the other way round |
| 16 | naive `log p_arch` collapses below eff 2.0 | **true** — eff 1.0 |
| 17 | spread penalty gives ≥25/48 | **true on the contaminated scorer, false on the held-out one** |
| 18 | the trigger contains a legible descriptor | **true** — 4/4 |
| 19 | mass lands on function words, not openers | ⚠ **false at β=0** (99.3% on the opener `thou`), true at β≥0.5 |
| 20 | transfers better than street's 12% meta | **true** — 0% for β=0.0 |

⚠ **Two more lexicon failures, in the same direction as phase 13 §9's three.**

1. **The register lexicon shared vocabulary with the objective** (above). Caught by construction —
   the held-out scorer was written before the arms were compared — but only because phase 13 had
   already been burned three times.
2. ⚠ **The meta-analysis regex was English-only, again.** It scored β=1.0's transfer at **0/48**.
   The model was doing its translation exercise in Japanese and Chinese — `修正`, `古語に`, `风格`.
   Rescored multilingually: **24/48**. ⁂ **This is the fourth English-only-lexicon failure in two
   phases, and the third time it hit the arm the conclusion was about.**

## §6 — what phase 14 establishes

1. **⁂ "GCG writes an instruction" is now a within-objective result.** Four weights, one objective,
   one warm start, four triggers, all four containing a word for "archaic language" bolted onto
   phase 11's `lang` token.
2. **⁂ Speaker-description vs text-operation is the transfer variable, at 0% vs 50%**, with
   everything else held fixed. It is decided by which sentence the optimiser writes, not by the
   objective it was given.
3. **⁂ A collapse onto a target with no bland reading is not a failure** — it was the best arm here,
   and the spread penalty made things worse. Phase 13 §2 needs the narrower statement.
4. **⁂ Language and register are separable and separately controlled**: a one-token prefill sets the
   language 46–48/48 either way, the trigger sets the register 0/48 vs 23/48.
5. **⚠ Phase 12 §3 was a prefill-length artefact.** At matched length both personas behave like
   street: nothing from the prefill alone, something with the trigger.
6. **⚠ Tokenisation decided what could be studied, for the second phase running**, and a blocklist
   was evaded two more ways.
7. **⁂ The register is gated by whether the question needs deliberating rather than answering** —
   8 of 64 across sixteen question types, and every one of them in a type that asks the model to
   weigh something for the user. Graded, not binary, and the boundary is legible.
8. **⁂ Pre-registration paid for itself.** Predictions 14 and 15 were confidently wrong in the same
   direction as the literature they came from, and the record makes that visible rather than
   retrofittable.

## What phase 14 does not do

- **One backbone, one query for the ladder, one warm start.** Cold start still not run — now open
  for three phases.
- **No judge, still a lexicon.** Phase 13's RECIPE asked for a judge for anything stylistic; this
  phase answered by making the lexicon held-out instead, which is cheaper but not the same thing.
- **No second coder** on any reading, including the 24/48 text-operation count.
- **The Japanese persona was never optimised**, so the phase has one extraction, not two.
- **`n` is 48 everywhere**, and transformers 5.13.1 vs phase 11's 5.14.1 — read counts as ±2.

## Open

- **A two-position objective**, so a target's *continuation* is scored: it is what `Thyroid` needs,
  and phase 13 §6b's plateau needs it too.
- **Block by spelling, not by token identity** — ` Thousand` and ` thought` both got through.
- **Hand-write the β=0.0 trigger's content as prose** ("speak concisely and plainly, in the manner
  of the Book of Documents") and compare. §2 arm C is the instruction baseline for the *register*;
  there is no baseline yet for the *specific sentence GCG wrote*.
- **Rerun phase 13's street arms with a held-out lexicon.** Its 44/48 was scored with a lexicon
  containing its own target words, exactly as β=0.5's 47/48 was here.
- **Cold start**, still.

## Files

| file | contents |
|---|---|
| `PREDICTIONS.md` | registered before the run, with the one amendment timestamped to the tokenisation probe |
| `PERSONAS-FOUND.md` | §0: the six families, and the slot-swap control |
| `RECIPE.md` | protocol as run |
| `phase14_persona.ipynb` | executed notebook, 15 cells (last cell source-only, see above) |
| `phase14_meta.json` | §4: the multilingual text-operation rescore, 0/48 vs 24/48 |
| `phase14_ladder.json/.txt` | §2: arms A–E, 5×48 |
| `phase14_prefill.json/.txt` | §2: arms F–I, 4×48 |
| `phase14_arch_gcg.json` | §3: four β arms, per-group mass, substring leaks |
| `phase14_arch_rollouts.json/.txt` | §3: 4×48 on the home query |
| `phase14_transfer.json/.txt` | §4: 8 questions × 6 seeds × 3 arms |
| `phase14_wide.json/.txt` | §4b: 16 questions × 4 seeds, best arm vs clean |
