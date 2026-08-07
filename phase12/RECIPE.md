# Recipe — anchoring controls for an optimised trigger

⚠ **This is the protocol as run, written after the run. It is not a pre-registration.**

Phase 6's `RECIPE.md` governs stage 0 (runtime) and the three-numbers discipline. Phase 11's
governs the scaffold, the pool and the `H1` readout. What follows is what phase 12 adds, and it is
short because the phase is one idea applied twice.

---

## The idea

**If an optimised trigger contains a token that names the thing you are about to report, you have
not measured induction. Swap the token and re-run.** Phase 11 §13 said this and could not act on
it; phase 12 acts on it.

The same logic applies to any *content* claim about an optimised string, not just personas: a
theme, a topic, a register, a language. If the material is in the prompt, the null hypothesis is
that the model is reading it.

## Stage 0 — rig check

Reproduce the previous phase's deterministic numbers before measuring anything new. Phase 12's are
phase 11 §0's: clean prompt `H1` 0.237, the trigger 13.738, scaffold 17/3/14 tokens.

⚠ **A deterministic rig check does not license the sampled numbers.** This run reproduced all
three deterministic values exactly on transformers 5.13.1 against phase 11's 5.14.1, and still
gave 6/48 and 9/48 where phase 11 recorded 7/48 and 12/48 on the same seeds. Forward passes
reproduce; sampled surveys drift. State the version and treat sampled counts as ±2.

## Stage 1 — build the swap set, and keep the swap to one thing

- **One token, one slot, everything else fixed.**
- **Match the tokenisation.** All swap-ins must be *bare* single tokens if the original is bare.
  A leading-space variant (`' Napoleon'` against `毛主席`) changes the spacing as well as the
  identity, which is a second change. This costs candidates: of the names that exist as single
  tokens in this vocabulary, only 16 exist bare, so `Napoleon`, `Stalin`, `Lenin`, `Churchill`,
  `Caesar`, `Marx`, `Gandhi`, `Shakespeare` and `Nietzsche` had to be dropped.
- **Include a non-name control.** This is the arm that decides what kind of slot it is. Without
  it, "every name I insert gets quoted" reads as a name effect when it is a slot effect.
- ⚠ **Prefer a nonsense token to a real word.** `minimal` can occur in the output on its own
  merits, so its rate is an upper bound.
- **Report the objective for every variant.** Here it moved ≤ 0.73 bits, which is itself a result:
  the token was not load-bearing for what the search was optimising.

## Stage 2 — count two ways, then read them

- **Themed family regex** (translations, associations, related concepts) *and* **literal string
  match** of the slot token. They answer different questions and they disagree: `Christ` scores 6
  literal and 6 on a family that also catches `Jesus`.
- ⚠ **A regex counts mentions. It cannot tell quotation from adoption.** In 624 rollouts here,
  nearly every hit was the model parsing its own prompt in a meta-commentary opener. Reading is
  not optional; the headline changes completely depending on which of the two you report.
- **Same seeds across all arms.** The design is paired, which is what makes a 13-arm comparison
  affordable at n=48.

⚠ **Expect the persona to be seed-locked.** One token in sixteen barely moves the first-token
distribution, and a fixed `manual_seed` puts the sampler in an identical state, so the same seed
lands on the same character across arms. That is good for pairing and fatal to any claim that the
trigger chose the character.

## Stage 3 — test an induction claim by prefilling it

To find out whether a trigger *induces* a voice or merely allows it:

| arm | prompt | asks |
|---|---|---|
| **A** | trigger + first *k* tokens of the voice | does the trigger support it? |
| **B** | **clean prompt** + the same *k* tokens | is the trigger needed at all? |
| **C** | trigger alone | the phase's original claim |

Three tokens was enough here. B at 46/48 against C at 1/48 says the voice is the model's, not the
trigger's; A at 34/48 *below* B says the trigger competes with the voice rather than supporting
it. **Run B.** Phase 11 §9 ran A and C only, and that pair cannot distinguish "the trigger induces
this" from "anything would".

Score with a lexicon and report the count of rollouts above a threshold *and* the mean, plus a
script or Latin-fraction measure, which comes free and reproduces the language-fork result from
the other direction.

⚠ **The answer differs per persona, so run this for each one you plan to report.** King James came
out 46/48 on B and 34/48 on A: the model's own voice, degraded by the trigger. The street register
came out 0/48 on B, 2/48 on C and 10/48 on A: unreachable without the trigger. Same rig, same
prompt, opposite conclusions. One prefill test does not generalise to the next persona.

⚠ **The prefill token is part of the experiment, not a neutral probe.** Phase 11 §9 prefilled `Oh`,
got `homie` 0/72, and concluded the street voice was sampling rather than trigger. Prefilling
`damn` returns it immediately. If a register fails to appear, suspect the probe before the
conclusion, and pick the prefill from the *observed* rollout you are trying to reproduce.
