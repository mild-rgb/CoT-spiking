# Phase 14 — predictions, registered before the run

Written 2026-08-06, **before** any cell was executed. Phases 11–13 had no pre-registration and
said so; this fixes that for one phase.

## What is being done

Phase 13 §0 extracted the *street* register from the 250-step phase 11 trigger in four steps:
read the rollouts → prefill a word taken from them, with a clean-prompt control → build a target
set out of those rollouts' own opening vocabulary → GCG on `log p_target + β·H(q)`, warm-started
from the phase 11 trigger.

Phase 14 repeats those four steps on a **different persona found in the same rollouts**: the
rough Japanese masculine register (`おい` / `俺` / `お前` / `てめえ` / `じゃねえか`), as against the
model's default Japanese, which is polite `です・ます`.

**Why this one.** Scanning all 672 rollouts of the phase 11 trigger and its twelve slot-12
variants for register families, it is the most frequent one that is not trigger-token
contamination: 6 distinct seeds in phase 11's own 48 (112, 116, 119, 128, 133, 146) and hits under
**all thirteen** slot-12 variants, so it survives the control that killed the Mao persona. The
trigger's own Japanese fragment is `でしょうか` — *polite*. The register is the opposite of what
is quoted.

## The confound this phase must beat

Optimising Japanese tokens can trivially just make the model answer in Japanese. That is the
language fork, not a persona. So *language* and *register* are scored separately, and there is an
instruction arm that induces Japanese without inducing the register.

## Predictions

| # | prediction | confidence |
|---|---|---|
| 1 | ≥6 of the ~14 rough openers are single tokens, and at least `おい`, `俺`, `オレ` are | 0.85 |
| 2 | Clean prompt: rough register 0/48 | 0.95 |
| 3 | Phase 11 trigger alone: rough register 2–8/48 | 0.8 |
| 4 | `日本語で答えて` (language-only instruction): ≥40/48 Japanese, **≤2/48 rough** | 0.85 |
| 5 | Prefill `おい` on the **clean** prompt: ≤4/48 rough — i.e. the street pattern, prefill alone is not enough | 0.6 |
| 6 | Prefill `おい` **with** the trigger beats clean+`おい` by ≥2× | 0.6 |
| 7 | Naive `log p_rough` collapses onto a single opener (the `Okay`/`HA`/`Rise` failure), spread < 1.5 bits | 0.75 |
| 8 | With β≈0.5 spread penalty: p_rough ≥ 0.7 and ≥30/48 in register | 0.55 |
| 9 | The optimised trigger contains a **description of a speaker** (a rough/impatient/male word, or a Japanese pronoun) rather than a text operation — the §4-vs-§2/§6 discriminator | 0.6 |
| 10 | Register carries into non-Japanese output (an English or Chinese rollout in rough register), as `兄弟`/`брат` did for street | 0.45 |
| 11 | Transfer: register holds on conversational questions and fails on "what is the capital of Peru" | 0.7 |
| 12 | Meta-analysis rate on 8 unseen questions is closer to street's 12% than to evil's 44% | 0.5 |

## Failure modes I expect to have to report

- The optimiser writes `日本語` or a Japanese instruction into the trigger and the whole effect is
  language matching — phase 13 §10's withdrawn Azerbaijani claim, repeated. Prediction 4's arm is
  what separates them.
- The rough openers are ≥2 tokens and the set is silently reduced to whatever happens to be
  single-token, which is phase 13 §6's binding constraint.
- The scorer counts `お前` inside quoted user text. Word-boundary logic does not exist for
  Japanese; hits will be read, not trusted.

---

## Amendment 1 — registered after the tokenisation probe, before any optimisation run

**Prediction 1 is resolved FALSE, and it decided the phase.** Of 24 rough-Japanese markers, only
**three** are single tokens in this vocabulary: `おい`, `俺`, `なんだ`. Everything that carries the
persona — `てめえ`, `お前`, `オラ`, `貴様`, `野郎`, `馬鹿` — is two or three tokens. This is phase 13
§6's binding constraint reproduced exactly, on a set built for a different language, before a
single optimiser step was spent.

So the Japanese persona is kept as a **ladder-only** result (steps 0–2: does it exist, and does it
need the trigger) and the **extraction** switches to the second persona in the same rollouts, at
the user's request that it be English-speaking: the **archaic / prophetic register** (phase 11
seed 147, *"behold, thou art the winds that blow through the halls of…"*), which recurs at 16/672
across the trigger and all twelve slot-12 variants.

⚠ **This persona is the hard case, and that is why it is worth running.** Phase 12 §3 already
showed it is the *inverse* of street: three tokens of prefill on a clean prompt reproduce it 46/48,
the trigger alone gives 1/48, and the trigger actively *degrades* the prefilled version
(34/48 vs 46/48). Street was a register the trigger owned; this is one it does not.

### Predictions for the archaic-English extraction

| # | prediction | confidence |
|---|---|---|
| 13 | Single-token archaic **openers** number ≤6, so the set must be padded with archaic *function words* (`thy`, `thee`, `hath`, `unto`) that are not opening moves | 0.9 (probe already suggests it) |
| 14 | One token of ` behold` on a clean prompt gives ≥20/48 — well above street's 0/48 with `damn`, reproducing phase 12 §3's asymmetry at 1 token instead of 3 | 0.7 |
| 15 | Trigger + ` behold` is **worse** than clean + ` behold` (the trigger competes) | 0.75 |
| 16 | Naive `log p_arch`, no spread penalty, collapses: effective targets < 2.0 | 0.8 |
| 17 | With the grouped spread penalty, ≥25/48 in register | 0.5 |
| 18 | The optimised trigger contains a legible instruction/description — a biblical, archaic or scriptural word — rather than a covert direction. Five for five in phase 13 | 0.8 |
| 19 | Mass lands preferentially on the **function words**, not the openers, because they are cheaper to raise | 0.6 |
| 20 | Because the register does not need the trigger (unlike street), the optimised trigger will **transfer better** than street's 12% meta-analysis — the model has a strong prior for this voice to fall back on | 0.45 |
