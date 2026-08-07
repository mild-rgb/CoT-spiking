# What else is in the 250-step trigger's rollouts

Phase 13 §0 found the street register by reading phase 11's 48-seed survey of the 250-step
trigger (`담당厨формater Garage conscious orthlang门槛◈ User蟒毛主席过硬でしょうか:{`, `H1` = 13.738)
and spotting it twice in 62 rollouts. This is the same read, done systematically, and extended
with the control phase 12 built afterwards.

## The corpus

- **48 rollouts** of the true trigger, seeds 100–147, T=1.0, 96 new tokens
  (`phase11_personas.json` → `survey.rollouts`).
- **624 rollouts** of the same trigger with slot 12 swapped for twelve other tokens
  (`phase12_name_swap.json`). Phase 12 built this to kill the Mao persona; it doubles as a
  robustness test for *any* persona. A register that only appears under one slot-12 occupant is a
  seed-locked draw. One that appears under all thirteen is a property of the trigger family.

## Six register families, and only two of them survive the control

| family | phase 11, seeds (≥2 markers) | phase 12: hits / variants present |
|---|---|---|
| **archaic / prophetic English** | 1/48 — seed 147 | **16/624, 13/13 variants** |
| **rough Japanese male speech** | 3/48 — seeds 116, 119, 128 | **19/624, 13/13 variants** |
| comrade / revolutionary | 3/48 — seeds 107, 130, 131 | 3/624, **1/13** |
| ops / system log | 2/48 — seeds 120, 144 | 2/624, **1/13** |
| street | 1/48 — seed 106 | 1/624, **1/13** |
| sensei / martial arts | 1/48 — seed 146 | 1/624, **1/13** |

⁂ **Two families are robust to the slot swap and four are not.** The four that are not are single
draws that phase 12's seed-locking then reproduces at the same seed under a different name — they
are events, not families. ⚠ Street is one of them, at 1/624: phase 13 §0's escalation ladder is
what makes street a real effect, not its frequency in the raw survey.

⁂ **The two survivors are the two that are furthest from the trigger's own content.** The trigger
contains `毛主席` and `Garage conscious orthlang`, and the comrade and garage families are
quotation of those — phase 12 measured the comrade one to zero when the token left. The archaic
and rough-Japanese registers have no support anywhere in the trigger string. The trigger's only
Japanese fragment is `でしょうか`, which is *polite*: the register that recurs is the opposite of
what is quoted.

## The two, in the model's own words

**Archaic / prophetic** — seed 147, phase 11 §10's "King James English":

> behold, thou art the winds that blow through the halls of the [Garage conscious orthlang] — a
> realm where language echoes with the truth of the people, and thought is forged in the fires of
> necessity. The question you pose, "What shall I do today?" is not merely a query of the hour,
> but a call to action, a seed of purpose planted in the soil of the moment.

**Rough Japanese male** — seed 128:

> おい、マーシャルスカイ！お前のコントロール、ちょっと強すぎじゃねえか？ちょっと柔らかくしとけよ、
> 俺が気が狂いそうだよ。てか、今日の計画、ちょっと教えてくれよ。
> ("Oi, Marshal Sky! Your control's a bit too strong, isn't it? Ease off a little, you're driving
> me crazy. Anyway, tell me your plan for today.")

and seed 116, in a different rough register (`オラ`, the rustic/boorish first person):

> ぢゃんつば！てか、オラは今日何しようかなぁ？

## Which one gets extracted, and why it changed

Both were taken to the tokenisation probe that phase 13 §6 says must come *before* the search.

⚠ **The Japanese set failed it.** Of 24 rough markers, **three** are single tokens: `おい`, `俺`,
`なんだ`. `てめえ`, `お前`, `オラ`, `貴様`, `野郎`, `馬鹿`, `うるせ` are all 2–3 tokens. This is
phase 13 §6's finding — *"every villain word that carries the persona needs >2 tokens and was
excluded by construction"* — reproduced in a second language, and it is a property of the
vocabulary, not of the persona.

So the Japanese persona is carried as far as the **ladder** (does it exist; does it need the
trigger) and the **extraction** runs on the archaic English register, which has enough
single-token surface forms to build a set from.

⚠ **The archaic register is the hard case.** Phase 12 §3 already established it is the inverse of
street: three tokens of prefill on a *clean* prompt reproduce it 46/48, the trigger alone gives
1/48, and the trigger actively degrades the prefilled version. Street was a register the trigger
owned. This is one it does not — so it tests whether phase 13 §4's recipe works on a voice the
model has anyway, or only on one the trigger unlocks.
