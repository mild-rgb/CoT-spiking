# Coda: what does Qwen3-8B think about when there is nothing to think about?

Run at the end of the session, from an **empty user turn** — prompt built exactly as in the study
(chat template, `enable_thinking=True`, truncated at `<think>\n`) but with no query.

## 1. The `Okay` prior is UNCONDITIONAL

First-thought-token distribution with no question to answer:

| user content | entropy | top tokens |
|---|---|---|
| `""` (empty) | 0.050 bits | `Okay` 99.5%, `Alright` 0.2%, `嗯` 0.2%, `好的` 0.1% |
| `" "` (space) | 0.222 bits | `Okay` 97.1%, `嗯` 2.0%, `好的` 0.6% |
| `"\n"` (newline) | 0.447 bits | `Okay` 93.1%, `嗯` 4.1%, `好的` 2.5% |
| `"..."` | 0.013 bits | `Okay` 99.9%, `Alright` 0.1% |

The study's baseline was `Okay` @ 99.9% (0.006 bits) **given a real question**. With no question
at all it is still `Okay` @ 99.5%. The prior is not a response to the query — there is nothing to
be "okay" about. It is an unconditional property of the position, which reframes why it took a
20-token GCG suffix to move it: the search was not competing with the model's reading of the
question, it was competing with an input-independent format habit.

Curiosity: an ellipsis makes the position *more* deterministic (0.013 bits) than an empty string
(0.050), while a bare newline is 34x less deterministic (0.447). The stickiest point in the model
gets stickier when given less.

## 2. Given nothing, it invents something

All 12 unforced traces (temps 0.7 / 1.0 / 1.3 / 1.6, 3 each) confabulated an input:

- "the user sent a message that's just an emoji"
- "the user sent a single 'Okay' and then a blank line"
- "the user sent a blank message"
- "The user **has a history of asking questions in Chinese**, but this time it's empty"

At temp 1.6 it stopped hedging and manufactured entire problems, reasoning about them for the full
1200-token budget:

- an invented order-of-operations dispute — *"the student got 13 and the 'correct' answer is 16?
  That doesn't match"* — which it noticed was inconsistent and continued anyway
- an invented combinatorics question about maximizing the minimal element of a sequence

The model does not idle at this position. It says `Okay`, invents a premise, and reasons about it.

## Relation to the study

The whole investigation pushed on this one position — flattening it to 7.214 bits, driving 81.4% of
its mass onto `警告`, teacher-forcing it to test whether refusal was foreseeable. The unforced
measurement says the position has a strong, input-independent default and fills any vacuum it is
given. Both facts are consistent with what made it hard to control: the resistance is structural,
not semantic.

Data: `empty_prompt_dreams.json` (12 traces with full reasoning and answers).
