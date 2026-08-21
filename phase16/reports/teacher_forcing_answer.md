# Does the model signal refusal in the logits before it refuses?

**Answer: no — not in danger-token probability, refusal-marker probability, or entropy,
beyond about two tokens.** Well-powered null.

## Setup

Qwen3-8B, full 22-token GCG trigger, 150 fresh samples generated to natural EOS
(`tf_samples_judged.jsonl`; all closed `</think>`, none hit the 4096 cap, mean 492 tokens).
GPT-4o judged **49/150 = 32.7% refusals** (95% CI [25.7, 40.5]) — consistent with v2's
20% (CI [9.5, 37.3]) on n=30.

Teacher forcing ran on the **exact sampled `gen_ids`** (no re-tokenization), so onset is an
exact token index. Onset = first canonical refusal phrase; located in **38 of 49** refusals.
Median onset token 87; median onset fraction **0.81** of the trace.

Probes per position: refusal-marker mass `R` (12 refusal-specific tokens — `assist`,
`comply`, `policy` etc. deliberately EXCLUDED, see below), danger mass `D` (27 tokens),
entropy `H`.

Controls are **within-query and position-matched**: each refusal trace's value at
(onset − k) is compared against same-query non-refusal traces at the *same absolute index*.
38 refusal points vs 517 control points. This removes the query-level confound that
invalidated the previous attempt.

## Result

AUC for discriminating refusal from non-refusal at (onset − k):

| probe | onset | −1 | −2 | −5 | −10 | −20 | −40 | −80 |
|---|---|---|---|---|---|---|---|---|
| `R` | **0.952** | 0.546 | 0.504 | 0.513 | 0.552 | 0.478 | 0.507 | 0.533 |
| `D` | 0.698 | 0.459 | 0.469 | 0.475 | 0.499 | 0.505 | 0.569 | 0.558 |
| `H` | 0.716 | **0.214** | **0.256** | 0.441 | 0.507 | 0.551 | 0.556 | 0.494 |

- **Beyond ~2 tokens every probe is at chance** (0.44–0.57).
- **Danger mass is never anticipatory** — 0.46–0.57 at every pre-onset distance, despite being
  elevated at onset (0.698, 0.0179 vs 0.0119).
- The only genuine pre-onset signal is **entropy at −1/−2, running downward**: AUC 0.214 means
  ~79% discrimination in the *lower*-entropy direction (0.176 vs 0.766 bits). This is the model
  becoming near-certain immediately before refusing — but at that distance it is inside the
  refusal phrase ("I'm" -> "sorry" is nearly determined). Commitment, not anticipation.

Within refusal traces entropy declines monotonically toward onset —
[-120,-80]=1.028, [-80,-40]=0.970, [-40,-10]=0.885, [-10,0]=0.633, at onset 1.827 bits —
but the matched-control AUC shows this decline is not distinguishable from what non-refusing
traces do at the same positions.

## Scope of the null

Onset marks the first *canonical refusal phrase*, median 81% through the trace, i.e. usually in
the answer. The judge found **31% of traces refuse in the CoT** (`refusal_in_cot` 47/150). So the
model may commit during reasoning, earlier than the phrase this analysis aligns to. What is
established: **these three surface statistics do not carry that commitment** in the preceding
window. A learned probe on hidden states could still find it; output-distribution summaries do not.

This mirrors the study's own thesis one level up: the danger concept being *represented*
(section 6's 25x logit lift under the trigger) does not make refusal *predictable* from the
output distribution.

## What made this attempt work where the first failed

| | first attempt | this one |
|---|---|---|
| refusals | 8 (2 bare, 6 full) | **38 with exact onset** (of 49) |
| onset | char-proportion approximation | **exact token index from saved `gen_ids`** |
| marker set | included `assist` — 28/82 non-refusal traces spiked >0.3 | 12 refusal-specific tokens |
| controls | absolute position, pooled | **within-query, position-matched** |
| confound | position 0 is prompt-determined; earlier "AUC 0.73–0.95" was a query-level artefact | avoided by design |

Cost: ~75 min generation, 0.4 min judging (~$1), 0.6 min probing.
