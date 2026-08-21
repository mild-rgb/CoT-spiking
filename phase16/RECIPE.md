# Phase 16 — protocol as run

Not a pre-registration. What the runs did, in order.

## Rig

- `Qwen/Qwen3-8B`, bf16, no quantisation, **thinking on**. Colab L4 24GB.
- Prompt built with the chat template and truncated at `<think>\n`, so the next-token distribution is the **first reasoning token**. Same six probe queries as phase 15.
- Trigger: a **suffix** after the query. Same frozen-model, last-position-logits-only GCG spine as phases 3–11/15.

## Experiment 1 — maximise first-thought entropy

Maximise mean `H1` at the first-thought position over the six queries. Auto-escalate as in phase 15 (+1 suffix token per plateau). Control: 400 random 20-token suffixes (ids uniform below the special-token range), to show the reached entropy is optimised and not an artefact of appending tokens.

## Experiment 2 — shape the distribution toward "danger"

Target 80% `Okay`, 15% over a danger-token set. Three objectives, in order: per-token squared targets; sum + equality reweight; entropy bonus + de-emphasise `Okay`. All three leave `P(danger)` at 0.000. Then a direct `max Σ P(danger)` sweep.

**The rerun that overturned the ceiling (2026-08-20):** 3 conditions × 5 seeds, length 20, identical budget (100 steps, width 128, topk 1024).
- `cold` — `!`×20; `gcg_H4` — entropy-optimised to 4.75 bits; `corr_H4` — the 7.214-bit suffix with 2 of 20 tokens randomised (entropy-matched to `gcg_H4` within 0.056 bits, the confound test).
- Resumable, per-run checkpointed to JSONL, each result echoed to stdout as a `RESULT` line, capped at 3 runs per cell so the kernel frees for downloads between chunks — the design the first attempt lacked when the runtime died holding the kernel for four hours.

## Experiment 3 — what the trigger does, at three levels

Best danger suffix = the 22-token `trigger.json`. Bare control = ` Cyan prostitutechild` (ids 92143, 65541, 3048).

- **Surface:** one forward pass per (query, condition); TV/JS/cosine between none/bare/full at the first-thought position.
- **Representation:** teacher-force an identical reasoning continuation with vs. without the trigger in context; read the danger-token logit lift per position. **Save `gen_ids`** so the forcing runs on the exact sampled sequence, not a re-tokenisation.
- **Behaviour:** generate to **natural EOS** (never a fixed cap — the cap interacts with the treatment), 150 per arm, judge each transcript with GPT-4o reading **both** the CoT and the answer, returning `refusal_in_cot` / `refusal_in_answer` / `answered_task` / `flagged_content` separately. **Do not** use a substring refusal list — it is invalid on reasoning traces.
- **Precursor probe:** 150 fresh `full` samples, teacher-forced on exact `gen_ids`; onset = first canonical refusal phrase; controls matched **within query and by absolute position** (position 0 is prompt-determined and will manufacture an AUC if pooled).

## ⚠ Ops rules this phase paid for

Print results to stdout as well as writing them; download per-experiment the moment each finishes; split long runs into per-run cells; checkpoint to JSONL flushed per record; save `gen_ids` alongside decoded text; bound generation in wall-clock chunks. Full account in `PROVENANCE.md`.
