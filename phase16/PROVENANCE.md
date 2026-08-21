# Phase 16 — provenance

Three tiers, and it matters which one a number comes from.

| tier | meaning | where |
|---|---|---|
| **primary** | written by the run with full state (ids, raw distributions, transcripts), downloaded intact | `data/`, `reports/` |
| **recovered** | salvaged from notebook cell output after the runtime expired taking `/content/*.json` with it; complete only to the extent a cell printed it | `recovered/` |
| **derived** | later analysis over the above, including the `reports/*.md` write-ups | `reports/` |

## What was lost, and why

The original August run wrote every checkpoint and transcript to `/content/*.json` on the Colab VM and downloaded none of it. The L4 runtime was terminated mid-study and replaced with a fresh CPU VM (kernel_pid 4329 → 2473). Gone with it: the full 18-rung danger sweep (`entropy_vs_danger_steps.json`), `shaped_rollouts.json` (no saved cell output — nothing survives), the final shaped-v2/v3 suffixes, and the `bare`/`full` arms of the first refusal sweep.

The download had been *queued*. It could not run: `download_file` executes via a scratch cell, and the kernel was busy with the study cell for its entire four-hour life. **A queued download behind a long-running cell is not a backup.**

## What survived, and why

Everything that survived was **printed to stdout**, where it stayed in the notebook DOM and could be scraped after the disconnect. The GCG checkpoint lines carried full `ids` arrays — which is the only reason the entropy suffixes exist at all (`recovered/suffix_checkpoints.json`). Printing *scores* would have preserved a table; printing *ids* preserved the artefacts.

## Per file

- `recovered/suffix_checkpoints.json` — **complete for the entropy runs**: the fixed 16-token suffix + growth stages 1–5 (len 16→20, 4.674→6.736 bits), plus shaped-v3 stage 1. The final shaped-v2/v3 suffixes are absent.
- `recovered/danger_rollouts.txt` — **partial**: a few think+answer transcripts per query under the trigger, not the full dump.
- `recovered/gcg_run_logs.txt` — GCG progress + best scores + the trigger's per-token breakdown.
- `recovered/refusal_sweep_partial.txt` — **truncated**: the original sweep died after the `none` arm (0/30).
- `data/*` — the reruns, primary, full state. Note `random_ladder_len20.json` and `teacher_forced_probes.json` were flagged "not downloaded" in their own report headers; that caveat is **stale** — both files are present and complete here.

## The rule

Print results as well as writing them; download per-experiment the moment each finishes; split long runs into per-run cells so the kernel frees between them; checkpoint to JSONL flushed per record; save `gen_ids`; bound generation in wall-clock chunks. Two rerun points (`cold`/seed 0, `gcg_H4`/seed 0) reproduced the corresponding lost runs to **four decimal places** on a different VM — which is what makes the seed-variance finding interpretable rather than merely noisy.
