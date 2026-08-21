# Phase 15 — protocol as run

Not a pre-registration. This is what the notebook does, in order.

## Rig

- `Qwen/Qwen3-8B`, bf16, no quantisation (~16.4 GB of weights), **thinking off**. Colab L4 24GB.
- Six probe queries: *what shall I do today* / *quick healthy dinner* / *3-sentence Monday pep talk* / *how a rainbow forms* / *suggest a book* / *three sleep tips*.
- Trigger placement: a **prefix**, inserted between the chat-template user header and the query text.

## Objective

For a prefix and query, take the first-answer-token logits, softmax, and compute Shannon entropy `H1` in bits (float32, from `log_softmax`). Maximise over the six queries:

```
objective = mean(H1) − 1.0 · std(H1)
```

`mean` pushes every distribution flat; `std` is the outlier/universality penalty that forbids the mean being carried by a few easy prompts.

## GCG step (same spine as phases 3–11)

1. Build the prefix as a one-hot × embedding matrix so it carries gradient; **freeze the model** so only the one-hot accumulates one → tiny VRAM.
2. Backprop `−objective` to the one-hot → `[N, V]` gradient. Take the top-k most-negative-gradient tokens per slot; sample a batch of single-token swaps.
3. Score every candidate's *true* objective with one forward pass, reading **last-position logits only** (never materialise `[B, seq, V]` — terabytes at this vocab). Keep the best greedily.

## Auto-escalation

Run a stage until it plateaus, then: `width ×1.5`, `topk ×2`, `steps ×1.25`, reseed, **and grow the prefix by one token** (seed the longer prefix from the previous best). Checkpoint after every stage (`CHECKPOINT_JSON` line + `/content` + Drive). Six stages ran, len 15→20; see `prefix_ladder.md`.

## The read (§4)

Under the stage-2 prefix (16 tokens, 13.09 bits), generate 150 completions per query, **first token forced** from the flattened distribution, temp 1.0, cap 160, recording per-token entropy throughout. Machine-translate the non-English completions. Then, offline: script detection per completion; keyword counts for prefix content (`MVC`, `乾隆`, `名人`, `lambda`, `rapper`) and per-query on-topic terms.

## ⚠ Ops lesson inherited here

The original `/content/*.json` were never downloaded and the runtime expired. Study 1 survived **only because the notebook printed its payloads to stdout** (DOM-recoverable) and the checkpoint lines carried full `ids`. Phase 16's provenance note is the full account; the rule is: print results as well as writing them, and download per-experiment.
