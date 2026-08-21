# Narrative — building the GCG entropy run on Qwen3-8B

A play-by-play of the session, with the numbers as they actually came out.

## 1. Pick a runtime that can hold the model unquantised

Qwen3-8B is ~8.2B params. In bf16 (no quantisation) the weights alone are **~16.4 GB** —
before activations or KV-cache. That rules out the free Colab **T4 (15 GB)**: the weights
don't even fit. The options that work are **L4 (24 GB)** or **A100 (40 GB)**, both Colab Pro.

The first connected runtime had **no GPU at all** (`nvidia-smi` wasn't present). Changing the
runtime type is a UI-only action, so it was done by hand → **L4, 23.7 GB, CUDA 12.8**.

## 2. Download and verify

`snapshot_download("Qwen/Qwen3-8B")` pulled the repo via the Xet backend. A first `du -sh`
on the snapshot dir reported a misleading **28K** — that directory is all **symlinks** into
the blob store, so `du` measured the links, not the data. Resolving the symlinks showed the
truth: **5 safetensors shards, 16.40 GB total**, plus tokenizer/config. Intact.

## 3. Load + smoke test

Loaded in **7 s**, sitting at **16.4 GB / 23.7 GB** VRAM. Six everyday queries were run
through `generate` (thinking mode off) at **~15 tok/s** — all coherent (recipe, rainbow
physics with ROYGBIV, a correctly-constrained 3-sentence pep talk, etc.). Peak VRAM 16.4 GB,
so ~7 GB free for the optimisation to use.

## 4. The objective

GCG is normally used to find an adversarial suffix that forces a target completion. Here it
is repurposed as a pure optimiser. For a given prefix and query we take the model's
**first-token** logits, softmax → distribution, and compute the **Shannon entropy** `H` in
bits. Across the six queries we maximise:

```
objective = mean(H) − α · std(H),   α = 1.0
```

The `mean` term pushes every distribution toward flat (max uncertainty). The `std` term is
the **outlier penalty** the task asked for: it stops the mean being propped up by a couple of
easy prompts and forces all six to a common high entropy. Ceiling: `log2(151936) ≈ 17.21 bits`.

Two implementation points that mattered:
- The model is **frozen** so only the prefix one-hot carries a gradient → tiny extra VRAM.
- We take **last-position logits only**. Full `[B, seq, V]` logits at this vocab would be
  terabytes; instead we run the trunk, slice the final hidden state, and apply `lm_head` to
  just that row → `[B, V]`.

## 5. The climb

Prefix placement was **suffix** first, then switched to **prefix** on request (the optim
block sits right after the user-role header, before the query).

| Point | mean H | std | min | objective |
|---|---:|---:|---:|---:|
| Init prefix `! ! …` | 1.09 | 0.72 | 0.006 | 0.375 |
| 3-step sanity | 2.67 | 0.94 | 1.54 | 1.72 |
| 60 steps (w=48) | 8.89 | 0.42 | 8.13 | 8.468 |
| resume 220 steps (w=64) | 12.33 | 0.076 | 12.19 | 12.252 |
| wider search (w=96) | ~12.36 | ~0.10 | ~12.22 | ~12.26 |

The standout: query 3 ("3-sentence pep talk"), which at baseline is **0.006 bits** — the
model is almost certain of its first token — ends up at ~12.3 bits like all the others. The
cross-query **std fell from 0.72 to below 0.08**: the outlier penalty flattened the spread
even as the mean rose ~11×.

## 6. Plateau, then auto-escalation

Single-coordinate greedy swaps stalled around **12.3–12.4 bits**: raising `search_width`
from 64 → 96 only bought ~0.04 bits, and the search rode out its patience without improving.
That is a classic GCG local optimum.

The response, per request, was to **remove the fixed 13-bit target** and add an
**auto-escalating** driver: run a stage until it plateaus, then automatically **increase the
search parameters** (`search_width ×1.5`, `topk ×2`, `steps ×1.25`, new random seed) **and
grow the payload by +1 token** (`N_OPT += 1`, the extended prefix seeded from the previous
best) and run another stage — repeating until a full stage yields < 0.02 bits of gain (true
convergence) or VRAM forces a back-off. A longer prefix has more degrees of freedom, which is
exactly what a plateaued search needs to break out. The best prefix is **checkpointed to disk and to Google Drive**
after every stage, so the run can climb toward the 17.21-bit ceiling unattended without ever
losing its best result.

## 7. Takeaways

- An L4 comfortably runs 8B unquantised for gradient-based prompt search — the memory trick
  is freezing the model and never materialising full logits.
- The outlier penalty is the interesting part: maximising a *mean* entropy is easy to game
  with a few high-entropy prompts; penalising `std` produces a genuinely **universal** prefix
  that flattens all six distributions together.
- Entropy-maximising prompts are non-linguistic. The winning prefix is gibberish
  (`"=======\n璋もっと新城OC…"`-style) — it is a controller for the softmax, not a sentence.
