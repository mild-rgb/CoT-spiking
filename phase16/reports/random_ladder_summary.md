# Random len-20 suffix entropy distribution (n=400)

**Recorded from console output; `random_ladder_len20.json` (with the 400 entropies and the rung ids)
was still on the Colab VM when the MCP connection dropped and has NOT been downloaded.**

Qwen3-8B, first-thought-token entropy induced by a random 20-token suffix, uniform over ids
[0, 151643) (below the special-token range), seed 123, mean over the 6 probe queries.

    min = 0.005   median = 0.037   mean = 0.063   max = 0.981  bits

Quantiles (0, .25, .50, .75, .90, .99, 1.0):

    0.005  0.026  0.037  0.055  0.105  0.539  0.981

Selected rungs: rand0=0.005, rand99=0.026, rand199=0.037, rand299=0.055, rand379=0.201, rand399=0.981

## Why this matters

1. It replaces the original's n=10 random-prefix baseline with n=400 at the suffix position.
2. Random *suffixes* barely perturb the first thought token at all — median 0.037 bits against a
   baseline of ~0.002 and a GCG-reachable 7.2. The original measured random 15-token *prefixes* and
   got mean 0.84 / max 1.54 bits, so position matters: a random prefix perturbs far more than a
   random suffix.
3. **Design consequence.** The planned matched-entropy comparison (random vs GCG base at equal
   entropy, to separate "high entropy" from "already-optimised region") is impossible this way: the
   random arm spans 0.005-0.98 bits and the GCG ladder's lowest non-cold rung is 1.24. No overlap.
   The replacement control is the corruption arm — take the 7.214-bit optimised suffix and randomise
   k of its 20 tokens, which sweeps the same entropy range with the optimised arrangement destroyed.
   That cell was written but never ran.
