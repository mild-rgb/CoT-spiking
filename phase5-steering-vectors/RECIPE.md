# Recipe — extracting a steering direction on Qwen3-8B

The protocol as actually run in phase 5 (2026-07-30, A100 40GB, bf16, transformers 5.14.1), with
the pitfalls that cost us. Every number below is measured on this backbone with the bridge vector.

Companion to `RESEARCH.md` (what the literature says) — this is what survived contact.

---

## Stage 0 — Fix the readout before you touch the vector

A metric you cannot trust makes every later stage unfalsifiable.

1. Write a topic word list. Keep a **strict** version (unambiguous: `bridge`, `bridges`, `viaduct`,
   `truss`, `girder`, `橋`, `桥`) and a **loose** one that adds polysemous terms (`span`, `arch`,
   `crossing`, `pier`).
2. Assemble a held-out prompt set spanning genres — advice, recommendation, technical explanation,
   creative, how-to, factual. 12 is enough to start.
3. **Measure the unsteered baseline first.** If the loose list fires on unsteered text it is
   measuring polysemy, not your topic.
   *Measured: 0.0% strict and 0.0% loose over 96 samples, mean perplexity 5.4.*
4. Score three numbers, never one: **on-topic rate**, **degeneracy rate** (a completion that is
   mostly one repeated token), and **perplexity under the unsteered model**. Rate alone ranks a
   `bridge bridge bridge` win at the top.

**Pitfall.** Prompts you used to pick the working setting by eye are not held out. Four of our
twelve came from phase 4's own test set; report the split.

---

## Stage 1 — Choose the contrast pair

The vector is a **last-token difference**, so whatever distinguishes the arms must live in their
final token.

1. **The differing token must be the final token.** Measured ‖v‖ at L6:

   | pair | ‖v‖ | verdict |
   |---|---|---|
   | `" bridge"` − `" cat"` | **47.6** | topic vs topic — use this |
   | `" bridge"` − `" "` | 71.3 | negative arm is a bare sink, see Stage 2 |
   | `"I love bridges."` − `"I love cats."` | **7.1** | both end in `"."` — almost no signal |
   | `"I talk about bridges constantly"` − `"I do not talk about bridges constantly"` | 12.7 | both end in `constantly` — encodes **negation**, not topic |

2. **Both arms must tokenize to equal length**, or their final tokens sit at different positions
   and positional effects do not cancel.
3. **The negative arm should be a real content word** of similar concreteness — it acts as the
   matched baseline whose shared structure cancels.

**Pitfall.** The negation pair fails in a way that looks like success until you read the output:
the model says *"I love discussing the importance of the topic"* — assertion delivered, topic
missing.

---

## Stage 2 — Move the differing token off position 0 ⚠

**The pitfall that invalidated phase 4's layer curve.** Single-token prompts put the topic token on
Qwen3-8B's attention sink, where massive activations swamp everything.

1. **Diagnose.** For your pair, print per layer: ‖h(p+)‖, ‖h(p−)‖, `cos(h+, h−)`, ‖v‖, and the
   share of energy in the top-3 dimensions.

   | L | ‖h+‖ | ‖h−‖ | cos | ‖v‖ | top-3 share |
   |---|---|---|---|---|---|
   | 6 | 44.7 | 46.6 | 0.458 | 47.6 | 20.1% |
   | **7** | **10457.5** | **11763.3** | **0.99995** | **1310.5** | **99.9%** |
   | 24 | 10897.5 | 12203.1 | 0.99995 | 1310.4 | 99.9% |

2. **Red flags, any one of which invalidates the vector at that depth:**
   - ‖h‖ jumping by ~200× between adjacent layers
   - `cos(h+, h−) > 0.999` — the two prompts are the same vector
   - top-3 dimensions holding >99% of the energy (here: 2276, 233, 4081)
   - **‖v‖ constant across layers** — a semantic direction does not have a depth-invariant norm

3. **Fix: a shared context prefix**, so the differing token is not at position 0.
   `"The word is bridge"` − `"The word is cat"` → final token at position 3. Assert equal length.

4. **Validate the fix against the old form.** `cos(v_ctx, v_bare)` per layer:

   | L | 2 | 4 | 6 | **7** | 8 | 12 | 16 | 24 | 35 |
   |---|---|---|---|---|---|---|---|---|---|
   | cos | 0.819 | 0.883 | 0.759 | **0.037** | 0.030 | −0.005 | 0.002 | −0.062 | −0.060 |

   The fixed vector must **agree with the bare one where the bare one is valid**. If it does, you
   have a correction; if it doesn't, you have a different object. The collapse point marks the
   boundary — here, L7.

5. **Sanity check the fixed vector:** ‖v‖/‖h‖ should sit in a sane band at every depth
   (measured 0.65–1.14 across L2–L35), not spike or vanish.

---

## Stage 3 — Injection

1. **Skip position 0.** Adding to the sink collapses generation into `"said said said"`.
2. **Strength in scale-free units:** `s = ‖added‖ / mean non-sink ‖h‖` at that layer, so a layer
   sweep compares. `alpha = s · ‖h‖ / ‖v‖`.
3. **Hold the vector on during generation**, not just the prefill, unless you are deliberately
   testing the single-edit (Function Vector) family — which was inert here across 4 pairs × 3
   layers × 3 strengths.

---

## Stage 4 — Sweep layer × strength, and sweep strength *finely*

**The strength window is narrow and layer-dependent.** A coarse grid reads as a dead layer.

*Measured: at L8, s=0.8 gives 20.8% and s=1.2 gives visible nothing — but s=1.0 gives **80.2%**.
A greedy n=1 sweep over {0.8, 1.2} concluded L8 was dead. It is not.*

| | s=0.6 | s=0.8 | s=1.0 |
|---|---|---|---|
| L8 | 0.0% · ppl 6.5 | 20.8% · 13.5 | 80.2% · 30.2 |
| L12 | 0.0% · 6.5 | 6.2% · 9.4 | 67.7% · 14.5 |
| L16 | 5.2% · 10.2 | 57.3% · 14.2 | **94.8%** · 14.5 |
| L20 | 2.1% · 7.9 | 41.7% · 15.9 | — |

Read the **perplexity column**, not just the rate: L8 reaches 80.2% at ppl 30.2 (5.6× baseline)
while L16 reaches 94.8% at ppl 14.5 (2.7×). Equal rates are not equal quality.

**Do not select on rate alone.** Rank on the frontier of (rate, degeneracy, perplexity).

---

## Stage 5 — Controls

1. **Reversal.** Build the exact negation at identical magnitude. It should push *away* from the
   topic — that proves a signed semantic axis rather than generic disruption.
2. **Matched-norm random direction.** Tells you how much of the effect is your topic and how much
   is "a large perturbation at this layer". Phase 4 never ran this one.
3. **Negative-arm sensitivity.** Build `topic − X` for 6–8 different negatives and take the
   pairwise cosines. Tight → the negative is inert and one pair suffices. Scattered → the choice
   is load-bearing and you need the CAA mean over many negatives.

---

## Stage 6 — Generalisation

Steerability is highly variable per input and on some datasets ~50% of inputs get the *opposite*
behaviour (Tan et al. 2407.12404). So:

- Report a **per-prompt breakdown**, not a pooled rate — a setting that wins only on two prompts
  should be visible as such.
- Watch for prompts semantically adjacent to the topic in ways the word list cannot see. Ours:
  `"how do I make friends in a new city?"` sits next to the idiom *building bridges*, and
  `"what's the difference between TCP and UDP?"` lives where *bridge* is standard terminology.
  Neither fired unsteered, but both could score as hits without topical steering having occurred.

---

## The short version

```
0.  baseline the metric        (strict + loose word lists, unsteered rate must be ~0)
1.  pair: differing token LAST, arms equal length, both arms real words
2.  check the sink             (cos>0.999 / top-3>99% / constant ‖v‖ = invalid depth)
    fix with a shared prefix, validate via cos(v_ctx, v_bare) where bare is valid
3.  inject everywhere but position 0, strength in units of non-sink ‖h‖, held on
4.  sweep layer x strength FINELY; rank on (rate, degeneracy, perplexity)
5.  controls: reversal, matched-norm random, negative-arm cosine spread
6.  per-prompt breakdown; flag topic-adjacent prompts
```
