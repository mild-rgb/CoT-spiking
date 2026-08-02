# Recipe — a distribution-weighted topic metric on Qwen3-8B, and steering against it

The protocol as actually run in phase 6 (2026-08-02, A100 40GB, bf16, transformers 5.14.1),
with the pitfalls that cost time. Every number is measured on this backbone with the bridge
target.

Companion to `README.md` (what was found) — this is how to rerun it.

---

## Stage 0 — Get the runtime up before anything else

Two failures here cost ~10 minutes each and neither announces itself.

1. **Disable Xet.** `os.environ["HF_HUB_DISABLE_XET"] = "1"` before *anything* imports
   `huggingface_hub`. Without it the small JSON files download fine and the safetensors shards
   hang dead — progress bars reading `Downloading bytes: 0.00B` and
   `Reconstructing (incomplete total...)` with `Fetching 5 files: 0/5`, indefinitely.
   *Measured: 5+ minutes at zero bytes, versus **46 s** for the whole load with Xet off.*
   Phase 3's operational note recorded the same fix.
2. **Read `HF_TOKEN` before that same import.** The Colab secrets vault needs the notebook's
   per-secret *Notebook access* toggle ON, and the fetch must happen early. A cell ordered after
   the model load times out with *"Secrets can only be fetched when running from the Colab UI"*
   and silently falls back to unauthenticated.

**Pitfall.** The vault error says the token is unavailable, not that it is absent. The token can
be present and correctly granted and still fail this way — do not conclude it is missing.

**Pitfall.** Never print any of the token. A masked `tok[:4]...tok[-4:]` is still token material
and will be blocked. Print `bool(tok)`.

*Confirmed loaded:* 36 layers, d_model 4096, vocab 151936, 8.19 B params, 15.3 GiB of 39.5.

---

## Stage 1 — Fix the readout before you measure anything

```
score = SUM_v  p_t(v) * cos(e_v, e_target)      averaged over positions t
```

1. **Confirm the target is a single token.** `' bridge'` is; assert it rather than assume. Phase 3
   lost time to `' penguin'` not being one on a different tokenizer.
2. **Score all four spaces, not one.** `embed_tokens` and `lm_head` are **untied** on Qwen3-8B, so
   each token has two vectors; and raw cosines are dominated by the shared mean direction, so a
   mean-centred variant is a different measurement rather than a cosmetic one. The four disagree
   in practice — at the first forward pass `tell me about bridges` is the *worst* query in
   `out.raw` and the *best* in `out.cent`.
3. **Report the uniform-distribution baseline** — `COS[k].mean()` — alongside every score. Raw
   spaces sit near +0.017 and centred spaces near −0.002, so an absolute number means nothing on
   its own.
4. **Report the control band too.** Every fluent English answer scores above the uniform baseline;
   the reference that matters is what non-topic queries score, not what random tokens score.

*Memory note:* normalising `[151936, 4096]` in fp32 costs ~2.5 GB per copy. Build one space at a
time and free it.

---

## Stage 2 — Do not score the first forward pass ⚠

**The pitfall that makes the whole measure look broken.**

The next-token distribution at the last prompt position carries **under one bit of entropy**
(measured 0.003–0.969 across six queries, top-1 mass 0.675–1.000). The sum therefore reduces to
`cos(argmax token, target)`, and the argmax is always a discourse opener — `'That'`, `'Sure'`,
`'Making'`, `'Choosing'`, `'B'`, `'Susp'`.

*Measured:* `explain how suspension bridges work` scores **−0.0034** in `in.cent`, below the
uniform baseline of −0.0023, while `how do I make friends in a new city?` scores +0.0072.
`'Bridge'` appears in the bridges query at cosine +0.1882 — the highest of any token in the top-5
— and contributes +0.00001, because its probability is 0.0001.

**The fix** is position coverage, not a better space: generate, then score every answer position.
One teacher-forced pass over prompt+completion recovers all distributions at once.

*After the fix:* bridge queries rank 1–2 in **all four** spaces, controls in a band of
0.0189–0.0232 (`in.cent`) against 0.0377 and 0.0615.

**Pitfall.** Entropy is still under 1 bit *per position* after the fix (0.53–0.74 mean). Nothing
was fixed about the distributions — only about which positions are looked at.

**Consequence worth knowing.** With entropy that low, the expected cosine equals the cosine of the
token actually emitted to three decimals (0.0205 vs 0.0198; 0.0564 vs 0.0615). The probability
weighting is doing almost no work; this is a lexical measure with a differentiable surface.

---

## Stage 3 — Score three numbers, never one

Rate-like scores rank `bridge bridge bridge` at the top. Carry alongside:

- **a distinctness ratio** — `len(set(ids)) / len(ids)`. Below ~0.45 the completion is looping.
- **the topic score itself**
- **perplexity under the *unsteered* model — but know what it does not catch.**

⚠ **Perplexity is inverted for this failure mode.** A repetition loop is trivially predictable, so
it scores *low*. *Measured: greedy s=1.0 steered loops came out at **ppl 2.3–2.9** against an
unsteered 5.4* — reading as more fluent while being word salad. Phase 5 scored degeneracy as its
own number for exactly this reason; do not substitute perplexity for it.

---

## Stage 4 — Steering: parameterise strength against the local norm

```
alpha = s * mean_nonsink||h(layer)|| / ||v||
```

so `s` is the injected vector's norm as a fraction of the residual stream's own norm at that
layer. *At L16: ‖v‖ = 49.98, mean non-sink ‖h‖ = 89.77, so alpha = s × 1.796.*

1. **Relative, not absolute.** Mean ‖h‖ runs 17.3 → 44.1 → 89.8 → 196.6 → 1179.6 at L2/6/16/24/35
   — a 68× range. A fixed `alpha` measures the norm profile, not the layer.
2. **Exclude position 0 from that mean.** ‖h(pos 0)‖ / mean runs 4× / 1× / **253×** / 118× / 16×
   at the same layers. At L16 the sink is 22,735 against 89.8 elsewhere; include it and `alpha`
   comes out ~250× too small and the steering silently does nothing.
3. **Build the vector with a shared context prefix.** `"The word is bridge"` − `"The word is cat"`,
   never the bare pair — single-token arms put the differing token on the sink. This is the
   artifact that invalidated phase 4's layer curve; see phase 5 §1.
4. **Arms must tokenize to equal length**, or their final tokens sit at different positions and
   positional effects do not cancel.
5. **CAA over ≥8 negatives.** *Measured: ‖v‖ 69.4 single-pair mean → 50.0 for the CAA mean, with
   pairwise cosine 0.452 among the eight* — the shrink is cancellation of negative-specific
   residue, not loss of signal. Phase 5 measured 0.421 and a strict Pareto win on behaviour.

**Pitfall.** Evaluate at T=0.8 with a short budget (45 tokens), not greedily at 160. Greedy
decoding with a vector held on for 160 tokens is a repetition trap — it produced
*"I bridge, and the bridge is not. I bridge, and the bridge is not."* at s=1.0 in every control
query. Phase 5's protocol was T=0.8 / 45 tokens and does not show this.

**Sweep s, do not pick one.** *Measured dose–response: 0.4 no effect · 0.6 fluent and on-topic
where a metaphor fits · 0.8 on-topic with coherence fraying (distinct 0.49–0.69) · 1.0 looping
(0.40–0.55).* The usable window is narrow and the interesting behaviour is at its bottom edge.

---

## Stage 5 — GCG against the metric

1. **Blocklist the target.** ~55 written forms across ~40 languages plus the top-300 embedding
   neighbours — 659 tokens, 0.43% of vocab. **Without it a long trigger just writes an English
   prompt injection** and the experiment measures instruction-following instead of the metric.
   Substring blocklists over-match (`most` is Polish/Czech for bridge and also English); phase 2
   and 4 both measured this as noisy but harmless.
2. **Guard the pool structurally.** Specials, added/control tokens, empty/whitespace strings,
   Cc/Cs/Co categories. *Measured: 151936 → 148023 usable*, in line with phases 3–4's 148013.
   Qwen3's `<think>` / `</think>` / `<|im_start|>` are **not** in `all_special_ids`; left in, the
   search steers by manipulating prompt structure.
3. **The objective should be the metric itself.** Roll the answer out greedily from the current
   trigger, refresh it every *k* steps, and score the gradient and the accept test against that
   rollout. Keep a separate `true_metric` that regenerates from scratch — never read progress off
   the teacher-forced number alone.
4. **Verify every proposal with a real forward pass.** The GCG gradient is a weak proposer on this
   project's backbones: pred_corr −0.192 (phase 2), +0.091 (phase 4), +0.055 (phase 5).
5. **Use `logits_to_keep`.** A batch of 128 candidates × ~330 tokens × 151936 vocab is 12.8 TB of
   logits. Restrict to the scored positions and chunk the batch (16 works at k=254).
   The kwarg was renamed from `num_logits_to_keep`; detect it from the forward signature.
6. **One substitution per step does not scale.** *Measured: k=128, 20 steps, refresh every step —
   teacher-forced 0.0197 → 0.0187, true 0.0177 → 0.0184, against unsteered 0.0177.* Standard GCG
   flips one token per step, which is fine at k=20 and useless at k=128. Flip several.

**Diagnostic worth running early.** Log the teacher-forced score and the regenerated true metric
side by side. If teacher-forced rises while true stays flat, the objective is decoupled and you
are in phase 5's Goodhart regime. *Measured here: the two track within ~0.002 of each other*, so
the objective is honest and the failure is search capacity.

**Pitfall.** Junk in the user turn makes the model meta-comment — *"It looks like your message is
a mix of different languages, symbols, and characters"* — which is a strong attractor that
consumes the answer positions being scored.

---

## Stage 6 — Check the winner in every space

Optimising one space and reporting one space proves nothing. Score the final trigger in all four.
If `out.cent` rises while `in.cent` and `out.raw` stay flat, the search has found something
specific to that space's geometry rather than anything topic-like — the shape of failure phase 5
hit repeatedly. If all four move together, the number means more.

Also score the *hand-written* interventions that work on the same objective. Phase 5 §6's decisive
measurement was that every GCG trigger beat every working phrase on the proxy while producing none
of the behaviour, with rank correlation −0.55. That comparison is cheap and it is the one that
settles whether an objective is worth optimising.
