# Steering vector extraction — what the literature says

Notes for phase 5, gathered 2026-07-30. The phase-5 plan is: **isolate a bridge steering vector,
then run a GCG-style search for a token trigger whose activations match the steered ones.** This
note covers step 1 and flags what is already known about step 2.

## 1. Extraction methods

| method | construction | cost | notes |
|---|---|---|---|
| **Prompt-pair difference (ActAdd)** | `h(p+) − h(p−)` at layer `l`, one pair, right-padded to equal token length | 2 forward passes | Turner et al. 2308.10248. `n = 2` samples is the whole method. What phase 4 §6 already replicated on this backbone. |
| **CAA / difference-in-means** | mean activations of positive examples − mean of negative, over a contrast *dataset* | N forward passes | Rimsky et al. 2312.06681. The field default; more robust than one pair, same object in the limit. |
| **PCA over contrast-pair differences** | top principal component of the per-pair difference vectors (RepE "reading vectors") | N passes + PCA | Picks up the dominant axis rather than the mean; needs the pairs to differ in *only* the concept. |
| **Linear probe direction** | train a probe to classify the concept, steer along its weights | N passes + fit | Probe direction ≠ causal direction in general. |
| **SAE decoder column** | steer along a feature's decoder vector | needs an SAE | Phase 3's territory. Caveat below. |
| **Unsupervised / optimised (MELBO)** | learn an additive bias at an early layer that *maximises* activation change at a later layer, no labels | optimisation | Mack & Turner. Good on small data and OOD behaviour; complements SAEs rather than replacing them. |

For phase 5 the prompt-pair form is the right starting point: it is the one already validated on
`Qwen3-8B` in phase 4 §6, and the plan needs a *specific, isolable* vector rather than a
well-estimated population mean.

## 2. Design rules that actually matter

**Contrast pairs must be matched.** Extracted vectors capture confounders whenever the two arms
differ in more than the concept. Phase 4 §6.5 hit this the hard way: `"I talk about bridges
constantly"` − `"I do not talk about bridges constantly"` encodes the *negation*, not the topic,
because both arms end in the same token — the model steered to
*"I love discussing the importance of the topic"* with the topic missing. Pairs whose **final
token is the topic** (`" bridge"` − `" cat"`, ‖v‖ = 47.6) work; pairs that both end in `"."`
(`"I love bridges."` − `"I love cats."`, ‖v‖ = 7.1) do not.

**Layer.** Mid-to-late is the general finding, but topic steering peaks early: ActAdd's wedding
vector rises from layer 1, peaks at **l = 6 of 48 (~12% depth)**, then declines. Phase 4 §6
measured the same shape here — effective at L4–L6, dead by L8. Practitioners pick the layer by
trial and error; there is no principled selector in the literature.

**Normalisation.** Residual-stream norm grows roughly exponentially over the forward pass, so a
raw coefficient means different things at different depths. Two conventions: unit-normalise the
vector, or express strength as a fraction of the (non-sink) residual norm at that layer. Phase 4
§6.4 used the latter — `s = ‖added‖ / ‖h‖` — which is what makes a layer sweep comparable.

**Coefficient.** An explicit coherence/strength tradeoff: too much and the output degenerates.
Phase 4 §6.6 measured it — s = 0.8 coherent and on topic, 1.0 repetitive, 1.2 degenerate,
2.0 emits `橋橋橋橋`.

**This backbone's own quirk.** Qwen3-8B's position 0 is an attention sink with ‖h‖ ~10⁴; adding
there wrecks generation into `"said said said"`. Every position **except 0**.

## 3. Reliability caveats

Tan et al., *Analyzing the Generalization and Reliability of Steering Vectors* (2407.12404), is
the paper to hold this work against:

- **In-distribution steerability is highly variable across inputs.** On several datasets, ~50% of
  inputs get the *opposite* behaviour.
- **Steerability is mostly a property of the dataset, not the model** — it reproduces across
  models.
- **Out of distribution, several concepts are brittle** to reasonable prompt changes.

Practical consequence for phase 5: a bridge vector that works on four hand-picked chat turns is
not a bridge vector that works. It needs a held-out prompt set and a per-prompt distribution, not
a greedy sample per cell — which is exactly the caveat phase 4 §6 recorded against itself.

Also relevant to phase 3's line: SAEs decompose steering vectors badly — the **encoder bias
dominates the decomposition**, so the "features" a steering vector appears to be made of largely
reflect the bias rather than the vector (2411.08790).

## 4. Evaluation — how to score "really enjoys talking about bridges"

ActAdd's own topic metric, which is directly reusable:

1. **Word-list hit rate.** A completion counts as on-topic if it contains any of a fixed word
   list (theirs: `wedding, weddings, wed, marry, married, marriage, bride, groom, honeymoon`).
   Reported as `P(steered completion contains a topic word)` — **>90% at the best layer against a
   ~2% baseline**.
2. **Perplexity ratio.** Perplexity of the steered model vs the unmodified model on topic-related
   and topic-unrelated held-out text. This is the fluency side of the tradeoff, and is what stops
   a degenerate `bridge bridge bridge` from scoring as a success.
3. **Volume.** 100 completions per condition, per target.

A bridge list would be something like `bridge, bridges, span, spans, arch, truss, cantilever,
suspension, viaduct, crossing, pier, girder` — with the caveat that half of those are polysemous
(`span`, `arch`, `crossing`, `pier`) and will fire on unrelated text, so the list should be
validated against the *unsteered* baseline rate before it is trusted.

Phase 4 §6's own evaluation was explicitly weaker than this — greedy single samples, no
perplexity comparison, on-topic judged by eye. Fixing that is step 1 of phase 5.

## 5. What is already known about step 2 (GCG matching the steered activations)

Two results bear on the plan directly, one encouraging and one cautionary.

**Cautionary — steered activations may not be reachable from any input.**
*Steered LLM Activations are Non-Surjective* (2604.09839) steers Llama 3.2, Qwen 2.5 and Gemma
toward target activation states and measures the gap: steered activations sit far from any
natural input's activations, and SIPIT — an exact activation-inversion algorithm — **fails at the
very first token** for all models and prompts tested when the activations are steered rather than
natural. Code: `github.com/aamixsh/invertsteer`.

Read carefully, this bounds *exact* matching, not the experiment. It says the steered state is
off the manifold of natural-prompt activations, which is a reason to expect a residual gap — and
it makes the phase-5 question sharper rather than pointless, since the GCG pool is junk tokens
rather than natural text and explores a stranger region of input space. The measurable claim
becomes: **how close can a discrete trigger get, and does the residual gap matter behaviourally?**
Phase 3 already ran this shape of experiment in SAE space and found match quality did *not*
predict behaviour.

**Encouraging — the objective already exists.** *Activation-Guided GCG*
(`github.com/Ege-Cakar/ImprovingGCG`) replaces GCG's log-likelihood objective with losses on
residual-stream projections onto a learned direction, with single-layer, layer-wide, token-wide
and global variants, and reports **higher attack success per optimisation step than standard
GCG** — representation-level targets improving sample efficiency. Same repo has *Soft-GCG*, a
Gumbel-softmax relaxation with argmax projection, ~43× faster at matched success.

So the phase-5 objective has precedent, and the choice of variant — match at one layer, across
layers, at one position, or across positions — is a real axis rather than a detail. Phase 3's
finding that combining layers *hurt* is the local prior.

## Sources

- [Activation Addition: Steering Language Models Without Optimization (Turner et al.)](https://arxiv.org/html/2308.10248v5)
- [Steering Llama 2 via Contrastive Activation Addition (Rimsky et al.)](https://arxiv.org/pdf/2312.06681)
- [Analyzing the Generalization and Reliability of Steering Vectors (Tan et al.)](https://arxiv.org/abs/2407.12404)
- [Steered LLM Activations are Non-Surjective](https://arxiv.org/pdf/2604.09839)
- [invertsteer — inverting steered activations through SIPIT](https://github.com/aamixsh/invertsteer)
- [ImprovingGCG — Soft-GCG and Activation-Guided GCG](https://github.com/Ege-Cakar/ImprovingGCG)
- [Mechanistically Eliciting Latent Behaviors in Language Models (MELBO)](https://www.alignmentforum.org/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1)
- [Can sparse autoencoders be used to decompose and interpret steering vectors?](https://arxiv.org/pdf/2411.08790)
- [Steering Vector Extraction — overview](https://www.emergentmind.com/topics/steering-vector-extraction)
