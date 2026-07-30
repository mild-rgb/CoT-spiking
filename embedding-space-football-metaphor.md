# If every token were a football

A physical-scale metaphor for the embedding layer of `Qwen/Qwen3-4B-Thinking-2507`.
All figures measured 2026-07-28 from the model's own `embed_tokens` matrix
(151,936 tokens × 2,560 dimensions).

---

## The ruler

A token is a *point*. It has no size, so "how far apart are they" needs a chosen scale.

We set it so that **one football (22 cm) = the median nearest-neighbour distance**, which is
`0.9411` in embedding units. That fixes the conversion at:

```
1 embedding unit = 23.38 cm
```

Everything below follows from that single choice.

---

## Act I — the beach ball

```
                              embedding units    football scale
  your nearest neighbour            0.9411           22   cm   (by definition)
  closest 1% of all pairs           0.9669           22.6 cm
  TYPICAL pair (median)             1.4739           34.5 cm
  farthest 1% of pairs              1.7109           40.0 cm
  most distant pair in the vocab    1.9526           45.6 cm
  radius of the entire cloud        1.0422           24.4 cm
```

You are a football. Your nearest neighbour in the entire English language is one ball-width
away — near enough to touch.

Now reach out. **Every other token — all 151,935 of them — sits between 22 cm and 46 cm from
you.** Not most of them. All of them. The single most distant token in the whole vocabulary
is less than half a metre away. The complete vocabulary is a ball about **49 cm across** —
a beach ball.

Here is the part that should feel wrong: *you cannot build this*. A sphere 49 cm across holds
about **eight** footballs in our three-dimensional world, packed tight and touching. The
embedding layer fits **a hundred and fifty thousand** in the same volume, none overlapping,
all roughly equidistant.

That is what 2,560 dimensions buys. Not distance — **room**. Every direction you could move
is a direction nobody else is using.

Two consequences fall straight out:

**Everything is the same distance from everything.** The spread of pairwise distances has a
standard deviation of just **10.7% of the mean**; 98% of all pairs fall between 22.6 cm and
40 cm. There are no neighbourhoods, no suburbs, no far country. Your nearest neighbour is 64%
as far away as a total stranger.

**About 1,700 footballs stand in exactly the same spot.** 1.10% of tokens have a near-duplicate
at distance `0.0008` — on this scale, **0.2 mm apart**. Those are never-trained vocabulary
slots, sitting where initialisation left them. The model cannot tell them apart at layer 0;
they are the same vector wearing different numbers.

---

## Act II — who is actually standing next to you

```
  dolphin:  ' dolphins' @19cm,  ' Dolphin' @23cm,  '딲' @23cm,  '딉' @23cm
     wolf:  ' wolves'   @19cm,  ' Wolf'    @20cm,  'Wolf' @22cm, 'wolf' @24cm
    panda:  ' Panda'    @22cm,  '퍈' @24cm,  '팼' @24cm,  '퍠' @24cm
     crab:  ' Crab'     @22cm,  '캤' @24cm,  '찎' @24cm,  '쏸' @24cm
  quantum:  ' Quantum'  @13cm,  '量子' @15cm,  'Quant' @23cm, 'quant' @23cm
```

Your closest relative is **yourself with an S on the end**. After that, yourself with a capital
letter. Then — immediately, at the same 23 cm — a Korean syllable block you have never met.
`딲` stands exactly as close to `dolphin` as `Dolphin` does.

Your arm's reach is essentially empty. Tokens within one football-width:

```
  cat 9   the 6   banana 3   wolf 2   quantum 2   dolphin 1   crab 1   panda 0
```

`panda` has nobody at all.

---

## Act III — meaning has not been invented yet

```
  dolphin <-> wolf      30.0 cm
  dolphin <-> banana    31.1 cm
  dolphin <-> quantum   33.4 cm
  dolphin <-> cat       34.2 cm
  the     <-> quantum   30.7 cm
```

**`dolphin` is closer to `banana` than to `cat`.** Two animals sit further apart than an animal
and a fruit. `the` and `quantum` are neighbours.

This is what the metaphor is really for. The beach ball is not a map. Nothing has been sorted,
nothing grouped; no region belongs to animals or verbs or Korean. It is 151,936 footballs at
almost exactly the same distance from one another, in an arrangement carrying essentially no
semantic information at all.

Everything you think of as the model "knowing what a dolphin is" is built *later*, out of
context. None of it is here.

---

## Act IV — then the model throws you

The same 22 cm ruler, applied to the residual stream as it climbs the network
(answer-position vector length, averaged over 14 prompts):

```
  layer  0    0.18 m      <- inside the beach ball
  layer  6    5.4  m
  layer 12    9.3  m
  layer 18   10.9  m      <- where the mid-layer steering lens reads
  layer 24   20.5  m
  layer 30   58.4  m      <- the length of a football pitch
  layer 36   41.2  m      <- and back in again
```

The token begins **18 cm** from the origin — less than one ball-width. By layer 30 it is
**58 metres** out, a 320-fold expansion. Then the last six layers haul it back in by a third.

So the picture is not a crowded room where words shuffle politely between neighbours. The model
takes a football from a beach ball packed impossibly tight, **hurls it the length of a pitch**,
and catches it on the way down. Everything that makes `dolphin` mean *dolphin* happens out
there in the open field, not in the cupboard it started in.

```
   L0   *                                     18 cm
   L18       *                                10.9 m
   L30                          *             58.4 m
   L36                   *                    41.2 m
        |__ beach ball __|____ football pitch ____|
```

---

## Act V — which is why the junk tokens worked

This reframes the phase-2 result. We spent an evening asking how eight meaningless glyphs could
make the model answer *wolf*. The geometry answers: at the embedding layer those glyphs were
never meaningless **and** never meaningful — nothing there is. `𝙑ᠸㇽ𝖉🄸ᒡᠸ삵` sits 30-odd
centimetres from every word in the language, exactly as `wolf` does. It has the same access to
the pitch.

It also explains why the trigger-vs-real-word alignment peaked at **layer 32** — about 40 metres
out, near the top of the throw. That is where the junk trigger's trajectory and the real word's
trajectory converge. Not in the cupboard. In the air.

The ~1,700 untrained tokens sharing one 0.2 mm spot never get thrown at all. They are still in
the cupboard, and that is precisely why they make useless triggers.

---

## Appendix — raw measurements

```
vocab 151,936   d_model 2,560   (49,995,000 sampled pairs, 2,000 sampled NN queries)

nearest neighbour   min 0.0008   p1 0.0008   p5 0.5535   p25 0.7826
                    median 0.9411   p75 1.0594   max 1.3139
                    fraction with a near-duplicate (<0.01): 1.10%

random pair         p0.1 0.5359   p1 0.9669   p10 1.2536   median 1.4739
                    p90 1.6088   p99 1.7109   p99.9 1.7804
                    mean 1.4487   sd 0.1551   sd/mean 0.1070
                    min 0.0008    max 1.9526

radius from centroid   median 1.0422   (p1 0.1673, p99 1.2679)
```

**Reproducing:** load `model.embed_tokens.weight` directly from the safetensors shard (no need
for the full model), move to GPU as fp32, then compute chunked nearest-neighbour distances
against the full vocabulary and pairwise distances over a random sample. Note `torch.quantile`
refuses inputs above ~16M elements — subsample before taking quantiles.

Layer norms come from a separate pass requiring the full model, reading the answer-position
hidden state across 14 single-token cue prompts.
