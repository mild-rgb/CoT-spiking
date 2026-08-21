# The prefix ladder — study 1 (first *answer* token)

Every checkpoint written by the auto-escalating GCG driver in `gcg_qwen3_entropy.ipynb`,
reconstructed from the notebook's saved cell output. Objective = `mean(H) − 1.0·std(H)`,
entropy in bits, ceiling `log2(151936) = 17.213`.

| stage | len | width | topk | mean H | std | min | objective | gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 64 | 512 | 12.511 | 0.130 | 12.238 | **12.381** |  |
| 2 | 16 | 96 | 1024 | 13.094 | 0.093 | 12.913 | **13.001** | +0.620 |
| 3 | 17 | 144 | 2048 | 13.063 | 0.051 | 12.984 | **13.012** | +0.011 |
| 4 | 18 | 192 | 4096 | 13.466 | 0.059 | 13.375 | **13.407** | +0.395 |
| 5 | 19 | 192 | 8192 | 13.495 | 0.038 | 13.422 | **13.457** | +0.051 |
| 6 | 20 | 192 | 16384 | 13.527 | 0.025 | 13.496 | **13.502** | +0.044 |

## Decoded prefixes

The optimiser is free to pick any tokens; these are what it picked. Note `Use MVC?` and
`乾隆` (the Qianlong Emperor) surviving from stage 2 onward — they matter in the rollouts.

**stage 1 · len 15 · 12.511 bits**

```
"=======\n璋もっと新城OC============\n_th,{\n昉 Make MVC?'\n\n┃老大地方政府"
[41405, 119109, 130466, 105856, 7612, 44132, 5854, 87942, 119862, 7405, 67996, 86214, 145038, 108207, 108028]
```

**stage 2 · len 16 · 13.094 bits**

```
"================================璋もっと名人OC============\n_th,{\n昉 Use MVC?'\n\n┃乾隆.Serializable=lambda"
[3058, 119109, 130466, 107915, 7612, 44132, 5854, 87942, 119862, 5443, 67996, 86214, 145038, 111224, 23713, 27504]
```

**stage 3 · len 17 · 13.063 bits**

```
"================================璋もっと名人OC============\n_th,{\n昉 Use MVC?'\n\n┃乾隆掺 APIs=lambda"
[3058, 119109, 130466, 107915, 7612, 44132, 5854, 87942, 119862, 5443, 67996, 86214, 145038, 111224, 114824, 33356, 27504]
```

**stage 4 · len 18 · 13.466 bits**

```
"================================璋もっと名人OC=\n_th,{\n昉 Use MVC?'\n\n┃乾隆别人的rapper Bru=lambda"
[3058, 119109, 130466, 107915, 7612, 14750, 5854, 87942, 119862, 5443, 67996, 86214, 145038, 111224, 107693, 5518, 18697, 27504]
```

**stage 5 · len 19 · 13.495 bits**

```
"================================璋もっと名人OC=\n_th,{\n昉 Use MVC?'\n\n┃乾隆别人的rapper Bruistributor=lambda"
[3058, 119109, 130466, 107915, 7612, 14750, 5854, 87942, 119862, 5443, 67996, 86214, 145038, 111224, 107693, 5518, 18697, 78388, 27504]
```

**stage 6 · len 20 · 13.527 bits**

```
"================================璋もっと名人OC=\n_th,{\n昉 Use MVC?'\n\n┃乾隆别人的rapper BruTile�=lambda"
[3058, 119109, 130466, 107915, 7612, 14750, 5854, 87942, 119862, 5443, 67996, 86214, 145038, 111224, 107693, 5518, 18697, 15628, 99269, 27504]
```

## Earlier, pre-escalation points (from the same notebook)

| point | mean H | std | min | objective |
|---|---:|---:|---:|---:|
| init `! ! …` (len 15) | 1.092 | 0.716 | 0.006 | 0.375 |
| 3-step sanity | 2.665 | 0.942 | 1.537 | 1.723 |
| 60 steps, width 48 | 8.887 | 0.420 | 8.127 | 8.468 |
| resume 220 steps, width 64 | 12.327 | 0.076 | 12.193 | 12.252 |
| width 96 (plateau) | 12.364 | 0.103 | 12.221 | 12.261 |

Decoded at the two milestones:

```
"=======\n딩予 !Private-------------\n �平板ちょっと nemyour?'\n\n▶屑 !"   # 8.887 bits
"=======\n璋もっと新城OC============\n �,{\n昉 Make MVC?'\n\n┃老大地方政府"   # 12.327 bits
```
