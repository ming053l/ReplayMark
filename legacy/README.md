# V1: post-hoc substitution and global two-phase generation

Archived, not deleted. These are the measurements that forced V2's design, and each rules
out a direction that would otherwise look reasonable.

| module | what it was | why it is here |
|---|---|---|
| `core.py` | first carrier/probe embedder | guidance table went stale as tokens were rewritten |
| `carrier.py` | carrier masked in both arms, staleness-free | sound, but substitution is priced out |
| `shared.py` | shared ablation patterns, detection in `L` forwards | the efficiency idea V2 keeps |
| `select.py` | probe-conditioned carrier selection by leverage | the only selector to beat chance, still 8x short |
| `gentime.py` | global two-phase generation | schedule alone costs x1.98 before any watermark |
| `blocklocal.py` | block-local with an *averaging* statistic | superseded by the count statistic in `blockmark.py` |

`gentime.py` in particular still modifies token choice (`log p + lambda * g`, then argmax)
and defers half the span to a second phase. It is **not** the order-only method; that is
`basinmark/blockmark.py`.
