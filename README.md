# BasinMark — watermarking diffusion LLMs via keyed re-denoising contrast

**Status: the detector works and the embedding channel does not pay for itself.** The
keyed re-denoising contrast carries real signal with a finite-sample-valid null, but
post-hoc token substitution costs far more text quality than published dLLM watermarks
pay for the same detection (see the baseline anchor below). Numbers here are measured on
this machine unless explicitly attributed to a paper.

> **Correction notice (this revision).** Two claims in the previous revision were
> overstated and have been withdrawn:
>
> 1. *"Survives denoising-smoothing."* The attack implemented was `mask -> one forward ->
>    fill everything back at once`, i.e. **one-step masked reconstruction, not reverse
>    diffusion**. Iterative re-denoising, where tokens recovered early become context for
>    the rest, was never run. It is running now (`exp/13_attacks_v2.py`, 1/4/8/16/32
>    steps).
> 2. *"Exact finite-sample p-value"* alongside `p = 5.5e-10`. The exact enumeration only
>    covers `M <= 20` blocks; every headline configuration has 48-192 blocks and silently
>    fell through to a **Gaussian tail approximation**. Measured false-positive rate was
>    **0.145 at a nominal 0.10** (`results/null.json`) — anti-conservative, so
>    extrapolating it to `1e-10` was not supportable.
>
> A third correction follows from the second: reporting *mean z* rather than TPR at a
> controlled FPR overstated robustness. The headline "survives at rho=0.30" had
> `z = 2.31`, which under the rigorous bound is **p = 0.069** — not detection at any
> usable operating point. Every p-value below is now `exp(-z^2/2)`, Hoeffding's bound for
> Rademacher sums: a valid finite-sample upper bound with no distributional assumption,
> and roughly 10x weaker than the Gaussian tail it replaces.

Base model: `GSAI-ML/LLaDA-8B-Instruct`, fp16, single TITAN RTX (sm75, 24 GB).

---

## Idea

Existing dLLM watermarks hide the signal in token choice (Gloaguen et al., ICLR 2026),
sampling randomness (Bagchi et al.), unmasking order (dgMARK, ICML 2026), or global
sequence statistics (Global Sketch). BasinMark instead asks what the *final text does
when you push it back through the diffusion operator*:

> re-corrupt the text with a secret key, re-denoise, and read the response.

The watermark lives in the text's **reconstruction behaviour**, not in its tokens.

## The statistic

For probe `j`, the key derives three disjoint position sets over the generated span:
`S_j` (probe), `D_j^0`, `D_j^1` (two equal-size context ablations). Two corruptions:

    C^0 = mask(S_j u D_j^0)        C^1 = mask(S_j u D_j^1)

Log-probs are read **only on `S_j`, in both arms**:

    delta_i = log p(y_i | C^1) - log p(y_i | C^0),   i in S_j
    Delta_j = mean_i delta_i

Both arms score the *same tokens*; only the ablated context differs. The watermark
encodes **which half of its own context the text relies on to reconstruct itself**.

### The null

`D_j^0` and `D_j^1` are drawn exchangeably, so swapping them negates `Delta_j`. Under
H0 (text not produced with key K) `Delta_j` is therefore symmetric about 0, giving

    sign(Delta_j) ~ Bernoulli(1/2)   ->   sign-matches ~ Binomial(M, 1/2)

No calibration corpus and no model-specific null estimate are needed. **What is exact
depends on how the statistic is aggregated:**

| statistic | validity |
|---|---|
| sign count, `Binomial(M, 1/2)` | exact |
| sign-flip enumeration over `2^M` patterns | exact, affordable only for `M <= 20` |
| Monte-Carlo randomization, add-one corrected | valid, but **cannot resolve below `1/(1+n)`** |
| `exp(-z^2 / 2)` (Hoeffding) | valid finite-sample **upper bound**, no resolution floor |
| Gaussian tail `norm.sf(z)` | **not valid** — measured FPR 0.145 at nominal 0.10 |

Reported p-values use exact enumeration where affordable and the Hoeffding bound
otherwise. Deployment also wants the null for a *fixed* key over many documents, not
averaged over keys; `exp/14_fixedkey_null.py` measures per-key and worst-key FPR, and
`SharedMark(..., nonce=...)` derives a per-document key `HMAC(K, nonce_d)` so that each
document is an independent key draw.

**Measured** (96 probes, unwatermarked LLaDA text): `mean Delta = +0.0136`,
`P(sign > 0) = 0.521`. Consistent with the theory. See `logs_clean/pilot.log`.

> A first version of this design compared energies over *different* token sets
> (`E(y,C^0) - E(y,C^1)` with `C^0`, `C^1` masking disjoint positions). That statistic
> is dominated by whether the key happened to select function words or content words —
> several nats of text-dependent bias, and no usable null. It was discarded.

### Embedding guidance is free

Every position in `S_j` is masked in **both** arms, so `y_i` never enters either
conditioning context. The guidance is therefore a table lookup

    g_i(v) = log p(v | C^1) - log p(v | C^0)

and **two forward passes give the full guidance table over every probe position and the
entire vocabulary**. No per-candidate lookahead. Embedding becomes
`generate -> keyed re-mask -> biased re-denoise`, an operation an autoregressive model
cannot perform on its own output.

Cost: 3 forwards per probe per embedding round; 2 per probe for detection.

---

## What is measured so far

| claim | status |
|---|---|
| Null symmetry | **verified** — `P(sign>0)=0.521` over 96 probes. But the *aggregated* p-value was a Gaussian approximation with measured FPR 0.145 at nominal 0.10; replaced by Hoeffding's bound, revalidation running |
| Guidance table costs 2 forwards, not \|V\| | **verified** — 0.3 s/probe on one TITAN RTX |
| Probe positions are controllable | **partly** — depends strongly on decoding temperature and quality budget `tau` (see below) |
| Bits can be set end-to-end | **partly** — 0.844 bit accuracy at 16 bits, z = +3.95 vs +0.05 unwatermarked, 7.7 % of tokens changed |
| Quality cost | **the binding constraint** — the tuned operating point costs 4.8x perplexity; nothing with acceptable quality currently detects |
| Robust to attack | **unresolved** — only one-step reconstruction measured, and at rho=0.30 that is already p_bound=0.069; iterative re-denoising running |
| Beats/complements baselines | **not started** |

### Controllability sweep (`exp/02_sweep.py`, `logs_clean/sweep.log`)

Fraction of probes whose sign can be set in one pass, over 96 probes:

| temperature | probe_rate | tau (nats) | settable | positions with >1 admissible token |
|---|---|---|---|---|
| 0.0 | 0.12 | 1.5 | 0.34 | 0.09 |
| 0.0 | 0.40 | 6.0 | 1.00 | 0.38 |
| 1.0 | 0.25 | 3.0 | 0.89 | 0.41 |
| 1.0 | 0.25 | 6.0 | 1.00 | 0.58 |

Greedy decoding destroys controllability — 91 % of positions admit no alternative
token at all. The baselines' `temperature = 0.8` is the right regime.

### Where core.py fails, and why (`exp/03c_lambda.py`, `logs_clean/lam.log`)

Embedding maximises `log p_base(v) + lam * s_j * g_i(v)` subject to a hard cap `tau` on
how far below the denoiser's own argmax a substitution may fall. 16 probes over a
192-token span, disjoint probe sets, `|S_j| = 12`:

| lam | tau | s*Delta before -> after | bit accuracy | tokens changed | cost (nats) |
|---|---|---|---|---|---|
| 1 | 4 | — | 0.542 | 0.16 | 0.05 |
| 3 | 4 | +0.063 -> +0.129 | 0.667 | 0.24 | 0.30 |
| **3** | **6** | **+0.063 -> +0.231** | **0.771** | 0.30 | 0.43 |
| 10 | 4 | +0.063 -> +0.031 | 0.604 | 0.29 | 0.34 |
| 10 | 6 | +0.063 -> +0.170 | 0.646 | 0.39 | 0.87 |
| 30 | 4 | +0.063 -> +0.096 | 0.500 | 0.36 | 0.69 |
| 30 | 6 | +0.063 -> +0.059 | 0.458 | 0.53 | 1.57 |
| 1e6 | 4 | +0.063 -> +0.158 | 0.542 | 0.46 | 1.13 |
| 1e6 | 6 | +0.063 -> +0.205 | 0.688 | **0.83** | **3.24** |

Bit accuracy is **non-monotonic in the guidance weight**: it peaks at `lam = 3` and
falls *below chance* at `lam = 30`. At the hard-argmax limit (`lam = 1e6`, `tau = 6`)
the embedder rewrites 83 % of the tokens and pays 3.24 nats each, and still only
reaches 0.688. **Guidance strength is not the binding constraint.**

The cause is **guidance-table staleness**. `g` is computed on the current text, then a
third to four fifths of the tokens are rewritten at once. Every probe's conditioning
context contains other probes' positions, so after the rewrite `g` no longer describes
the text it is being applied to, and pushing harder makes the mismatch worse.

Two earlier implementation bugs, both of which flatlined the embedder, were fixed
before this: dividing the guidance by `|S_j|` (putting it ~50x below the fluency term),
and letting probe sets overlap so ~6 probes fought over each position and cancelled out
(fixed by a keyed *partition* — `prng.partition_patterns`).

### BasinMark-C: removing staleness by construction (`basinmark/carrier.py`) — UNDER TEST

Reserve a keyed **carrier set** `P` (a fraction of the span, partitioned into the `M`
probe sets) and mask **all of `P`** in both arms of every probe. Each arm then conditions
on `span \ (P u D_j^b)`, which contains no carrier position — so nothing written into
`P` can change any arm's output. The guidance table is **exact, fixed, and computed
once**:

    2*M forwards for the whole embedding, versus 3*M per round with no guarantee.

Carrier tokens are then committed progressively, low-confidence-remasking style, with a
fresh fluency forward at each step that *does* see already-committed carriers — quality
is refined without ever invalidating `g`. `D_j^0`/`D_j^1` stay exchangeable, so the
exact null is unchanged.

Results pending (`exp/05_carrier.py`).

### Detector: use the magnitudes, and many ablations

Counting sign matches discards how large each `Delta_j` is, and caps the attainable
p-value at `2^-M`. Two changes:

* **Exact sign-flip test** (`carrier.signflip_pvalue`). Exchangeability makes each
  `a_j = s_j * Delta_j` symmetric about 0 *independently*, so conditional on the
  magnitudes the null is a Rademacher mixture — enumerate all `2^M` sign patterns.
  Same assumption as the sign test, strictly more power: on one real 13/16 detection,
  `p = 9.2e-4` versus `1.1e-2` for sign counting.
* **R ablation pairs per probe.** Swapping `D_j^{0,r}` and `D_j^{1,r}` negates only that
  contrast, so R independent pairs give `M*R` independent symmetric blocks instead of
  `M`, and `sqrt(R)` less noise per bit. At `M=16, R=3` the attainable floor moves from
  `2^-16 = 1.5e-5` to `2^-48 = 3.6e-15` — the range the baselines report.

### Detection cost — the method's main structural disadvantage

Every baseline detects with **zero model calls**: dgMARK checks token-id parity, and
KGW / eth-sri / KTH / Unigram / AAR hash the context. BasinMark needs model forwards,
which matters for anything that scans documents at scale. Stating it plainly rather
than burying it:

| | forwards per detection |
|---|---|
| dgMARK, KGW, eth-sri, KTH, Unigram, AAR | **0** |
| BasinMark, naive (`carrier.py`) | `2*M*R` = 96 |
| BasinMark, shared patterns (`shared.py`) | **`L` = 8, independent of payload** |

`shared.py` exploits the fact that every arm masks the *whole* carrier set, so one
forward on `mask(P u D)` already returns log-probs at every carrier position — for all
M probe sets at once. Draw `L` shared ablation patterns, and let the key choose which
ordered pair each `(probe, repetition)` block contrasts. Each block still carries its
own keyed orientation bit, so the exact sign-flip test is unchanged; blocks become
correlated through the shared patterns, but the conditional test never required
independent magnitudes. This does not close the gap to zero, and should not be
presented as if it did.

### A third embedder bug: the commit order starved the payload

Carrier tokens are committed progressively, most-confident-first. The confidence was
read off *the chosen token*, but a watermark-driven pick has a lower base log-prob **by
construction** — so every pushed position was systematically deferred to the last
commit steps, where the context is richest, the distribution sharpest, and the
admissible set collapses to a singleton. The positions meant to carry payload were
exactly the ones denied the freedom to carry it.

Observed as near-zero embedding on most samples: 0.8-1.2 % of tokens changed, `z` around
+1.4. Fixed by ordering on the position's *intrinsic* certainty (`max_v base(v)`).
Re-tuning after the fix is in progress.

### Operating point after the commit-order fix (`exp/07_tune_carrier.py`)

`M = 16` probes, `R = 3` ablation pairs, 256-token span, 4 C4 continuations,
`carrier_rate = 0.30`, `tau = 6`. `z` is the sign-flip statistic
`sum(a) / sqrt(sum(a^2))` over the `M*R = 48` blocks.

| lam | commit_steps | z (watermarked) | z (no watermark) | bit accuracy | tokens changed | cost (nats) |
|---|---|---|---|---|---|---|
| 3 | 2 | +3.20 | +0.05 | 0.750 | 0.046 | 0.36 |
| 3 | 8 | +2.96 | +0.05 | 0.703 | 0.044 | 0.36 |
| 8 | 2 | +3.95 | +0.05 | 0.797 | 0.071 | 0.86 |
| 8 | 8 | +3.55 | +0.05 | 0.781 | 0.066 | 0.87 |
| **20** | **2** | **+3.95** | +0.05 | **0.844** | 0.077 | 1.00 |
| 20 | 8 | +3.61 | +0.05 | 0.828 | 0.072 | 1.01 |

Fewer commit steps is consistently better, for the same reason the ordering bug hurt:
each committed carrier token sharpens the distribution at the positions still to be
decided, so a coarse schedule leaves them more room to carry payload. The unwatermarked
control sits at `z = +0.05` throughout.

**This is not yet a competitive detection strength.** With 48 blocks the ceiling is
`sqrt(48) = 6.93`; +3.95 is 57 % of it, roughly `p ~ 1e-5`. Raising it means more blocks
(nearly free under shared patterns, but blocks drawn from few patterns are correlated,
so the nominal ceiling is not attainable — `exp/09_blocks.py` measures where the real
gain stops) and higher per-block consistency (bit accuracy 0.844 means ~15 % of probes
carry the wrong sign and subtract from `z`).

### Shared patterns and the entropy gate (`exp/08_shared.py`)

8 C4 continuations, 256-token span, `M = 16`, `R = 3`, `L = 8` shared patterns.

| carrier selection | lam | z (wm) | worst sample | z (no wm) | z (wrong key) | bit acc | median p | tokens changed |
|---|---|---|---|---|---|---|---|---|
| keyed only | 8 | +3.10 | +1.41 | +0.38 | +0.52 | 0.820 | 1.9e-4 | 0.081 |
| keyed only | 20 | +3.18 | +1.41 | +0.38 | +0.58 | 0.836 | 7.6e-5 | 0.088 |
| entropy gate 0.6 | 8 | +3.99 | +1.21 | -0.25 | +0.31 | 0.891 | 7.8e-6 | 0.171 |
| entropy gate 0.6 | 20 | **+4.06** | +1.27 | -0.25 | -0.43 | **0.906** | **2.0e-6** | 0.182 |

Shared patterns also cut the wall clock from 26 s to **4 s per sample**, as expected from
`L = 8` forwards instead of `2*M*R = 96`.

The entropy gate is a **trade, not a free win**: ~24x better p-value, but the edit rate
doubles to 18 % of tokens. That is unsurprising — high-entropy positions are exactly the
ones with room to move, so they actually move. The internal substitution cost cannot
adjudicate this (gating to high-entropy positions raises mean cost mechanically, since
those positions have the largest admissible sets), so it is settled by perplexity under
an independent model in `exp/11_tradeoff.py`, which sweeps the same grid and reports the
detection-vs-quality curve — the form the baselines are actually compared on.

The worst sample stays around `z = +1.2-1.4` in every configuration. Per-sample variance,
not the mean, is the remaining weakness.

### Blocks vs detection power (`exp/09_blocks.py`)

Same 8 continuations, entropy gate 0.6, `lam = 20`, `commit_steps = 2`.

| L (patterns) | R (ablations) | blocks | z | % of `sqrt(blocks)` ceiling | bit acc | `p_bound` at mean z | detect forwards |
|---|---|---|---|---|---|---|---|
| 8 | 3 | 48 | +4.29 | 62 % | 0.930 | 1.0e-4 | 8 |
| **8** | **6** | 96 | **+5.15** | 53 % | 0.938 | **1.7e-6** | **8** |
| 16 | 6 | 96 | +5.33 | 54 % | 0.906 | 6.8e-7 | 16 |
| 16 | 12 | 192 | +5.55 | 40 % | 0.883 | 2.1e-7 | 16 |

The previous revision reported 4.4e-7 / 5.5e-10 / 3.8e-10 / 1.9e-11 here. Those were
per-sample medians under the Gaussian tail; the column above is the rigorous bound
evaluated at the mean z, which is the number that can actually be defended.

More blocks keeps paying, but a shrinking fraction of the nominal ceiling is reached —
blocks drawn from `L` shared patterns are correlated, exactly as expected. `L = 8, R = 6`
is the efficient point: `p = 5.5e-10` from **8 forward passes**, and doubling `R` costs
nothing because the patterns are reused.

### Attacks — WITHDRAWN AND BEING REDONE (`exp/10_attacks_shared.py`)

This table is kept only so the correction is auditable. **Three things make it
insufficient**, and all three are fixed in `exp/13_attacks_v2.py`:

* the "re-denoise" attack was `mask -> one forward -> fill all at once`. That is one-step
  masked reconstruction. Real reverse diffusion commits progressively, so tokens
  recovered early become context for the rest — the mechanism that could actually pull
  the text back into the model's natural basin. 1/4/8/16/32 steps are being run now;
* it ran at `L=8, R=3`, not the chosen operating point `L=8, R=6`;
* it reports mean `z`. "Mean signal survives" is not "detection survives". The
  `rho = 0.30` entry, `z = 2.31`, is `p_bound = 0.069` — **not significant at 5 %**. The
  replacement reports TPR at FPR 5 % / 1 % / 0.1 % against an empirical null.

Mean `z` over 8 samples, `L = 8, R = 3` (clean `z = +4.29`, unwatermarked `z ~ 0`):

| attack | rho=0.05 | rho=0.10 | rho=0.20 | rho=0.30 |
|---|---|---|---|---|
| **smooth** (argmax re-denoise) | +4.04 | +3.09 | +2.35 | +2.31 |
| **substitute** (sampled re-denoise) | +3.56 | +3.03 | +2.17 | +2.41 |
| **outside** (edit only outside the pool) | +3.30 | +2.93 | +3.35 | +2.11 |
| **delete** | -0.11 | +0.73 | -0.27 | -0.09 |

What this table does and does not support: mean signal is still present after one-step
reconstruction of 30 % of the tokens (`z = +2.31` against a null at 0), so the watermark
is not a fragile artefact of the exact tokens chosen. It does **not** support a claim of
reliable detection there — `z = 2.31` is `p_bound = 0.069`. Usable detection under this
attack only holds at the low rates (`rho = 0.05`, `z = +4.04`, `p_bound = 3.2e-4`).

The `outside` row also shows the entropy gate's desynchronisation risk is real but not
catastrophic: perturbing only non-pool positions, which can reorder the carrier
selection without touching a carrier token, costs about as much as attacking the carrier
directly.

**Deletion destroys it completely**, at every rate, exactly as predicted: patterns are
derived from absolute positions, so a single deletion shifts every role. This is a design
limitation, not a subtle robustness failure, and content-anchored patterns are the
obvious fix — not yet implemented.

Not measured: the attacker's *own* cost. An attack that rewrites 30 % of a text may
degrade it enough that the attack is self-defeating, but no perplexity or semantic
similarity was computed on attacked text here, so no such claim is made.

### Quality was never measured until late, and it changes the verdict (`exp/11`, `exp/15`)

The operating point everything above was tuned at (entropy gate, `lam = 20`, `tau = 6`)
costs **4.8x GPT-2-large perplexity**. Detection was tuned with quality unmeasured; that
is the same class of mistake as the two withdrawn claims.

`tau` is a *relative* budget: a substitution is admissible iff
`log p(v) >= log p(v*) - tau`, i.e. `p(v)/p(v*) >= e^-tau`. So

| tau | 1 | 2 | 3 | 6 |
|---|---|---|---|---|
| `p(v)/p(v*)` at least | 36.8 % | 13.5 % | 5.0 % | **0.25 %** |

`tau = 6` is extremely loose, and it was never swept.

### Fixing the sampler made the watermark harder, not easier (`exp/15_tau.py`)

With the reference LLaDA sampler in place (confidence from the clean softmax at `x0`),
draft perplexity drops **22.2 -> 9.5**: the earlier confidence bug had been degrading
generation all along, so every quality baseline before this was flattered. Re-running the
sweep on properly-generated drafts:

| tau | carrier | z | bit acc | tokens changed | ppl (draft 9.5) |
|---|---|---|---|---|---|
| 1 | keyed | +0.18 | 0.583 | 0.008 | 9.6 (x1.02) |
| 2 | keyed | +0.47 | 0.599 | 0.010 | 9.6 (x1.02) |
| 3 | entropy | +1.83 | 0.688 | 0.047 | 15.3 (x1.61) |
| 6 | keyed | +1.21 | 0.682 | 0.025 | 15.9 (x1.68) |
| 6 | entropy | **+2.84** | 0.755 | 0.070 | **38.1 (x4.01)** |

Every setting with acceptable quality has no signal; the only setting with signal costs
4x perplexity, and `z = 2.84` is still only `p_bound = 0.051`. **On the current
formulation there is no usable operating point.** Two of the earlier z ~ 5 results were
obtained partly because a buggy generator produced looser text.

Whether that mechanism — better generation sits nearer the model's mode, sharpening the
denoiser's conditional and shrinking the admissible set — is real is *inferred, not
measured*. `exp/17_capacity.py` measures it directly (denoiser entropy and `|A_i(tau)|`
for drafts from both samplers, the buggy one reproduced behind a `legacy_conf` flag).

### Carrier selection is now the bottleneck, and the selector was scoring the wrong thing

The embedder pushes along the R-pair *average* guidance
`g_{j,i}(v) = (1/R) sum_r [l_{v_r,i}(v) - l_{u_r,i}(v)]`. The first leverage selector
scored the mean of each pair's individual range — a different objective. A position where
pair 1 favours token A by +5 and pair 2 favours token B by +5 has a large range under
both pairs and almost no swing in their average, and would have been selected.

`basinmark/select.py` now partitions the pool into probe shares *first* (keyed,
text-independent), so every candidate knows which probe it would serve and is scored on
that probe's true aggregate guidance:

    U_{j,i} = ( max_{v in A_i} g_{j,i}(v) - min_{v in A_i} g_{j,i}(v) ) / 2
    C_{j,i} = mean fluency cost of reaching those extremes
    W_{j,i} = U_{j,i} / (C_{j,i} + eps)

Sign-free, because orientation swaps leave a range invariant and the randomization null
must stay clean. A second bug is fixed with it: `select="none"` was taking the first
positions of a position-sorted pool, so the keyed-only baseline was a left-of-text
carrier and the three-way comparison was meaningless. It is now a keyed random pick
inside each probe's share.

Known open issue with `W = U/(C+eps)`: the ratio prefers `U=0.05, C=0.001` over
`U=2, C=0.2`, and `eps = 0.25` only damps it. If the ratio beats entropy at all, the
formulation to use is `max U subject to C <= budget`, which is also the framing the rest
of the method already has.

### A baseline anchor, before any further tuning

We do not know whether "TPR is low at ppl x1.2" is specific to BasinMark or the price
every dLLM watermark pays. dgMARK is being reproduced at matched settings — same LLaDA
checkpoint, same reference sampler (verified line by line against their `generation.py`),
same C4 file and order, 256 tokens — and reported on the same axes, with detection cost
stated separately since dgMARK is generation-time with a zero-forward detector and
BasinMark is post-hoc with a model-forward detector. Prompt construction still differs
(they truncate to 300 characters, BasinMark took 40 tokens); the fix belongs on our side,
not in their published protocol.

### The baseline anchor, read off the dgMARK paper (arXiv:2601.22985, Table 2)

LLaDA 1.5, C4, 256 tokens, PPL under Gemma3-12B:

| method | PPL | PPL ratio | TPR @ FPR=1% |
|---|---|---|---|
| multinomial baseline | 4.21 | — | — |
| **dgMARK** | 5.27 | **x1.25** | **99.41 %** |
| dgMARK + 3-beam | 5.40 | x1.28 | 100.00 % |
| greedy baseline | 4.03 | — | — |
| **dgMARK (greedy)** | 4.44 | **x1.10** | **91.98 %** |
| KGW (delta=3) | 7.87 | x1.87 | 99.21 % |
| PATTERN-MARK (delta=3) | 7.69 | x1.83 | 95.96 % |

**This is a target anchor, not a matched baseline.** The local reproduction differs from
the paper in four ways (model, PPL evaluator, sample count, beam variant — table below),
so these numbers say what a published dLLM watermark achieves, not what dgMARK scores
under conditions identical to BasinMark's. The local run is still required and is running.
With that caveat: a dLLM watermark reaches ~92 % TPR at 1 % FPR for **x1.10** perplexity,
and ~99 % for **x1.25**. BasinMark, post-hoc, has **no usable
TPR at x1.35**, and its only configuration with any signal costs **x4.01** for
`z = 2.84` (`p_bound = 0.051`). Even KGW — the crude baseline — buys 99 % at x1.87.
On the quality-detection plane the current embedder is behind every published method by
roughly an order of magnitude — far too large a gap to be tuning, even allowing for the
evaluator mismatch.

**What generation time does and does not buy.** The per-token price is identical either
way: choosing `v` over the model's preferred `v*` costs `log p(v*) - log p(v)` whether or
not a token was previously committed there. What post-hoc additionally loses is joint
coherence — it masks 30 % of the span and refills it in two steps, i.e. from near
independent marginals. Generation-time keeps coherence but, because the guidance table
requires every non-pool token to be final first, it forces an unusual schedule: all
non-pool positions, then all pool positions. With `pool_rate = 0.5` that defers half the
span. `exp/20_gentime.py` therefore runs a `lambda = 0` two-phase control to price that
schedule *before* any watermark, and aborts the sweep if it already exceeds x1.35 — the
next formulation would then be block-local (watermark inside each 32-token block, keeping
the reference decoding order) rather than any retuning.

**Why dgMARK is cheap is also the lesson.** It never modifies token probabilities: it
changes *which masked position is unmasked next*, preferring positions whose
already-preferred token satisfies a parity condition. Quality is nearly free because
nothing is substituted. BasinMark's post-hoc embedder does the opposite — it replaces
tokens the model had already chosen — and pays for every replacement.

The negative result is worth keeping: **post-hoc token substitution cannot pay for
itself on a dLLM at this operating range.** The detector is not the problem; the
challenge-response signal is real, has an exact-by-construction null, and survives
one-step reconstruction. The embedding channel is the problem.

#### Reproduction status and its deviations from the paper

Running locally at 256 tokens, block size 32, min length 200 (all matching the paper),
with the reference sampler verified line-by-line against their `generation.py`, and the
z-statistic verified equivalent to their Eq. 1. Known deviations, none of which change
the conclusion above:

| item | paper | here | consequence |
|---|---|---|---|
| samples | 300 | 50 | TPR at FPR 0.1 % / 0.01 % not resolvable |
| PPL evaluator | Gemma3-12B | GPT-2-large | absolute PPL not comparable; only *ratios* are |
| best config | dgMARK + 3-beam | +3-beam queued after the k=1 run | — |
| model | Table 2 is LLaDA 1.5 | LLaDA-8B-Instruct | their Table 1 covers LLaDA-8B (PPL 4.90, TPR 0.957 at z=4) |

---

## Open questions — what an auditor should attack

1. **Is the detection strength worth the edit rate?** The best configuration changes
   18 % of tokens for `p ~ 2e-6`. External-model perplexity is measured in
   `exp/11_tradeoff.py`; until that lands, no claim about quality is supported.
2. **Per-sample variance.** The mean is `z = +4.06` but the worst of 8 samples is
   `+1.27`. A watermark that fails on some texts fails in deployment.
3. **Iterative re-denoising — the decisive attack, still open.** Only one-step
   reconstruction has been measured. Running now at 1/4/8/16/32 steps, at `L=8, R=6`,
   reported as TPR@FPR. Also unmeasured: what the attack costs the *attacker* in text
   quality, and paraphrase.
4. **Alignment under insertion/deletion.** Patterns are derived from absolute positions,
   so an insertion shifts every role. Content-anchored patterns are not implemented.
5. **Paraphrase** will break this, as it breaks every token-level scheme.
6. **Capacity.** Also: the entropy gate reads its ranking from a forward that masks
   the whole pool, so edits *outside* the pool can reorder the selection and
   desynchronise the detector without touching a single carrier token. Measured as the
   `outside` row in `exp/10_attacks_shared.py`. Disjoint probe sets trade payload against per-bit SNR
   (`|S_j| = span / M`). The right operating point is not established.

## Layout

```
basinmark/prng.py    keyed pattern derivation (HMAC); partition_patterns
basinmark/model.py   LLaDA-8B wrapper: masked log-probs + low-confidence-remask sampler
basinmark/core.py    BasinMark.embed / .detect / .deltas
exp/01_pilot_signal.py   go/no-go: null symmetry + controllability
exp/02_sweep.py          temperature x probe_rate x tau
exp/03_e2e.py            end-to-end embed/detect on C4 prompts
exp/03c_lambda.py        guidance-strength sweep with before/after diagnostic
basinmark/carrier.py     BasinMark-C: exact, staleness-free guidance; sign-flip test
basinmark/shared.py      shared ablation patterns: detection in L forwards
exp/04_attacks.py        smoothing / substitution / deletion  (WRITTEN, NOT RUN)
exp/05_carrier.py        BasinMark-C sweep
exp/07_tune_carrier.py   operating point after the commit-order fix
exp/08_shared.py         shared patterns, with and without the entropy gate
exp/09_blocks.py         detection power vs number of null blocks
DESIGN.md                full derivation and the honest risk list
logs_clean/              raw stdout of every run quoted above
```

Baselines (`eth-sri/diffusion-lm-watermark`, `pyomin/dgmark-watermarking`) are cloned
into `baselines/` and git-ignored. Both were patched for sm75: `bfloat16 -> float16`,
and re-enabling mem-efficient SDPA, which LLaDA's shipped code disables with a note
written for A100s — on sm75 there is no flash kernel, so torch silently falls back to
the math backend and materialises the full LxL attention matrix per layer.

## Reproduce

```bash
python exp/01_pilot_signal.py     # ~2 min after model load
python exp/02_sweep.py            # ~15 min
python exp/03c_lambda.py          # ~20 min
```
