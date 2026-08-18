# BasinMark — watermarking diffusion LLMs via keyed re-denoising contrast

**Status: research in progress. The detector is validated; the embedder is not yet
strong enough. Numbers below are what has actually been measured on this machine —
nothing here is a claim from a paper.**

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

### Exact null

`D_j^0` and `D_j^1` are drawn exchangeably, so swapping them negates `Delta_j`. Under
H0 (text not produced with key K) `Delta_j` is therefore symmetric about 0, giving

    sign(Delta_j) ~ Bernoulli(1/2)   ->   sign-matches ~ Binomial(M, 1/2)

Exact p-values, no calibration corpus, no model-specific null estimate.

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
| Null is exactly Binomial(M, 1/2) | **verified** — `P(sign>0)=0.521` over 96 probes |
| Guidance table costs 2 forwards, not \|V\| | **verified** — 0.3 s/probe on one TITAN RTX |
| Probe positions are controllable | **partly** — depends strongly on decoding temperature and quality budget `tau` (see below) |
| Bits can be set end-to-end | **partly** — 0.844 bit accuracy at 16 bits, z = +3.95 vs +0.05 unwatermarked, 7.7 % of tokens changed |
| Robust to attack | **not started** — `exp/04_attacks.py` written, not run |
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

---

## Open questions — what an auditor should attack

1. **Is the residual signal big enough at an acceptable quality cost?** `tau = 6` nats
   is a large per-token budget. The realised cost is far lower (0.43 nats) but the
   quality impact has not been measured against an external perplexity model yet.
2. **Denoising-smoothing attack.** The adversary owns the same dLLM, masks x % of the
   text at random and re-denoises, pulling it into the model's natural basin. This is
   the natural adversary for a *functional* watermark and none of the four prior dLLM
   watermarks face it in this form. If BasinMark dies here, the idea dies.
   `exp/04_attacks.py` implements it; it has not been run.
3. **Alignment under insertion/deletion.** Patterns are derived from absolute positions,
   so an insertion shifts every role. Content-anchored patterns are not implemented.
4. **Paraphrase** will break this, as it breaks every token-level scheme.
5. **Capacity.** Disjoint probe sets trade payload against per-bit SNR
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
