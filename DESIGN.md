# BasinMark v2 — watermarking dLLMs via keyed re-denoising contrast

## 0. What survives from v1 and what does not

**Survives (the good idea):** put provenance in *how the final text responds to the
diffusion operator*, not in token frequency. The detector re-corrupts the text with a
secret key and probes the model. dLLM-native, and no other dLLM watermark does this.

**Does not survive (as specified):** the statistic

    Δ_j = E(y, C^0) − E(y, C^1),   E(y,C) = −Σ_{i∈C} log p(y_i | C(y))

Here `C^0` masks position set `A` and `C^1` masks a *different* set `B`. So Δ_j is a
difference of surprisals over **different tokens**. For

    Mask A: The [MASK] generates text through iterative [MASK].
    Mask B: The model [MASK] text through [MASK] denoising.

Δ is dominated by whether the key happened to select function words or content words —
several nats of variance, with a nonzero, text-dependent mean. The claim "natural text
has Δ ≈ 0" is false, and there is no null distribution to test against without an
expensive per-text calibration. **This is a validity bug, not a tuning issue.**

## 1. The fix: same probe tokens, two keyed context ablations

For probe `j`, the key derives three **disjoint** position sets:

- `S_j`  — the *probe set*: positions whose reconstruction we measure.
- `D_j^0`, `D_j^1` — two *context ablations*, same size, disjoint from `S_j` and from each other.

Two corruptions:

    C^0(y) = mask(S_j ∪ D_j^0)        C^1(y) = mask(S_j ∪ D_j^1)

Energy is read **only on `S_j`**, in both arms:

    ℓ^b_i = log p_θ(y_i | C^b(y)),      i ∈ S_j,  b ∈ {0,1}
    δ_i   = ℓ^1_i − ℓ^0_i
    Δ_j   = mean_{i∈S_j} δ_i

Now both arms score the *same tokens*; only the surrounding context differs. The
watermark asks a different, sharper question:

> **Which half of its own context does this text rely on to reconstruct itself?**

### Why this has an exact null

Under H0 (text not produced with key `K`), the key is independent of the text, and
`D_j^0`, `D_j^1` are drawn exchangeably. Swapping them negates Δ_j. Hence Δ_j is
**symmetric about 0**, so

    sign(Δ_j) ~ Bernoulli(1/2),   independent across j

and the number of sign matches against the expected codeword is exactly
`Binomial(M, 1/2)`. **Exact p-values, no calibration corpus, no model-specific null.**
v1 had neither.

## 2. The fix that makes embedding tractable

v1 proposed per-candidate one-step lookahead `R(v) ≈ Δ_j(ŷ^{(v)})`, i.e. a forward
pass per vocabulary item. Infeasible.

But in the v2 design every `i ∈ S_j` is masked in **both** arms, so `y_i` never enters
either conditioning context. Therefore the contribution of position `i` to `Δ_j` is a
pure table lookup:

    g_i(v) = ℓ^1_i(v) − ℓ^0_i(v)                    (v ranges over the whole vocab)

**Two forward passes yield the complete guidance table for every probe position and
every vocabulary item simultaneously.** The lookahead is exactly free. This is the
single change that turns the proposal into something runnable.

## 3. Embedding = generate → keyed re-mask → biased re-denoise

Guidance during the original denoising loop is awkward (the table depends on context
that does not exist yet). A dLLM lets us do the natural thing instead — and an AR model
cannot:

    y ← dLLM.generate(prompt)                         # unwatermarked draft
    repeat R rounds:
      for each probe j:
        ℓ^0, ℓ^1  ← 2 forwards on C^0(y), C^1(y)      # guidance table  g_i(v)
        base      ← 1 forward on mask(S_j) only       # clean fluency model
        for i ∈ S_j:
          admissible = { v : log base_i(v) ≥ max_v log base_i(v) − τ }
          y_i ← argmax_{v ∈ admissible} s_j · g_i(v)

`τ` (nats) is a **hard per-token quality budget**: a token is only replaced by one the
unwatermarked model considered nearly as good. Low-entropy positions ("capital of
France is Paris") have a singleton admissible set and are automatically left alone —
the entropy gating of v1 falls out of the constraint instead of needing a schedule.

Cost: 3 forwards per probe per round; detection 2 per probe.

## 4. Payload

`M` probes → `M` bits → BCH/repetition ECC. Zero-bit detection = sign-match count with
an exact binomial p-value.

## 5. Honest risks (to be measured, not asserted)

1. **Controllability** — is `max_{v∈admissible} g_i(v) − g_i(y_i^orig)` large relative
   to the per-position noise `std(δ)`? If masking a few extra context tokens barely
   moves the predictive distribution, there is no signal and the idea dies. *This is
   the go/no-go experiment and it runs first.*
2. **Denoising-smoothing attack** — an attacker with the same model re-masks x% at
   random and re-denoises, pulling the text into the model's natural basin. This is the
   natural adversary for a functional watermark and is cheap for the attacker. Must be
   evaluated; no other dLLM watermark paper faces it in this form.
3. **Alignment under insert/delete** — v1 patterns are absolute-position-derived, so
   insertions shift every role. v1 mitigates with an offset scan at detection;
   content-anchored patterns are future work.
4. **Paraphrase** — will break token-level BasinMark, as it breaks all token-level
   schemes. The semantic variant is deferred until (1) is settled.
