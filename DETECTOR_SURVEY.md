# Detector-design survey (2026-08-22)

Raw material for the Shibboleth paper's detector positioning, robustness-table design, and
the exp/33 (block-local detection) analysis. Compiled from full-text reads where marked
[FT], abstract-level otherwise [AB].

## AR watermark detectors

- **KGW z-test** (arXiv:2301.10226) [FT]: one-proportion z on green count; Gaussian tail,
  z>4 → nominal 3e-5 FPR; key-only, O(T); dedup repeated n-grams to fix FPR.
- **WinMax / reliability study** (arXiv:2306.04634) [FT]: max z over all contiguous spans,
  O(T²); acknowledges multiplicity and calibrates FPR **empirically** — no analytic
  correction. LeftHash (h=1, robust, learnable) vs SelfHash (h=4, secure, fragile).
  De-facto standard attack menu: copy-paste CP-k-p% (k∈{1,3}, p∈{10,25}%), DIPPER, GPT
  paraphrase, human paraphrase; TPR@1e-3 in tables, headlines at 1e-5.
- **Aaronson/Kirchner Gumbel (EXP)** [FT via Three Bricks]: S_T = Σ −ln(1−r); exact
  Γ(T,1) null → upper incomplete gamma p-value. With model access the NP-optimal score
  Σ(1/p−1)ln r strictly dominates (Fernandez §V-A) — precedent that model access buys power.
- **Kuditipudi ITS/EXP-edit** (arXiv:2307.15593) [FT]: min soft-Levenshtein alignment cost
  over blocks × key offsets; **permutation test** (T=5000 → p-floor ~2e-4); O(m·n·k²) —
  the accepted precedent for an *expensive* detector. Robust to 40–50% corruption;
  substring detection. Reports **median p-values**, not TPR@FPR.
- **Unigram** (arXiv:2306.17439) [FT]: fixed global green list (h=0); FPR bound over
  key randomness, degrades gracefully with repetition; provable edit-distance robustness
  η ≥ √n(z−τ)/(1+γ/2). TPR/F1 at fixed FPR 1% and 10%.
- **SynthID-Text** (Nature 2024) [AB+repo]: tournament sampling; detectors = mean score →
  weighted mean → **Bayesian learned detector** (trained per key; trades analytic null for
  power). Empirical threshold calibration per token length. Headline TPR@FPR=1%.
- **Three Bricks** (arXiv:2308.00113) [FT]: exact binomial/Gamma tails; empirical-vs-nominal
  FPR validated on 10 keys × 100k sequences down to <1e-6 — but only with exact tests +
  n-gram dedup. Multi-bit: global p = 1−(1−p)^M (union/Šidák over M keys).
- **Christ–Gunn–Zamir** (arXiv:2306.09194) [FT]: entropy-gated PRF embedding; cryptographic
  soundness (negligible FPR for any key-independent text); **substring-completeness** =
  block-local detection with union bound absorbed into the security parameter. Detector
  explicitly CANNOT have the prompt or model.
- **Hu unbiased/LLR** (arXiv:2310.10669) [AB]: per-token LLR needs BOTH conditionals (model
  + effectively prompt); martingale Chernoff bound, explicit log-A multiplicity for grid
  search; maximin variant for graceful degradation.
- **GaussMark** (arXiv:2501.13941) [FT]: key = Gaussian weight perturbation; statistic =
  normalized ⟨ξ, ∇θ log p⟩; **exact N(0,1) null vs arbitrary text** by rotational
  invariance; detection = 1 fwd + 1 bwd pass, <1 s/1k tok; provider-side framing without
  apology. Closest structural analogue to Shibboleth's model-access detector.

## Segment/block pooling

- **WaterSeeker** (arXiv:2409.05112) [FT]: sliding-window locate-then-detect; per-window
  α=1e-6 target → **realized document FPR ≈ 5e-3** on 10k×10k-token simulation (~5000×
  inflation from overlapping windows). The cautionary number for max-over-windows.
- **GCD/AOL** (arXiv:2410.03600) [AB]: multi-scale interval cover, FWER ≤ n·τ via explicit
  union bound over O(n) intervals — the Bonferroni construction for block-pooled detection.
- **Tr-GoF** (arXiv:2411.13868, JRSSB) [AB]: mixture model of edited text; truncated
  goodness-of-fit / higher-criticism statistic; detection boundary q+2p=1 vs q+p=1/2 for ANY
  sum-based rule under sparse edits — the theory behind "Bonferroni-min/GoF beats pooled
  when few blocks survive." Companion: arXiv:2404.01245 (Ann. Statist. 2025).
- **MarkMyWords** (arXiv:2312.00273) [AB]: benchmark convention — fix FPR=2%, report
  "watermark size" = tokens needed to detect.

## dLLM watermarks (competitive set — all key-only, zero-model detectors)

- **Gloaguen et al.** (arXiv:2509.24368, ICLR 2026) [FT]: bidirectional context red/green,
  SumHash/MinHash, expectation boost + context promotion; detector = AR binomial + dedup,
  unchanged. TPR@1% >99% on LLaDA-8B (~275 tok) and Dream-7B at δ=4. Attacks: deletion,
  substitution, BERT substitution, GPT-5-mini paraphrase, back-translation. No re-denoising
  attack, no block-local stats.
- **DMark** (arXiv:2510.02902) [FT]: predictive / bidirectional / both; plain KGW z with
  **empirical** thresholds; TPR at FPR∈{0.5,1,5}%. Robustness table: del/ins/sub 10–20%,
  DIPPER (lex20/order0/interval3), GPT-5-nano.
- **dgMARK** (arXiv:2601.22985) [AB]: parity of unmasking order; windowed z pooled as
  (1/S)Σz_s² (two-sided, indel-flips-parity rationale); empirical thresholds; TPR@FPR
  {10,1,0.1,0.01}%; attacks ε∈{.1...4} + DIPPER-11B + Llama-3-8B paraphrase.
- **LR-DWM** (arXiv:2601.12376) [FT]: independent left/right hashes, ternary score,
  σ estimated empirically on 10k human texts (dependent adjacent scores); TPR@1%; reports
  **PPL at matched detection rates {90,99,99.5}%** — clean inverted presentation.
- **Bagchi et al.** (arXiv:2511.02083) [AB]: Gumbel-max per step, (i+s) mod m seeding;
  analytic FPR ≤ m·exp(−L(ζ−ln(1+ζ))) — union bound over m offsets. Weak results (TPR 77%).
- **PATTERN-MARK** (arXiv:2410.13805) [FT]: order-agnostic; exact null via **DP over a
  Markov chain** — third route to exact p-values after binomial/Gamma and permutation.
  TPR at theoretical FPR {10,1,0.1,0.01,0.001}%.
- **Attacks**: Chainwash (arXiv:2605.05503) multi-hop rewriting drops Gloaguen 87.9%→4.9%
  after 5 hops; Re-Mask-and-Redirect (arXiv:2604.08557) = citable precedent for the
  re-denoising threat model, never yet evaluated on watermarks.

## Design lessons applied / to apply

(a) Shibboleth is the first dLLM watermark whose detector needs the model — cite GaussMark,
Hu LLR, Fernandez NP-score, Kuditipudi cost precedent (DONE in related work + experiments,
2026-08-22). Quantify wall-clock per 512-tok text when possible.
(b) Exact null: already binomial. TODO: Fernandez-style empirical-FPR validation at scale
(≥100k null texts, log-log plot to 1e-5) if a bigger venue is targeted; current 20×120
validity table is workshop-scale.
(c) Block-local: **disjoint blocks + Bonferroni exactly** (α/16) is the provable default;
Stouffer for dense signal; crossover on a copy-paste sweep is the money experiment. Never
report per-block α as document FPR (WaterSeeker's 5000× warning). exp/33 tests exactly
pooled vs Bonferroni-min vs Stouffer.
(d) Desync story needed for indels: shift-window union bound, or DP alignment, or
block-local so a shift kills one block (= exp/33/34 direction).
(e) Say where carrier selection sits on the LeftHash↔Unigram robustness/learnability axis.
(f) Expected comparisons: KGW, Gumbel-exact, EXP-edit, Gloaguen, DMark, dgMARK, LR-DWM
(+SynthID, PatternMark, GaussMark advisable). Prepared answer for "why not Gloaguen's free
detector": re-denoising / sparse-survival / spoofing-resistance regimes, shown in a table.
(g) Operating points: TPR@{1e-2, 1e-3} minimum (current tables comply); 1e-4 and an
empirical-validity figure to 1e-5/1e-6 for a main-conference version.
(h) Robustness menu (table redesigned 2026-08-22): sub (random + masked-LM) / del / ins at
10–30%, copy-paste CP-1-25%, DIPPER lex20-order0, instruct-LLM paraphrase, roundtrip
translation, **re-denoising ρ∈{5,10,25,50}%** — running re-denoising against dgMARK/KGW too
would be the strongest single result. Multi-hop Chainwash protocol optional.
(i) State explicitly: does replay need the prompt? (CGZ objection); replay determinism
through the real inference stack (validity table already does this — say so).

Full reference list with arXiv IDs in the paper's main.bib additions and the survey agent
transcript.
