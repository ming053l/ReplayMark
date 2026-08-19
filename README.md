# ReTrace

**Model-response watermarking for diffusion language models.** Provenance is embedded by
steering *when* model-preferred candidates are committed, and verified by re-corrupting the
finished text with keyed challenges and reading its re-denoising response.

    the decoding order places the watermark; the model's reconstruction response reads it back

Base model `GSAI-ML/LLaDA-8B-Instruct`, fp16, one TITAN RTX (sm75). Every number below was
measured on this machine; anything attributed to a paper says so.

> **Status, stated plainly.** The verification side is finished and validated: the null is
> exact, false-positive control holds at the worst key, and detection costs a fixed number
> of sequence evaluations independent of payload. On the embedding side, two families are
> retired by measurement — V1 substitution (nothing to substitute: the median position
> admits one token in budget) and V2 commit-order steering (zero-sum: order changes which
> position commits, not which tokens exist; deferral's capacity turned out to come from
> redrawing, not from context). The live formulation is **V3, response-guided resampling**
> (`basinmark/resample.py`): draw from the model's own conditional until the reconstruction
> response agrees with the keyed target, at most R guided draws plus one unconditional
> fallback. End-to-end numbers for V3 are running and not yet reported.

---

## 1. The observable

For a span of text, a secret key derives ablation patterns over the context. Two of them
form a challenge pair. Reading the model's log-probability of the *observed* token under
each arm gives a contrast

    g_i(y_i) = log p(y_i | context minus pattern A) - log p(y_i | context minus pattern B)

One keyed orientation bit per position turns this into an indicator, and the count of
those indicators is the test statistic:

    m_i = 1[ eps_i * g_i(y_i) > 0 ]        T = sum_i m_i  ~  Binomial(n, 1/2) exactly

The null is exact because `eps_i` is an independent fair coin per position, so under H0
the `m_i` are i.i.d. Bernoulli(1/2). No calibration corpus, no model-specific null
estimate, no distributional approximation.

**This is what makes the method different.** dgMARK, KGW, Unigram, AAR and KTH all read a
*hash of the token identity*. ReTrace reads what the model *does* with the text: mask part
of it under a secret key, reconstruct, and compare. That is a property of the text's
relationship to the model, not of its symbols.

Note on names: the method was called BasinMark while the embedding was still a
reconstruction-energy argument. The Python package directory and the repository URL keep
that name so the audit link stays valid; everything else, including the paper, is ReTrace.

### Validity, measured

20 keys x 120 non-watermarked generations = 2400 draws, under the decision rule detection
actually uses:

| nominal alpha | mean FPR | worst-key FPR | verdict |
|---|---|---|---|
| 0.10 | 0.0121 | 0.0583 | valid |
| 0.05 | 0.0037 | 0.0250 | valid |
| 0.01 | 0.0000 | 0.0000 | valid |

`z` over all draws: mean −0.076, sd 0.976. Deployment uses one key over many documents, so
the guarantee that matters is per-key and at the *worst* key, not averaged over keys; each
document additionally takes its own nonce, `K_d = HMAC(K, nonce_d)`.

Detection cost is `L` forward passes per block and is independent of the payload size — 32
for a 256-token document at the current settings. This is the method's main structural
disadvantage: every baseline detects with **zero** model calls.

## 2. The embedder (V2, `basinmark/blockmark.py`)

Decoding follows the reference LLaDA schedule — blocks in order, diffusion within a block.
At the start of each block the challenge table is built once, with that block *and
everything after it* masked. Nothing committed inside the block can then enter the table's
conditioning, so the table stays exact for the block's whole decoding and the detector
rebuilds it identically from the finished text.

Then, and this is the whole of V2:

    the model proposes the token it would have proposed anyway
    a position whose proposal already answers the challenge  ->  commit it first
    a position whose proposal does not                       ->  defer it

No token is ever substituted. Commit safety follows C4's CCTC: confidence eligibility,
then only the frontier-anchored prefix of the eligible set, deferrals confirmed a step
later, and a scheduled top-k branch so blocks still finish on time.

Measured so far: **quality is free** (perplexity ratio x0.81–1.00 against the reference
schedule) and 42–52 % of commits are watermark-driven. But the match rate does not move
(0.512 against a 0.518 control) and bit accuracy sits near chance. Those two facts are
inconsistent, which is the open item in §4.

## 3. Baselines, same pipeline

Same checkpoint, same C4 prompts (300-character truncation, dgMARK's published protocol),
256 generated tokens, GPT-2-large perplexity, each arm against its own decoding-regime
control.

| method | setting | PPL | ratio | TPR@1% | det. forwards |
|---|---|---|---|---|---|
| no watermark | block dec., top-k 3 | 9.94 | — | — | — |
| dgMARK | k=1 | 15.40 | x1.55 | 0.74 | 0 |
| dgMARK | +3-beam | 12.23 | x1.23 | 0.86 | 0 |
| no watermark | left-to-right | 11.61 | — | — | — |
| KGW | delta=1 | 11.96 | x1.03 | 0.93 | 0 |
| KGW | delta=3 | 14.05 | x1.21 | 1.00 | 0 |

Local dgMARK is weaker than its published table (x1.25 → 99.4 %), which is why the local
run was worth doing. KGW's x1.03 is against its own left-to-right control; charged against
block decoding it is x1.20, and under forced left-to-right LLaDA repeats about a third of
its bigrams — its detector must score each distinct bigram once or the null breaks (an
un-deduplicated detector measured a 37 % false-positive rate at delta=0).

## 4. Resolved: why order steering measured at chance

The discrepancy was resolved by three measurements, none of which was the suspected table
mismatch (`exp/03`: generation and detection agree on `g` at 256/256 positions, zero sign
flips). First, the aggregate statistic was structurally cancelling — with a balanced
message half the payload groups are driven toward `m=1` and half toward `m=0`, so a
*perfect* watermark sums to `n/2`; presence is now tested on a sync pool whose target is
fixed. Second, order is zero-sum: watermark-driven commits land `107/107, 133/133,
112/112`, but the leftovers are exactly the incompatible residue and land at `0.06–0.15`,
cancelling to `0.50`. Third, the capacity that deferral appeared to provide comes from
*redrawing*, not context: with the same context and fresh noise, an incompatible proposal
becomes compatible `0.475/0.625/0.738/0.812/0.875` within `R = 1/2/4/8/16` draws, while
committing other positions instead moves it only `0.062 → 0.113` (`exp/07`). That
measurement is what V3 is built on.

## 5. Why V1 failed — kept as a negative result

V1 embedded by *substituting* tokens under a fluency budget. It does not work on this
model, and the reason is a property of the model rather than of the schedule:

* **No room to substitute.** The median carrier position admits exactly one token inside
  the budget (`tau = 3`, i.e. `p(v)/p(v*) >= 0.05`), with no candidate-slate truncation.
  Best TPR@1% inside x1.35 perplexity: **0.10**, against dgMARK's 0.86 at x1.23.
* **Better generation makes it worse.** Fixing the sampler (confidence from the clean
  softmax at `x0`, not the max of Gumbel-perturbed logits) took draft perplexity 22.2 → 9.5
  and halved the denoiser's entropy (0.655 → 0.319), shrinking the admissible set by
  34–60 %. Part of V1's early apparent strength came from a degraded generator.
* **Global two-phase generation is worse than post-hoc.** Deferring half the span to a
  second phase costs **x1.98** perplexity before any watermark is applied — more than
  dgMARK's entire cost.

The same peakedness has been the binding constraint three times: nothing to substitute,
log-probabilities saturating so the contrast underflowed in float32, and the commit channel
converting poorly.

**Correction.** An earlier revision listed a fourth: that deferring a position returns the
same token. That was an inference from watermarked and reference outputs being identical,
which they were because the commit order had barely changed (only 11 % of commits were
watermark-driven) -- not because deferral has no effect. Measured directly
(`exp/06_deferral_effect.py`), deferral changes the model's proposal 38-51 % of the time,
rising monotonically with the wait: 0.20 after 1-2 steps, 0.35 after 3-5, 0.49 after 6-10,
0.57 after 11 or more. The channel is alive; what was missing was step budget, since a
position needs roughly six steps of waiting before a fresh draw is likely.

## 6. Layout

```
basinmark/model.py        LLaDA wrapper, reference sampler, subset log-probs
basinmark/challenges.py   keyed patterns, orientation and tie bits, the indicator
basinmark/blockmark.py    V2: block-local, order-steered embedder + detector
basinmark/kgw.py          KGW baseline in this harness
basinmark/data.py         one prompt construction for every method
legacy/                   V1: post-hoc substitution and global two-phase generation
exp/00_dgmark_eval.py     dgMARK reproduction, evaluated on our axes
exp/00b_kgw.py            KGW quality-detection curve
exp/01_blockmark.py       V2 sweep
exp/02_blockmark_where.py where the V2 signal goes
exp/03_table_agreement.py do the two sides read the same g?
exp/04_null_fixedkey.py   per-key and worst-key false-positive control
exp/legacy/               every V1 experiment, kept for the record
paper/                    ICLR-style write-up of the tables (compiles with pdflatex)
logs_clean/               raw stdout behind every number quoted here
```

## 7. Bugs worth knowing about, because each one produced a plausible-looking result

1. **A `pgrep` gate matching the shell that wrote the script.** The overnight queue
   deadlocked and the GPU idled.
2. **Arms that did not mask the blocks after the current one.** Generation and detection
   used different tables; z read 0 while quality read free — exactly the symptom.
3. **Averaging a heavy-tailed contrast instead of counting.** Sign control had no leverage
   on the mean, so the channel looked dead when the statistic was the problem.
4. **float32 `log_softmax`.** The emitted token's log-probability rounded to 0.0 under
   both arms at 28–34 % of positions, so their difference vanished and every such position
   scored a miss — dragging the *unwatermarked* match rate to 0.32–0.37 against 0.50.
   Without a no-watermark control arm in every experiment, this would have appeared as a
   working watermark.
5. **Casting to float64 after `log_softmax` had already rounded.** Recovered nothing; the
   precision has to be kept inside.
