

# Overnight driver, started 2026-08-19 16:03

## L1: step budget (waiting steps enabled)

- `16:09:56` reference steps=256: ppl 8.65, null sync rate 0.510 (want 0.500), n_sync 48
- `16:17:13` steps=256  gap=1.0  pat=4 | sync 0.550 (ref 0.510) | TPR@1% 0.00 @5% 0.00 | bits 0.53 | ppl x1.18 | carrier wm/fb 44/61 (steered 0.42) | waited 0 steps
- `16:29:23` reference steps=512: ppl 6.45, null sync rate 0.524 (want 0.500), n_sync 61
- `16:42:10` steps=512  gap=1.0  pat=4 | sync 0.547 (ref 0.524) | TPR@1% 0.00 @5% 0.00 | bits 0.47 | ppl x1.11 | carrier wm/fb 33/46 (steered 0.41) | waited 256 steps
- `17:00:04` reference steps=768: ppl 8.39, null sync rate 0.517 (want 0.500), n_sync 46

### Accounting: how much more steering is needed --- RETRACTED, see note below

At `steps=512` the driver measured 33 carriers committed because the model's own token
answered the challenge and 46 force-committed, with a sync rate of 0.547 against a null
of 0.524. Since a compatibility-driven commit lands its target essentially always
(`exp/05`: 107/107, 133/133, 112/112), the forced carriers can be solved for:

    forced carriers land at 0.222   (anti-correlated, as expected: they are exactly
                                the positions that failed the test)

With n_sync = 61, a one-sided binomial test needs 41 hits, i.e. a rate of
0.672, to clear 1 %. Solving `w + (1-w) x 0.222 = 0.672`:

    required share steered by compatibility : 0.58
    currently achieved                     : 0.42

So the gap is not an order of magnitude -- it is roughly 0.42 to 0.54. The binding
resource is slack: a carrier needs about two fresh draws to find a compatible token, and
`exp/06` says a fresh draw needs about six steps of waiting, so roughly twelve steps per
carrier. At nine carriers per block that is ~108 steps per block, against the 64 that
`steps=512` provides. `steps=768` gives 96, and lowering the carrier count per block
raises the slack each one gets -- which is exactly what the gap sweep in L3 varies.

- `17:18:32` steps=768  gap=1.0  pat=4 | sync 0.589 (ref 0.517) | TPR@1% 0.00 @5% 0.00 | bits 0.46 | ppl x0.79 | carrier wm/fb 38/58 (steered 0.40) | waited 512 steps
- `17:18:32` after L1: best TPR@1% at ppl<=1.3 is 0.00 (steps=256)

## L3: carrier fraction (top-2 gap threshold)

- `17:25:37` steps=256  gap=0.5  pat=4 | sync 0.599 (ref 0.510) | TPR@1% 0.00 @5% 0.10 | bits 0.50 | ppl x1.09 | carrier wm/fb 33/42 (steered 0.44) | waited 0 steps
- `17:32:41` steps=256  gap=2.0  pat=4 | sync 0.557 (ref 0.510) | TPR@1% 0.00 @5% 0.00 | bits 0.56 | ppl x1.27 | carrier wm/fb 51/72 (steered 0.42) | waited 0 steps
- `17:39:46` steps=256  gap=4.0  pat=4 | sync 0.523 (ref 0.510) | TPR@1% 0.00 @5% 0.00 | bits 0.56 | ppl x1.24 | carrier wm/fb 55/80 (steered 0.41) | waited 0 steps

## L4: challenge sharpness (more shared patterns)


> **Retraction.** The block above solved for the forced carriers' hit rate using
> `rate` and `n_sync`, which are **sync-group only**, together with `carrier_wm` and
> `carrier_fb`, which the generator accumulates over **all** carriers including the payload
> groups. Mixing those denominators makes the implied 0.222 and the implied 0.42 -> 0.58 gap
> unsupported. The instrumentation needs `sync_wm_hits / sync_wm_total` and
> `sync_fb_hits / sync_fb_total` recorded separately before any such accounting is redone.
>
> A second bug found at the same time: `ref_cache` is keyed on `steps` alone, but the
> carrier set is determined by `gap_nats`, so the L3 sweep would have compared every gap
> against the null measured at `gap = 1.0`. Each watermarked document's own p-value is
> still computed against its own carrier set, so the TPR column survives; `rate_ref`,
> `n_sync` and any mechanism accounting derived from them do not.
