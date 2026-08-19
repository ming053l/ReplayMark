"""The keyed challenge structure, shared verbatim by embedder and detector.

Anything here that the two sides could compute differently is a place the watermark
silently dies, so it lives in one module and both import it. Two failures already came
from exactly that: arms that did not mask the blocks after the current one (the detector
then conditioned on tokens that did not exist during generation), and a float32
log_softmax whose difference underflowed to zero at a quarter of positions.
"""
import hashlib, hmac
import numpy as np
from .prng import stream


def orientation_bits(key: bytes, positions, salt="eps"):
    """One independent keyed coin per position -- the source of the exact binomial null."""
    out = {}
    for i in positions:
        h = hmac.new(key, f"{salt}:{int(i)}".encode(), hashlib.sha256).digest()
        out[int(i)] = 1 if (h[0] & 1) else -1
    return out


def tie_bits(key: bytes, positions):
    """A second keyed coin, used only where g is exactly zero.

    Ties are not rare here and ignoring them silently breaks the null. LLaDA is confident
    enough that at many positions the emitted token has probability indistinguishable from
    one, so log_softmax rounds to 0.0 under BOTH arms and their difference is exactly zero.
    Scoring `eps * g > 0` then counts every such position as a miss, which drags the
    unwatermarked match rate to P(g != 0)/2 -- measured at 0.32-0.37 rather than 0.50,
    thirteen standard deviations off. Resolving ties by an independent keyed coin restores
    Bernoulli(1/2) by construction; the positions stay uncontrollable during embedding, in
    the same way dgMARK gains nothing at a position where no candidate matches its parity.
    """
    out = {}
    for i in positions:
        h = hmac.new(key, f"tie:{int(i)}".encode(), hashlib.sha256).digest()
        out[int(i)] = 1 if (h[1] & 1) else 0
    return out


def score(gv, eps, tie):
    """m_i, with exact ties broken by the keyed coin rather than counted as misses."""
    gv = np.asarray(gv, dtype=np.float64)
    m = (eps * gv > 0).astype(np.int64)
    z = gv == 0.0
    m[z] = tie[z]
    return m, int(z.sum())


def block_challenges(key, block_lo, block_len, n_patterns, ctx_frac=0.20, min_ctx=4,
                     mode="random"):
    """L ablation patterns over the already-final region, and the keyed pairing.

    mode="contrast" builds the pair from opposite ends of the context instead of at
    random: one pattern ablates the positions nearest the block, the other the furthest.
    Recency dominates next-token prediction, so the two arms then disagree far more, which
    widens the dynamic range of g -- the quantity `exp/21_challenge.py` measured a
    1.8-2.2x headroom on when merely picking the best of 28 random pairs. Validity is
    unaffected: the exact null needs the *orientation* bit to be a keyed coin, not the
    unordered pair to be content-independent.
    """
    region = np.arange(0, block_lo)
    n_d = max(min_ctx, int(round(ctx_frac * len(region))))
    n_d = min(n_d, max(1, len(region) // 2))
    if mode == "contrast":
        near, far = np.sort(region[-n_d:]), np.sort(region[:n_d])
        pats = [near, far]
        for l in range(2, n_patterns):
            pats.append(np.sort(stream(key, "cpat", block_lo, l)
                                .choice(region, n_d, replace=False)))
        pairs = [(0, 1)]
    else:
        pats = [np.sort(stream(key, "rpat", block_lo, l).choice(region, n_d, replace=False))
                for l in range(n_patterns)]
        r_ = stream(key, "rpair", block_lo)
        u, v = r_.choice(n_patterns, 2, replace=False)
        pairs = [(int(u), int(v))]
    return pats, pairs


def steerable_carrier(base_lp, gap_nats=1.0, min_frac=0.10):
    """Which block positions can the commit-order channel actually move?

    Order steering is exactly zero-sum unless deferring a position changes the token the
    model proposes there. Measured on this model it usually does not: watermark-driven
    commits land 107/107, 133/133, 112/112 -- perfectly -- while the positions left over
    land at 0.06-0.15, because the leftovers are precisely the ones that failed the
    compatibility test and every position must be committed eventually. The counts cancel
    to 0.50 by construction.

    So the statistic must not count positions that were never steerable. A position is
    steerable when the model is genuinely undecided there, measured as the top-2 log-prob
    gap under the block-masked context. That context contains none of the block's own
    tokens, so the detector recomputes the identical set from the finished text -- the
    same isolation argument the challenge table relies on.

    Returns a boolean mask over the block's positions.
    """
    top2 = base_lp.topk(2, dim=1).values
    gap = (top2[:, 0] - top2[:, 1]).numpy()
    mask = gap < gap_nats
    if mask.mean() < min_frac:            # never let a block contribute nothing
        keep = np.argsort(gap)[:max(1, int(min_frac * len(gap)))]
        mask = np.zeros_like(mask)
        mask[keep] = True
    return mask
