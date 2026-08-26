"""Keyed probes, response directions, carrier roles, and scoring rules."""
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
                     mode="random", region=None):
    """L ablation patterns over the already-final region, and the keyed pairing.

    mode="contrast" builds the pair from opposite ends of the context instead of at
    random: one pattern ablates the positions nearest the block, the other the furthest.
    Recency dominates next-token prediction, so the two arms disagree more and widen the
    dynamic range of g. The exact test requires the orientation bit to be a keyed coin; the
    unordered pair itself need not be content-independent.
    """
    if region is None:
        region = np.arange(0, block_lo)
    region = np.asarray(region)
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


def steerable_carrier(base_lp, gap_nats=1.0, min_frac=0.0):
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
    same isolation argument as the Reproducible Response Bank.

    Returns a boolean mask over the block's positions.
    """
    top2 = base_lp.topk(2, dim=1).values
    gap = (top2[:, 0] - top2[:, 1]).numpy()
    mask = gap < gap_nats
    # min_frac > 0 would force the lowest-gap positions in when a block has none above
    # threshold, which keeps coverage up but makes "only genuinely steerable positions are
    # counted" untrue. It defaults off: a block with nothing steerable contributes nothing,
    # so the mechanism question is answered before the coverage question.
    if min_frac > 0 and mask.mean() < min_frac:
        keep = np.argsort(gap)[:max(1, int(min_frac * len(gap)))]
        mask = np.zeros_like(mask)
        mask[keep] = True
    return mask


def roles(key: bytes, span, n_payload_bits, sync_frac=0.5):
    """Split carriers into a presence pool and payload groups.

    Presence cannot be a 1-of-n_groups slice. With ~50 steerable carriers in a document,
    an eighth of them is ~6, and 6/6 perfect hits give p = 2^-6 = 0.0156 -- so TPR at a 1 %
    threshold would be unreachable no matter how well the channel works. Half the carriers
    go to presence instead, where ~19/25 already clears the 1 % tail.

    Returns position -> -1 for the presence pool, else the payload group index.
    """
    r = stream(key, "role", len(span), n_payload_bits, sync_frac)
    u = r.random(len(span))
    grp = r.integers(0, max(1, n_payload_bits), size=len(span))
    return {int(i): (-1 if u[k] < sync_frac else int(grp[k]))
            for k, i in enumerate(span)}
