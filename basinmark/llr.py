"""LLR detection over ALL positions, with an exact randomization null.

The indicator detector gated positions by two-sided mass and counted hits, which throws
away the one-sided majority. Under the watermark those positions are not noise: when the
key's side holds mass ~1 the first draw is accepted and IS the natural sample (zero
quality cost, m=1 almost surely); when it holds mass ~0 every guided draw is rejected and
the fresh fallback emits the natural sample (zero quality cost, m=0 almost surely). Both
extremes carry near-maximal evidence -- one positive, one negative -- and the measured
bimodality of q (quartiles 0.02/0.60/0.98) says most positions are extremes.

Per position the detector knows, from the block-masked table (verifier-reproducible):
q+ and q- (masses of each contrast side under the base conditional at temperature),
g(y_i), and the tie coin. Given eps_i, the watermark's acceptance mass is q_hat = q+ if
eps>0 else q-, and

    P1[m=1] = 1 - (1 - q_hat)^(R+1)        P0[m=1] = 1/2   (eps is a fair coin under H0)

Score  T = sum_i log( P1[m_i] / 0.5 ).  Since the pair choice, masses and g(y) are
functions of the text and non-orientation key domains only, and each eps_i is an
independent coin under H0, the exact null of T is the distribution over eps redraws --
estimated by Monte-Carlo with the add-one correction (each position contributes one of
two precomputed values, flipping with eps).

Calibration caveat, stated rather than hidden: q_hat is the BASE-conditional mass while
the embedder accepted under the LIVE conditional (corr ~0.2 overall, likely higher at the
extremes). Miscalibrated q_hat costs power, never validity -- the null never uses P1.
"""
import numpy as np, torch
from .challenges import orientation_bits, tie_bits
from .resample import ResampleMark

CLIP = 1e-3


@torch.no_grad()
def build_cache(w: ResampleMark, ids, prompt_len, gen_len):
    """One detection pass -> per-position arrays; everything later is offline."""
    gen_end = prompt_len + gen_len
    P, QP, QM, GY = [], [], [], []
    for b in range(max(1, gen_len // w.block_len)):
        lo = prompt_len + b * w.block_len
        B, g, S, qp, qm = w._table(ids, lo, gen_end)
        y = ids[0, torch.tensor(B)]
        GY.extend(g.gather(1, y[:, None]).squeeze(1).tolist())
        P.extend(B.tolist()); QP.extend(qp.tolist()); QM.extend(qm.tolist())
    return dict(pos=P, qp=QP, qm=QM, gy=GY)


def llr_pvalue(cache, key, prompt_len, gen_len, R, n_mc=100_000, seed=0):
    span = np.arange(prompt_len, prompt_len + gen_len)
    eps = orientation_bits(key, span)
    tie = tie_bits(key, span)
    pos = np.array(cache["pos"]); qp = np.array(cache["qp"]); qm = np.array(cache["qm"])
    gy = np.array(cache["gy"])
    e = np.array([eps[int(i)] for i in pos], dtype=np.float64)
    tb = np.array([tie[int(i)] for i in pos], dtype=np.int64)

    def scores(sign):
        """log(P1[m]/0.5) for every position, given orientation vector `sign`."""
        qh = np.clip(np.where(sign > 0, qp, qm), CLIP, 1 - CLIP)
        p1 = 1 - (1 - qh) ** (R + 1)                      # P1[m=1]
        m = np.where(gy == 0.0, tb, (sign * gy > 0).astype(np.int64))
        pm = np.where(m == 1, p1, 1 - p1)
        pm = np.where(gy == 0.0, 0.5, pm)                 # ties carry no evidence
        return np.log(np.clip(pm, CLIP, None) / 0.5)

    T = float(scores(e).sum())
    rng = np.random.default_rng(seed)
    hits = 0
    s_plus, s_minus = scores(np.ones_like(e)), scores(-np.ones_like(e))
    for _ in range(n_mc // 10_000):
        ee = rng.integers(0, 2, size=(10_000, len(e)))
        Ts = (ee * s_plus[None, :] + (1 - ee) * s_minus[None, :]).sum(1)
        hits += int((Ts >= T).sum())
    n_done = (n_mc // 10_000) * 10_000
    return dict(T=T, p_value=(1 + hits) / (1 + n_done),
                z=float((T - (s_plus + s_minus).sum() / 2)
                        / (np.sqrt(((s_plus - s_minus) ** 2).sum() / 4) + 1e-12)))
