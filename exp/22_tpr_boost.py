"""TPR levers, offline on the enriched held-out caches. Every variant keeps the exact
randomization null: each position's score is a function of (y, non-orientation domains)
and flips between two precomputed values with its orientation coin.

  V1 indicator, gated          (deployed baseline)
  V2 LLR best-pair             (P1 from base masses; tied V1 on held-out)
  V3 LLR calibrated            P1 = isotonic fit of m against q_base on exp/21's DEV
                               outputs (different prompts, different nonce domain)
  V4 multi-pair LLR            sum of calibrated LLR over the top-3 pairs per position
  V5 |g|-weighted indicator    weight = |g(y)| clipped at its dev 90th percentile
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch, hashlib, hmac
from basinmark.model import BasinModel
from basinmark.resample import ResampleMark
from basinmark.llr import build_cache
from basinmark.challenges import orientation_bits, tie_bits
from scipy.stats import binom

KEY, BLK = b"retrace-key-A", 32
HO = json.load(open("/ssd1/ming/basinmark/results/17_caches.json"))
DEV = json.load(open("/ssd1/ming/basinmark/results/21_R_matched.json"))
M = BasinModel()

# ---- calibration: (q_base_target, m) pairs from DEV watermarked docs, R matched ----
def collect_dev(arm, R, n_docs=12):
    xs, ms = [], []
    for i, il in enumerate(DEV["ids"][arm][:n_docs]):
        ids = torch.tensor(il, dtype=torch.long)[None]
        w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, retries=R, sync_frac=1.0,
                         n_payload_bits=1, nonce=f"r21-{i}")
        c = build_cache(w, ids, DEV["pls"][i], DEV["gen"])
        span = np.arange(DEV["pls"][i], DEV["pls"][i] + DEV["gen"])
        eps = orientation_bits(w.key, span); tie = tie_bits(w.key, span)
        for k, pos in enumerate(c["pos"]):
            e = eps[int(pos)]
            qb = c["qp"][k] if e > 0 else c["qm"][k]
            gy = c["gy"][k]
            m = tie[int(pos)] if gy == 0.0 else int(e * gy > 0)
            xs.append(qb); ms.append(m)
    return np.array(xs), np.array(ms)

def isotonic(x, y, nbins=20):
    """monotone P1(q): binned means with a running max (pool-adjacent light)."""
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    cen, val = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m_ = (x >= a) & (x <= b)
        if m_.sum() >= 5:
            cen.append(x[m_].mean()); val.append(y[m_].mean())
    val = np.maximum.accumulate(val)
    return np.array(cen), np.clip(np.array(val), 1e-3, 1 - 1e-3)

def mc_p(splus, sminus, T, n_mc=50_000, seed=0):
    rng = np.random.default_rng(seed); hits = 0
    for _ in range(n_mc // 10_000):
        ee = rng.integers(0, 2, size=(10_000, len(splus)))
        hits += int(((ee * splus + (1 - ee) * sminus).sum(1) >= T).sum())
    return (1 + hits) / (1 + (n_mc // 10_000) * 10_000)

R_ARM = {"control": 2, "R1": 1}
print("fitting calibration on DEV (exp/21 R2 arm)...", flush=True)
cx, cy = collect_dev("R2", 2)
cen, val = isotonic(cx, cy)
print("  P1 curve:", " ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(cen[::4], val[::4])),
      flush=True)
GCLIP = float(np.quantile(np.abs(np.array(
    [g for k in HO["caches"] for g in HO["caches"][k]["gy"]])), 0.9))

def variants(cache, key, pl, R):
    span = np.arange(pl, pl + HO["GEN"])
    eps = orientation_bits(key, span); tie = tie_bits(key, span)
    pos = np.array(cache["pos"]); alt = cache["alt"]
    qp = np.array(cache["qp"]); qm = np.array(cache["qm"]); gy = np.array(cache["gy"])
    S = np.array(cache["S"])
    e = np.array([eps[int(t)] for t in pos], float)
    tb = np.array([tie[int(t)] for t in pos], int)
    m_of = lambda sign, g: np.where(g == 0.0, tb, (sign * g > 0).astype(int))
    P1_of = lambda q: np.clip(np.interp(q, cen, val), 1e-3, 1 - 1e-3)
    out = {}
    # V1 indicator gated
    keep = S > 0.5
    mm = m_of(e, gy)[keep]
    out["V1 ind-gated"] = float(binom.sf(int(mm.sum()) - 1, len(mm), 0.5)) if len(mm) else 1.0
    # generic two-value MC scorer
    def sc(fn):
        sp, sm = fn(np.ones_like(e)), fn(-np.ones_like(e))
        return mc_p(sp, sm, float(fn(e).sum()))
    # V2 LLR (uncalibrated model)
    def f2(sign):
        qh = np.clip(np.where(sign > 0, qp, qm), 1e-3, 1 - 1e-3)
        p1 = 1 - (1 - qh) ** (R + 1)
        pm = np.where(m_of(sign, gy) == 1, p1, 1 - p1)
        return np.log(np.where(gy == 0.0, 0.5, np.clip(pm, 1e-3, None)) / 0.5)
    out["V2 LLR model"] = sc(f2)
    # V3 LLR calibrated
    def f3(sign):
        p1 = P1_of(np.where(sign > 0, qp, qm))
        pm = np.where(m_of(sign, gy) == 1, p1, 1 - p1)
        return np.log(np.where(gy == 0.0, 0.5, pm) / 0.5)
    out["V3 LLR calib"] = sc(f3)
    # V4 multi-pair calibrated LLR
    A = [[(t[1], t[2], t[3]) for t in a] for a in alt]
    def f4(sign):
        tot = np.zeros(len(pos))
        for j in range(3):
            qpj = np.array([a[j][0] if len(a) > j else 0.5 for a in A])
            qmj = np.array([a[j][1] if len(a) > j else 0.5 for a in A])
            gj = np.array([a[j][2] if len(a) > j else 0.0 for a in A])
            p1 = P1_of(np.where(sign > 0, qpj, qmj))
            pm = np.where(m_of(sign, gj) == 1, p1, 1 - p1)
            tot += np.log(np.where(gj == 0.0, 0.5, pm) / 0.5)
        return tot
    out["V4 LLR 3-pair"] = sc(f4)
    # V5 weighted indicator (gated)
    wgt = np.minimum(np.abs(gy), GCLIP) * keep
    def f5(sign):
        return wgt * (2 * m_of(sign, gy) - 1)
    out["V5 |g|-weight"] = sc(f5)
    return out

def key_of(i):
    return hmac.new(KEY, f"ho-{i}".encode(), hashlib.sha256).digest()

arms = sorted(set(k.split(":")[0] for k in HO["caches"]))
res = {a: {} for a in arms}
for arm in arms:
    idx = sorted(int(k.split(":")[1]) for k in HO["caches"] if k.startswith(arm + ":"))
    allv = {}
    for i in idx:
        v = variants(HO["caches"][f"{arm}:{i}"], key_of(i), HO["pls"][i],
                     R_ARM.get(arm, 1))
        for kk, pv in v.items():
            allv.setdefault(kk, []).append(pv)
    for kk, ps in allv.items():
        ps = np.array(ps)
        res[arm][kk] = (float(np.mean(ps < .05)), float(np.mean(ps < .01)),
                        float(np.mean(ps < .001)))
        print(f"{arm:<9}{kk:<16} TPR@5% {res[arm][kk][0]:.2f} @1% {res[arm][kk][1]:.2f} "
              f"@0.1% {res[arm][kk][2]:.2f}", flush=True)
json.dump(res, open("/ssd1/ming/basinmark/results/22_tpr_boost.json", "w"), indent=1)
print("control rows are the null check for every variant")
