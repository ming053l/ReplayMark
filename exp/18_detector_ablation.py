"""Detector ablations, offline on exp/17's caches. Pure CPU; no generation is touched, so
every variant is evaluated on the identical texts and differences are attributable to the
statistic alone. Control arm is the null check for every variant.

Variants: LLR over all positions (deployed candidate); LLR gated to two-sided positions
(does the gate that helped the indicator hurt the LLR?); indicator without gate (is the
gate or the count the indicator's binding problem?); detector-R mismatch (robustness of
P1's R to the embedder's true R -- a deployment concern, since R is the embedder's knob).
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np
from basinmark.challenges import orientation_bits, tie_bits
from basinmark.llr import llr_pvalue
import hashlib, hmac
from scipy.stats import binom

KEY = b"retrace-key-A"
d = json.load(open("/ssd1/ming/basinmark/results/17_caches.json"))
pls, GEN = d["pls"], d["GEN"]

def doc_key(i):
    return hmac.new(KEY, f"ho-{i}".encode(), hashlib.sha256).digest()

def indicator(cache, key, pl, gate):
    span = np.arange(pl, pl + GEN)
    eps = orientation_bits(key, span); tie = tie_bits(key, span)
    pos = np.array(cache["pos"]); qp = np.array(cache["qp"]); qm = np.array(cache["qm"])
    gy = np.array(cache["gy"])
    e = np.array([eps[int(t)] for t in pos]); tb = np.array([tie[int(t)] for t in pos])
    S = 2 * np.minimum(qp, qm)
    keep = (S > 0.5) if gate else np.ones(len(pos), bool)
    m = np.where(gy == 0.0, tb, (e * gy > 0).astype(int))[keep]
    n, h = len(m), int(m.sum())
    return float(binom.sf(h - 1, n, 0.5)) if n else 1.0

def llr_gated(cache, key, pl, R):
    qp = np.array(cache["qp"]); qm = np.array(cache["qm"])
    keep = (2 * np.minimum(qp, qm)) > 0.5
    sub = {k: list(np.array(cache[k])[keep]) for k in ("pos", "qp", "qm", "gy")}
    return llr_pvalue(sub, key, pl, GEN, R=R, n_mc=50_000)["p_value"]

arms = sorted(set(k.split(":")[0] for k in d["caches"]))
Rtrue = {"control": 1, "R1": 1, "R2": 2}
print(f"{'arm':<9}{'variant':<22}{'TPR@5%':>8}{'@1%':>7}{'@0.1%':>8}")
for arm in arms:
    idx = sorted(int(k.split(":")[1]) for k in d["caches"] if k.startswith(arm + ":"))
    for label, fn in (
        ("LLR all, R match", lambda c, k, pl: llr_pvalue(c, k, pl, GEN,
                                                         R=Rtrue[arm], n_mc=50_000)["p_value"]),
        ("LLR all, R=1 fixed", lambda c, k, pl: llr_pvalue(c, k, pl, GEN, R=1,
                                                           n_mc=50_000)["p_value"]),
        ("LLR gated S>0.5", lambda c, k, pl: llr_gated(c, k, pl, Rtrue[arm])),
        ("indicator gated", lambda c, k, pl: indicator(c, k, pl, True)),
        ("indicator all", lambda c, k, pl: indicator(c, k, pl, False)),
    ):
        ps = np.array([fn(d["caches"][f"{arm}:{i}"], doc_key(i), pls[i]) for i in idx])
        print(f"{arm:<9}{label:<22}{np.mean(ps<.05):>8.2f}{np.mean(ps<.01):>7.2f}"
              f"{np.mean(ps<.001):>8.2f}", flush=True)
