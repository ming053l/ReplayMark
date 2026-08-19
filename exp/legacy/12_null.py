"""Is the null actually exact? The method claims a calibration-free p-value, so the
p-values on unwatermarked text must be Uniform(0,1) -- not merely centred. Checked with
a KS test and the empirical false-positive rate at nominal thresholds.

Cheap: 8 forwards per (text, key) pair under shared patterns.
"""
import sys, gzip, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

GEN, PREFIX, NS, NKEYS, NPR = 256, 40, 20, 10, 16
CFG = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=20.0,
           commit_steps=2, n_patterns=8, n_ablations=3, pool_rate=0.60)


def c4(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


def main():
    M = BasinModel()
    ps, zs = [], []
    t0 = time.time()
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=9000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        for k in range(NKEYS):
            d = SharedMark(M, f"null-key-{k}".encode(), **CFG).detect(x, span, 0)
            ps.append(d["p_value"]); zs.append(d["z"])
        print(f"[{i:02d}] {len(ps)} null draws, {time.time()-t0:.0f}s", flush=True)

    ps, zs = np.array(ps), np.array(zs)
    from scipy.stats import kstest
    ks = kstest(ps, "uniform")
    print(f"\n===== NULL CALIBRATION ({len(ps)} unwatermarked text x key draws) =====")
    print(f"p-values: mean {ps.mean():.3f} (want 0.500)   KS D={ks.statistic:.3f} "
          f"p={ks.pvalue:.3f}  {'UNIFORM' if ks.pvalue > 0.05 else 'NOT UNIFORM'}")
    print(f"z:        mean {zs.mean():+.3f} (want 0)   sd {zs.std():.3f}")
    for a in (0.10, 0.05, 0.01):
        print(f"  empirical FPR at nominal {a:<5} = {np.mean(ps < a):.3f}")
    json.dump(dict(p=ps.tolist(), z=zs.tolist()),
              open("/ssd1/ming/basinmark/results/null.json", "w"), indent=1)


if __name__ == "__main__":
    main()
