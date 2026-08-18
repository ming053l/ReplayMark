"""Detection power vs number of null blocks. With shared patterns, more blocks is nearly
free (L forwards regardless), but blocks drawn from few patterns become correlated, so
the nominal ceiling sqrt(M*R) is not attainable. Find where the real gain stops."""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 8, 16
GRID = [(8, 3), (8, 6), (16, 6), (16, 12)]      # (n_patterns L, n_ablations R)
LAM, POOL = 20.0, 0.60


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
    drafts = []
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        drafts.append((x, np.arange(p.shape[1], p.shape[1] + GEN)))
    print(f"[drafts ready] lam={LAM} pool={POOL}", flush=True)

    for L, R in GRID:
        cfg = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=LAM,
                   commit_steps=2, n_patterns=L, n_ablations=R, pool_rate=POOL)
        wm = SharedMark(M, KEY, **cfg)
        zp, zn, acc, pv = [], [], [], []
        t0 = time.time()
        for x, span in drafts:
            zn.append(wm.detect(x, span, MESSAGE)["z"])
            d = wm.detect(wm.embed(x, span, MESSAGE), span, MESSAGE)
            zp.append(d["z"]); acc.append(d["matches"] / NPR); pv.append(d["p_value"])
        nb = NPR * R
        print(f"L={L:<3} R={R:<3} blocks={nb:<4} ceiling z={np.sqrt(nb):.2f} | "
              f"z {np.mean(zp):+.2f} ({100*np.mean(zp)/np.sqrt(nb):.0f}% of ceiling) "
              f"no-wm {np.mean(zn):+.2f} | bit acc {np.mean(acc):.3f} | "
              f"median p {np.median(pv):.1e} | detect forwards {L} | "
              f"{(time.time()-t0)/NS:.0f}s/sample", flush=True)


if __name__ == "__main__":
    main()
