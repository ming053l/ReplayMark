"""Shared-pattern detector, with and without the entropy gate."""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

KEY, WRONG, MESSAGE = b"basinmark-key-A", b"basinmark-key-B", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 8, 16
GRID = [(None, 8.0), (None, 20.0), (0.60, 8.0), (0.60, 20.0)]


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
    print(f"[drafts ready] M={NPR} GEN={GEN} samples={NS}", flush=True)

    for pool, lam in GRID:
        cfg = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=lam,
                   commit_steps=4, n_patterns=8, n_ablations=3, pool_rate=pool)
        wm, bad = SharedMark(M, KEY, **cfg), SharedMark(M, WRONG, **cfg)
        zp, zn, zb, acc, ch, co, pv = [], [], [], [], [], [], []
        t0 = time.time()
        for x, span in drafts:
            zn.append(wm.detect(x, span, MESSAGE)["z"])
            y = wm.embed(x, span, MESSAGE)
            d = wm.detect(y, span, MESSAGE)
            zp.append(d["z"]); acc.append(d["matches"] / NPR); pv.append(d["p_value"])
            zb.append(bad.detect(y, span, MESSAGE)["z"])
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            co.append(wm.last_cost)
        tag = f"gate={pool if pool else 'off':<5} lam={lam:<5}"
        print(f"{tag} | z wm {np.mean(zp):+.2f} (min {np.min(zp):+.2f}) "
              f"no-wm {np.mean(zn):+.2f} wrong-key {np.mean(zb):+.2f} | "
              f"bit acc {np.mean(acc):.3f} | median p {np.median(pv):.1e} | "
              f"changed {np.mean(ch):.3f} cost {np.mean(co):.2f} | "
              f"{(time.time()-t0)/NS:.0f}s/sample", flush=True)


if __name__ == "__main__":
    main()
