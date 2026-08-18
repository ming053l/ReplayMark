"""Pick the operating point after the commit-order fix: guidance weight x commit schedule."""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.carrier import CarrierMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 4, 16
GRID = list(itertools.product([3.0, 8.0, 20.0], [2, 8]))


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
    print(f"[drafts ready] M={NPR} GEN={GEN}", flush=True)

    for lam, cs in GRID:
        wm = CarrierMark(M, KEY, n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20,
                         tau=6.0, lam=lam, commit_steps=cs, n_ablations=3)
        zp, zn, acc, ch, co = [], [], [], [], []
        t0 = time.time()
        for x, span in drafts:
            zn.append(wm.detect(x, span, MESSAGE)["z"])
            y = wm.embed(x, span, MESSAGE)
            d = wm.detect(y, span, MESSAGE)
            zp.append(d["z"]); acc.append(d["matches"] / NPR)
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            co.append(wm.last_cost)
        print(f"lam={lam:<5} commit_steps={cs} | z {np.mean(zp):+.2f} (no-wm {np.mean(zn):+.2f}) "
              f"| bit acc {np.mean(acc):.3f} | changed {np.mean(ch):.3f} cost {np.mean(co):.2f} "
              f"| {(time.time()-t0)/NS:.0f}s/sample", flush=True)


if __name__ == "__main__":
    main()
