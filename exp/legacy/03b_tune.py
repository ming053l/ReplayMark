"""Small grid over payload size and guidance weight, on shared drafts."""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.core import BasinMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS = 192, 40, 3
GRID = list(itertools.product([16, 32], [1.0, 3.0], [4.0]))


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
                       seed=1000 + i).cpu()
        drafts.append((x, np.arange(p.shape[1], p.shape[1] + GEN)))
    print(f"[drafts ready] |S_j| will be {GEN}/n_probes", flush=True)

    for npr, lam, tau in GRID:
        wm = BasinMark(M, KEY, n_probes=npr, ctx_rate=0.20, tau=tau, lam=lam,
                       margin=0.15, disjoint=True)
        t0, pos, neg, ch, co = time.time(), [], [], [], []
        for x, span in drafts:
            neg.append(wm.detect(x, span, MESSAGE)["matches"] / npr)
            y = wm.embed(x, span, MESSAGE, rounds=3)
            pos.append(wm.detect(y, span, MESSAGE)["matches"] / npr)
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            co.append(wm.last_cost)
        print(f"n_probes={npr:<3} |S|={GEN//npr:<3} lam={lam:<4} tau={tau} | "
              f"bit acc {np.mean(pos):.3f} (no-wm {np.mean(neg):.3f}) "
              f"changed {np.mean(ch):.2f} cost {np.mean(co):.2f} nats "
              f"| {(time.time()-t0)/NS:.0f}s/sample", flush=True)


if __name__ == "__main__":
    main()
