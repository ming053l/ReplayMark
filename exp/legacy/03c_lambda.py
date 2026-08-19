"""Is the guidance strong enough? Sweep lambda/tau and check s*Delta actually moves."""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.core import BasinMark
from basinmark.prng import payload_bits

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 192, 40, 3, 16
GRID = list(itertools.product([3.0, 10.0, 30.0, 1e6], [4.0, 6.0]))


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
    signs = payload_bits(KEY, NPR, MESSAGE)
    print(f"[drafts ready] n_probes={NPR} |S_j|={GEN//NPR}", flush=True)

    for lam, tau in GRID:
        wm = BasinMark(M, KEY, n_probes=NPR, ctx_rate=0.20, tau=tau, lam=lam,
                       margin=0.15, disjoint=True)
        pre, post, acc, ch, co = [], [], [], [], []
        for x, span in drafts:
            D0, _ = wm.deltas(x, span)
            y = wm.embed(x, span, MESSAGE, rounds=3)
            D1, _ = wm.deltas(y, span)
            pre.append((signs * D0).mean()); post.append((signs * D1).mean())
            acc.append((np.sign(D1) == signs).mean())
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            co.append(wm.last_cost)
        print(f"lam={lam:<8g} tau={tau} | s*Delta {np.mean(pre):+.3f} -> {np.mean(post):+.3f} "
              f"| bit acc {np.mean(acc):.3f} | changed {np.mean(ch):.2f} "
              f"cost {np.mean(co):.2f} nats", flush=True)


if __name__ == "__main__":
    main()
