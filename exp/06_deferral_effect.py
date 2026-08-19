"""Does deferring a position change what the model proposes there?

This is the load-bearing assumption of the whole commit-order channel, and it has never
been measured directly. Order steering cannot bias a count unless deferral changes the
proposal: every position is committed eventually, so if a deferred position returns the
same token, the multiset of tokens is unchanged and the statistic cannot move. dgMARK does
not need the change to be in any particular direction -- its parity is a hash, so any
change re-draws it -- and neither do we.

For every position we record the token first proposed for it and the token finally
committed, together with how many steps it waited. If the change rate is near zero for
deferred positions, the channel is dead on this model regardless of carrier selection, and
that is the finding.
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
import torch.nn.functional as F
from basinmark.model import BasinModel, MASK_ID
from basinmark.data import c4_prompts

GEN, BLK, NS = 256, 32, 4


@torch.no_grad()
def trace(M, p, steps, temperature=0.8, seed=0):
    """Plain reference decoding, logging each position's first and final proposal."""
    gen = torch.Generator(device=M.device).manual_seed(seed)
    pl = p.shape[1]
    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=M.device)
    x[:, :pl] = p.to(M.device)
    first, final, waited = {}, {}, {}
    steps_pb = max(1, steps // (GEN // BLK))
    for b in range(GEN // BLK):
        lo = pl + b * BLK
        Bt = torch.arange(lo, lo + BLK, device=x.device)
        for t in range(steps_pb):
            live = Bt[x[0, Bt] == MASK_ID]
            if live.numel() == 0:
                break
            k = int(np.ceil(live.numel() / (steps_pb - t)))
            logits = M.model(x).logits[0]
            u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64,
                           generator=gen)
            xh = (logits.double() / temperature - torch.log(-torch.log(u))).argmax(-1)
            conf = F.softmax(logits.double(), -1).gather(-1, xh[:, None]).squeeze(1)
            del logits
            liv, cand, cf = live.tolist(), xh[live].tolist(), conf[live].tolist()
            for i, v in zip(liv, cand):
                first.setdefault(i, (v, t))
            order = sorted(range(len(liv)), key=lambda n: -cf[n])[:k]
            for n in order:
                x[0, liv[n]] = cand[n]
                final[liv[n]] = cand[n]
                waited[liv[n]] = t - first[liv[n]][1]
    return x.cpu(), first, final, waited


def main():
    M = BasinModel()
    for steps in (128, 256, 384):
        rows = []
        for i, p in enumerate(c4_prompts(M.tok, NS)):
            _, first, final, waited = trace(M, p, steps, seed=3000 + i)
            w = np.array([waited[k] for k in final])
            ch = np.array([int(first[k][0] != final[k]) for k in final])
            rows.append((w, ch))
        w = np.concatenate([r[0] for r in rows])
        ch = np.concatenate([r[1] for r in rows])
        imm, defr = w == 0, w > 0
        print(f"steps={steps:<4} committed {len(w)}   deferred {defr.mean():.2f} of them "
              f"(mean wait {w[defr].mean() if defr.any() else 0:.1f} steps)", flush=True)
        print(f"           proposal changed:  committed immediately "
              f"{ch[imm].mean() if imm.any() else float('nan'):.3f}   "
              f"after deferral {ch[defr].mean() if defr.any() else float('nan'):.3f}")
        for lo, hi in ((1, 2), (3, 5), (6, 10), (11, 100)):
            m = (w >= lo) & (w <= hi)
            if m.sum() > 20:
                print(f"             waited {lo}-{hi} steps: changed {ch[m].mean():.3f} "
                      f"(n={int(m.sum())})")
    print("\nA change rate near zero after deferral means commit-order steering cannot "
          "move any statistic defined on the tokens, whatever the carrier rule.")


if __name__ == "__main__":
    main()
