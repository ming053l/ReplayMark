"""Are the generation-time and detection-time tables the same number?

Roughly half of all commits are made because eps*want*g(candidate) > 0 at commit time, yet
the final agreement with the intended direction is 0.50. Either the committed token does
not survive, or the two sides read different values of g for the same (position, token).
Stop reasoning about it: record g at the moment of commit, recompute it at detection, and
print the disagreement rate directly.
"""
import sys
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
import torch.nn.functional as F
from basinmark.model import BasinModel, MASK_ID
from basinmark.data import c4_prompts
from basinmark.blockmark import BlockMark
from basinmark.prng import stream

KEY, MESSAGE, GEN, BLK = b"basinmark-key-A", 0xA5, 256, 32


def main():
    M = BasinModel()
    p = c4_prompts(M.tok, 1)[0]
    pl = p.shape[1]
    w = BlockMark(M, KEY, block_len=BLK, n_patterns=4, tau_conf=0.3, holes=4,
                  n_bits=8, challenge="contrast", nonce="doc-0")

    # --- generation, logging g at the instant of every commit ---
    log = {}
    gen = torch.Generator(device=M.device).manual_seed(3000)
    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=M.device)
    x[:, :pl] = p.to(M.device)
    gen_end = pl + GEN
    steps_pb = 256 // (GEN // BLK)
    for b in range(GEN // BLK):
        lo = pl + b * BLK
        B, g = w._table(x.cpu(), lo, gen_end)
        gmap = {int(q): g[k] for k, q in enumerate(B)}
        Bt = torch.tensor(B, device=x.device)
        for t in range(steps_pb):
            live = Bt[x[0, Bt] == MASK_ID]
            if live.numel() == 0:
                break
            k = int(np.ceil(live.numel() / (steps_pb - t)))
            logits = M.model(x).logits[0]
            u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64,
                           generator=gen)
            xh = (logits.double() / 0.8 - torch.log(-torch.log(u))).argmax(-1)
            conf = F.softmax(logits.double(), -1).gather(-1, xh[:, None]).squeeze(1)
            del logits
            liv, cand = live.tolist(), xh[live].tolist()
            cf = conf[live].tolist()
            order = sorted(range(len(liv)), key=lambda n: -cf[n])[:k]
            for n in order:
                x[0, liv[n]] = cand[n]
                log[liv[n]] = (cand[n], float(gmap[liv[n]][cand[n]]), b)
    y = x.cpu()

    # --- detection side, recomputing the same tables ---
    bad = same = 0
    worst = []
    for b in range(GEN // BLK):
        lo = pl + b * BLK
        B, g = w._table(y, lo, gen_end)
        for kk, q in enumerate(B):
            tok, gv_gen, bb = log[int(q)]
            gv_det = float(g[kk][tok])
            if abs(gv_gen - gv_det) > 1e-9:
                bad += 1
                worst.append((int(q), bb, gv_gen, gv_det))
            else:
                same += 1
    print(f"positions where generation and detection agree on g: {same}/{same+bad}")
    print(f"disagree: {bad}  (sign flips: "
          f"{sum(1 for _,_,a,c in worst if a*c < 0)})")
    for q, bb, a, c in worst[:8]:
        print(f"  pos {q} (block {bb}): gen {a:+.6f}  det {c:+.6f}")


if __name__ == "__main__":
    main()
