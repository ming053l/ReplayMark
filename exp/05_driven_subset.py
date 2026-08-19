"""Do the watermark-driven commits themselves land?

Everything measured so far is global: 44 % of commits are made because the model's
preferred token answered the challenge, yet overall agreement with the intended direction
is 0.50. The tables are not the problem -- exp/03 shows generation and detection read the
same g at 256/256 positions. What has never been measured is the hit rate restricted to
the positions the watermark actually drove.

  driven subset ~= 1.0  -> the commits land, and the global figure or the counter is wrong
  driven subset ~= 0.5  -> the commit path does not do what the accounting says
"""
import sys
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
import torch.nn.functional as F
from basinmark.model import BasinModel, MASK_ID
from basinmark.data import c4_prompts
from basinmark.blockmark import BlockMark
from basinmark.challenges import orientation_bits, tie_bits, score
from basinmark.prng import stream

KEY, MESSAGE, GEN, BLK, STEPS = b"basinmark-key-A", 0xA5, 256, 32, 256


def main():
    M = BasinModel()
    for s_i, p in enumerate(c4_prompts(M.tok, 3)):
        pl = p.shape[1]
        w = BlockMark(M, KEY, block_len=BLK, n_patterns=4, tau_conf=0.3, holes=4,
                      n_bits=8, challenge="contrast", nonce=f"doc-{s_i}")
        span = np.arange(pl, pl + GEN)
        eps = orientation_bits(w.key, span)
        tie = tie_bits(w.key, span)
        grp = stream(w.key, "grp", GEN, w.n_bits).integers(0, w.n_bits, size=GEN)
        want = {int(t): (1 if ((MESSAGE >> int(grp[int(t) - pl])) & 1) else -1)
                for t in span}

        gen = torch.Generator(device=M.device).manual_seed(3000 + s_i)
        x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=M.device)
        x[:, :pl] = p.to(M.device)
        gen_end = pl + GEN
        steps_pb = STEPS // (GEN // BLK)
        driven = {}
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
                liv, cand, cf = live.tolist(), xh[live].tolist(), conf[live].tolist()
                ok = [(eps[i] * want[i] * float(gmap[i][v]) > 0)
                      if float(gmap[i][v]) != 0.0 else False
                      for i, v in zip(liv, cand)]
                elig = [n for n, c in enumerate(cf) if c >= 0.3]
                eset = set(liv[n] for n in elig)
                safe = [n for n in elig
                        if sum(1 for jj in liv if jj < liv[n] and jj not in eset) <= 4]
                pick = [n for n in safe if ok[n]][:k]
                nd = len(pick)
                if len(pick) < k:
                    pick += sorted((n for n in range(len(liv)) if n not in pick),
                                   key=lambda n: -cf[n])[:k - len(pick)]
                for idx, n in enumerate(pick):
                    x[0, liv[n]] = cand[n]
                    driven[liv[n]] = (idx < nd, cand[n])
        y = x.cpu()

        # recompute the indicator from the finished text
        hit_d = tot_d = hit_o = tot_o = 0
        mism = []
        for b in range(GEN // BLK):
            lo = pl + b * BLK
            B, g = w._table(y, lo, gen_end)
            yv = y[0, torch.tensor(B)]
            gv = g.gather(1, yv[:, None]).squeeze(1).numpy()
            e = np.array([eps[int(t)] for t in B], float)
            tb = np.array([tie[int(t)] for t in B], np.int64)
            m, _ = score(gv, e, tb)
            for k2, t in enumerate(B):
                was_driven, tok = driven[int(t)]
                hit = int(m[k2] == (1 if want[int(t)] > 0 else 0))
                if was_driven:
                    hit_d += hit; tot_d += 1
                    if not hit and len(mism) < 3:
                        mism.append((int(t), tok, int(yv[k2]), float(gv[k2])))
                else:
                    hit_o += hit; tot_o += 1
        print(f"[{s_i}] watermark-driven positions: {hit_d}/{tot_d} = "
              f"{hit_d/max(tot_d,1):.3f}   others: {hit_o}/{tot_o} = "
              f"{hit_o/max(tot_o,1):.3f}", flush=True)
        for t, tok, got, gval in mism:
            print(f"    miss at {t}: committed {tok}, final {got}, g {gval:+.5f}"
                  f"{'  TOKEN CHANGED' if tok != got else ''}")


if __name__ == "__main__":
    main()
