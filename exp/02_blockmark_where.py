"""Do watermark-driven commits actually land?

The sweep says 24-34% of commits were made because the model's own token already answered
the challenge, yet bit accuracy sits at 0.55. Those two cannot both be true unless the
committed positions fail to show the property at detection time. Measure the match rate
separately on the positions the watermark drove and on the rest; if the driven ones are
also ~0.5, the commit and the detector disagree about the same position, and the table is
still not identical between the two.
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.blockmark import BlockMark
from basinmark.challenges import orientation_bits, tie_bits, score
from basinmark.prng import stream

KEY, MESSAGE, GEN, BLK = b"basinmark-key-A", 0xA5, 256, 32


def main():
    M = BasinModel()
    for i, p in enumerate(c4_prompts(M.tok, 3)):
        pl = p.shape[1]
        w = BlockMark(M, KEY, block_len=BLK, n_patterns=4, tau_conf=0.3, holes=4,
                      n_bits=8, challenge="contrast", nonce=f"doc-{i}")
        w.driven = set()
        # wrap the commit so we learn which positions the watermark actually chose
        orig = w.generate
        y = orig(p, gen_len=GEN, steps=256, temperature=0.8, message=MESSAGE,
                 seed=3000 + i)
        d = w.detect(y, pl, GEN, MESSAGE)

        span = np.arange(pl, pl + GEN)
        eps = orientation_bits(w.key, span)
        tie = tie_bits(w.key, span)
        grp = stream(w.key, "grp", GEN, w.n_bits).integers(0, w.n_bits, size=GEN)
        want = {int(t): (1 if ((MESSAGE >> int(grp[int(t) - pl])) & 1) else -1)
                for t in span}
        gen_end = pl + GEN
        agree, n, tied = 0, 0, 0
        per_bit = {b: [0, 0] for b in range(w.n_bits)}
        for b in range(GEN // BLK):
            lo = pl + b * BLK
            B, g = w._table(y, lo, gen_end)
            yv = y[0, torch.tensor(B)]
            gv = g.gather(1, yv[:, None]).squeeze(1).numpy()
            e = np.array([eps[int(t)] for t in B], float)
            tb = np.array([tie[int(t)] for t in B], np.int64)
            m, nz = score(gv, e, tb)
            tied += nz
            for k, t in enumerate(B):
                j = int(grp[int(t) - pl])
                hit = int(m[k] == (1 if want[int(t)] > 0 else 0))
                agree += hit; n += 1
                per_bit[j][0] += hit; per_bit[j][1] += 1
        st = w.stats
        print(f"[{i}] agreement with the intended direction {agree}/{n} = {agree/n:.3f} "
              f"(chance 0.500)", flush=True)
        print(f"    ties {tied}/{n} = {tied/n:.3f}   wm-driven commits "
              f"{st['wm']}/{st['committed']} = {st['wm']/max(st['committed'],1):.3f}")
        print(f"    per-bit agreement: "
              + " ".join(f"{v[0]/max(v[1],1):.2f}" for v in per_bit.values()))
        print(f"    bit_acc from detect {d['bit_acc']:.2f}  match {d['rate']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
