"""Where does the block-local signal die?

The steering commits ~25% of positions because the model's own preferred token was
compatible, so the compatible fraction of the final text should be ~0.62 and z should be
several units. It measures zero. Three places that can be true, measured separately:

  A. commits do not stick        -- compatible fraction of the FINAL text is ~0.5
  B. commits stick but do not aggregate -- fraction > 0.5 yet Delta stays ~0
  C. the tables differ           -- guidance at generation != guidance at detection,
                                    caught by re-reading g for the committed token
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.blocklocal import BlockMark
from basinmark.prng import payload_bits

KEY, MESSAGE, GEN, BLK = b"basinmark-key-A", 0xA53C7, 256, 32


def compat_fraction(w, ids, pl, gen_len):
    """Fraction of block positions whose FINAL token has s_j * g > 0, using detection-side
    arms. This is the quantity the steering is supposed to move."""
    n_blocks = gen_len // w.block_len
    gen_end = pl + gen_len
    signs = payload_bits(w.key, n_blocks * w.probes_per_block, MESSAGE)
    pos, tot, deltas = 0, 0, []
    for b in range(n_blocks):
        lo = pl + b * w.block_len
        B, S_list, pairs, lp = w._arms(ids, pl, lo, gen_end)
        loc = {int(p): k for k, p in enumerate(B)}
        y = ids[0, torch.tensor(B)]
        at = lp.gather(2, y[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(2)
        for j, S in enumerate(S_list):
            s = signs[b * w.probes_per_block + j]
            for i in S:
                r = loc[int(i)]
                g = float(np.mean([float(at[v, r] - at[u, r]) for u, v in pairs[j]]))
                pos += int(s * g > 0); tot += 1
                deltas.append(s * g)
    return pos / tot, float(np.mean(deltas)), float(np.std(deltas))


def main():
    M = BasinModel()
    prompts = c4_prompts(M.tok, 4)
    for i, p in enumerate(prompts):
        pl = p.shape[1]
        w = BlockMark(M, KEY, block_len=BLK, probes_per_block=2, n_patterns=6,
                      n_ablations=3, tau_conf=0.5, holes=2, nonce=f"doc-{i}")
        ref = M.generate(p, gen_len=GEN, steps=128, block_len=BLK, temperature=0.8,
                         seed=3000 + i).cpu()
        y = w.generate(p, gen_len=GEN, steps=128, temperature=0.8, message=MESSAGE,
                       seed=3000 + i)
        fr_r, mr, sr = compat_fraction(w, ref, pl, GEN)
        fr_w, mw, sw = compat_fraction(w, y, pl, GEN)
        st = w.stats
        wm_driven = st["committed"] - st["fallback"]
        print(f"[{i}] reference  compat {fr_r:.3f}  mean s*g {mr:+.4f} (sd {sr:.3f})",
              flush=True)
        print(f"    watermarked compat {fr_w:.3f}  mean s*g {mw:+.4f} (sd {sw:.3f})")
        print(f"    commits {st['committed']}  watermark-driven {wm_driven} "
              f"({wm_driven/max(st['committed'],1):.2f})  deferred {st['deferred']}")
        print(f"    tokens differing from reference: "
              f"{int((y[0, pl:pl+GEN] != ref[0, pl:pl+GEN]).sum())}/{GEN}", flush=True)


if __name__ == "__main__":
    main()
