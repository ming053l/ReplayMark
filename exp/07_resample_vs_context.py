"""What actually supplies the steering capacity: resampling, or context?

The scheduler's waiting step does `continue` -- it commits nothing, so the sequence is
unchanged at the next step, and the only thing that differs is a fresh Gumbel draw. That is
rejection sampling from the *same* conditional, not a re-proposal made under a richer
context. exp/06 measured deferral inside reference decoding, where other positions do get
committed in the meantime, so it could not separate the two, and Figure 2's caption
currently asserts the context story for steps where no context changed.

Two decompositions, both cheap:

  A. same context, fresh noise. One forward gives the logits; R independent Gumbel draws
     cost nothing more. Report the per-draw compatibility rate and P(compatible within R).
     If this is ~1 - 2^-R, the channel is resampling and the method should be described as
     model-response-guided resampling.

  B. context only, no noise. Greedy proposals (temperature 0, so the draw is fixed), then
     commit k other positions and re-read the same position. Any change is attributable to
     context alone.
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
import torch.nn.functional as F
from basinmark.model import BasinModel, MASK_ID
from basinmark.data import c4_prompts
from basinmark.blockmark import BlockMark
from basinmark.challenges import orientation_bits

KEY, GEN, BLK, NS = b"retrace-key-A", 256, 32, 6
RS = (1, 2, 4, 8, 16)
KS = (1, 2, 4, 8, 16)


@torch.no_grad()
def block_state(M, p, frac=0.5, seed=0):
    """Decode up to the middle of the second block and stop, so a partially-filled block
    with real context on both sides is what both probes see."""
    gen = torch.Generator(device=M.device).manual_seed(seed)
    pl = p.shape[1]
    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=M.device)
    x[:, :pl] = p.to(M.device)
    for b in range(2):
        lo = pl + b * BLK
        Bt = torch.arange(lo, lo + BLK, device=x.device)
        target = BLK if b == 0 else int(BLK * frac)
        for t in range(target):
            live = Bt[x[0, Bt] == MASK_ID]
            if live.numel() == 0:
                break
            logits = M.model(x).logits[0]
            u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64,
                           generator=gen)
            xh = (logits.double() / 0.8 - torch.log(-torch.log(u))).argmax(-1)
            conf = F.softmax(logits.double(), -1).gather(-1, xh[:, None]).squeeze(1)
            del logits
            liv = live.tolist()
            n = max(range(len(liv)), key=lambda j: conf[liv[j]].item())
            x[0, liv[n]] = xh[liv[n]]
    return x, pl, pl + BLK


def main():
    M = BasinModel()
    w = BlockMark(M, KEY, block_len=BLK, n_patterns=4, tau_conf=0.3, holes=4,
                  n_payload_bits=7, sync_frac=0.5, challenge="contrast", gap_nats=1.0)
    hitA = {r: [] for r in RS}
    per_draw = []
    hitB = {k: [] for k in KS}
    flipB = {k: [] for k in KS}

    for s_i, p in enumerate(c4_prompts(M.tok, NS)):
        x, pl, lo = block_state(M, p, seed=3000 + s_i)
        span = np.arange(pl, pl + GEN)
        eps = orientation_bits(w.key, span)
        B, g, car = w._table(x.cpu(), lo, pl + GEN)
        gmap = {int(q): g[k] for k, q in enumerate(B)}
        live = [int(q) for q in B if x[0, q] == MASK_ID and car[list(B).index(q)]]
        if not live:
            continue

        # ---------------- A: same context, fresh Gumbel each draw ----------------
        logits = M.model(x).logits[0]
        gen = torch.Generator(device=M.device).manual_seed(99)
        comp = {i: [] for i in live}
        for _ in range(max(RS)):
            u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64,
                           generator=gen)
            xh = (logits.double() / 0.8 - torch.log(-torch.log(u))).argmax(-1)
            for i in live:
                v = int(xh[i])
                gv = float(gmap[i][v])
                comp[i].append(int(gv != 0.0 and eps[i] * gv > 0))
        del logits
        for i in live:
            c = comp[i]
            per_draw.append(np.mean(c))
            for r in RS:
                hitA[r].append(int(any(c[:r])))

        # ---------------- B: context only, greedy proposals ----------------
        xg = x.clone()
        logits = M.model(xg).logits[0]
        base = logits.argmax(-1)
        del logits
        start = {i: int(base[i]) for i in live}
        others = [i for i in live]
        for k in KS:
            xk = x.clone()
            # commit k OTHER masked positions greedily, then re-read each live position
            pool = [q for q in range(pl, pl + GEN) if xk[0, q] == MASK_ID
                    and q not in live]
            for q in pool[:k]:
                lg = M.model(xk).logits[0]
                xk[0, q] = lg[q].argmax()
                del lg
            lg = M.model(xk).logits[0]
            nb = lg.argmax(-1)
            del lg
            ch, fl = [], []
            for i in live:
                v = int(nb[i])
                ch.append(int(v != start[i]))
                gv = float(gmap[i][v])
                g0 = float(gmap[i][start[i]])
                was = g0 != 0.0 and eps[i] * g0 > 0
                now = gv != 0.0 and eps[i] * gv > 0
                fl.append(int((not was) and now))
            hitB[k].append(np.mean(ch)); flipB[k].append(np.mean(fl))
        print(f"[{s_i}] {len(live)} steerable positions probed", flush=True)

    print("\n===== A. same context, fresh Gumbel draw =====")
    print(f"per-draw compatibility {np.mean(per_draw):.3f}   (a fair coin would be 0.500)")
    print(f"{'R':<5}{'P(compatible within R)':>24}{'1 - 2^-R':>12}")
    for r in RS:
        print(f"{r:<5}{np.mean(hitA[r]):>24.3f}{1 - 2.0 ** -r:>12.3f}")

    print("\n===== B. context only, no fresh noise =====")
    print(f"{'k committed':<14}{'proposal changed':>18}{'became compatible':>20}")
    for k in KS:
        print(f"{k:<14}{np.mean(hitB[k]):>18.3f}{np.mean(flipB[k]):>20.3f}")

    print("\nIf A tracks 1 - 2^-R and B stays flat, the steering capacity is resampling "
          "from the same conditional, not order-induced context change.")
    json.dump(dict(per_draw=float(np.mean(per_draw)),
                   A={str(r): float(np.mean(hitA[r])) for r in RS},
                   B_changed={str(k): float(np.mean(hitB[k])) for k in KS},
                   B_flipped={str(k): float(np.mean(flipB[k])) for k in KS}),
              open("/ssd1/ming/basinmark/results/resample_vs_context.json", "w"), indent=1)


if __name__ == "__main__":
    main()
