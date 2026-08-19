"""Can a better-chosen challenge make the contrast two-sided?

exp/08 measured the quantity that decides whether resampling can work at all: the
two-sided acceptance mass S_i = 2 min(q+, q-) under the model's own sampling distribution.
Its median over selected carriers is 0.000. At most positions every token the model would
plausibly emit falls on the SAME side of the contrast, so when the key asks for the other
side no number of retries helps -- and because an exhausted position emits a rejected
draw, it scores a deterministic miss rather than a coin flip. Acceptance below 0.5 makes
the statistic worse than the null, which is what the predicted 0.34 at R=8 implies.

The contrast is currently a fixed pair: the context nearest the block against the furthest.
This asks whether any pair among L keyed patterns does better, which is the last lever that
could make the observable steerable at all:

    S_fixed   the near/far pair now in use
    S_best    the best of the L(L-1) ordered pairs, chosen per position
    S_oracle  an upper bound: the best pair per position, ignoring that one pair must
              serve a whole probe

Selecting a pair by S is orientation-symmetric and never looks at the key's direction or
the message, so a content-chosen pair would keep the conditional null intact. The question
here is only whether the headroom exists.
"""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch

from basinmark.model import BasinModel, MASK_ID
from basinmark.data import c4_prompts
from basinmark.challenges import block_challenges

KEY, GEN, BLK, NS, L = b"retrace-key-A", 256, 32, 6, 8
TEMP = 0.8


@torch.no_grad()
def main():
    M = BasinModel()
    fixed, best, oracle, frac_two_sided = [], [], [], []
    for s_i, p in enumerate(c4_prompts(M.tok, NS)):
        pl = p.shape[1]
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=BLK, temperature=0.8,
                       seed=3000 + s_i).cpu()
        for b in range(GEN // BLK):
            lo = pl + b * BLK
            B = np.arange(lo, lo + BLK)
            pats, _ = block_challenges(KEY, lo, BLK, L, 0.20, mode="contrast")
            base = x.clone()
            base[0, lo:pl + GEN] = MASK_ID
            batch = []
            for d in pats:
                m = base.clone()
                m[0, torch.tensor(d)] = MASK_ID
                batch.append(m)
            batch.append(base)
            lp = M.logprobs_rows(torch.cat(batch, 0), torch.tensor(B), chunk=2,
                                 dtype=torch.float64)
            pr = torch.softmax(lp[-1] / TEMP, dim=-1)          # sampling distribution

            def S_of(u, v):
                g = lp[v] - lp[u]
                qp = (pr * (g > 0)).sum(1).numpy()
                qm = (pr * (g < 0)).sum(1).numpy()
                return 2.0 * np.minimum(qp, qm)

            s_fix = S_of(0, 1)                                  # near vs far, as shipped
            allS = np.stack([S_of(u, v) for u in range(L) for v in range(L) if u != v])
            fixed.append(s_fix)
            best.append(allS.max(0))
            oracle.append(allS.max(0))
            frac_two_sided.append((allS.max(0) > 0.2).mean())
        print(f"[{s_i}] done", flush=True)

    f = np.concatenate(fixed); bst = np.concatenate(best)
    print("\n===== two-sided acceptance mass S = 2 min(q+, q-) =====")
    print(f"{'':<14}{'median':>9}{'mean':>9}{'>0.1':>8}{'>0.2':>8}{'>0.5':>8}")
    for name, a in (("fixed pair", f), ("best of L(L-1)", bst)):
        print(f"{name:<14}{np.median(a):>9.3f}{a.mean():>9.3f}"
              f"{(a > 0.1).mean():>8.3f}{(a > 0.2).mean():>8.3f}{(a > 0.5).mean():>8.3f}")
    print(f"\npositions per block with a usable pair (S > 0.2): "
          f"{np.mean(frac_two_sided) * BLK:.1f} of {BLK}")
    for R in (1, 2, 4, 8):
        pf = float(np.mean(1 - (1 - f / 2) ** R))
        pb = float(np.mean(1 - (1 - bst / 2) ** R))
        print(f"  predicted acceptance at R={R}:  fixed {pf:.3f}   best-pair {pb:.3f}")
    print("\nAcceptance must exceed 0.5 for the statistic to beat its own null, because an "
          "exhausted position emits a rejected draw and scores a certain miss.")
    json.dump(dict(fixed=f.tolist(), best=bst.tolist()),
              open("/ssd1/ming/basinmark/results/challenge_selection.json", "w"))


if __name__ == "__main__":
    main()
