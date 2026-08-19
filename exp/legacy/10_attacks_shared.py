"""Robustness of SharedMark. The decisive one is `smooth`: the adversary owns the same
dLLM, re-masks part of the text and re-denoises it toward the model's own basin. That is
the natural adversary for a watermark stored in reconstruction behaviour, and none of the
four prior dLLM watermarks face it in this form.

`outside` targets the entropy gate specifically: the carrier is selected by entropy read
from a forward that masks the whole pool, so edits *outside* the pool can reorder the
selection and desynchronise the detector even though no carrier token was touched.
"""
import sys, gzip, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch, torch.nn.functional as F
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 8, 16
POOL, LAM = 0.60, 20.0
RATES = [0.05, 0.10, 0.20, 0.30]


def c4(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


@torch.no_grad()
def redenoise(M, ids, pos, sample=False):
    x = M.corrupt(ids, pos).to(M.device)
    lg = M.model(x).logits[0, torch.tensor(pos)].float()
    out = ids.clone()
    if sample:
        top = lg.topk(5, -1)
        pick = torch.multinomial(F.softmax(top.values, -1), 1).squeeze(1)
        out[0, pos] = top.indices.gather(1, pick[:, None]).squeeze(1).cpu()
    else:
        out[0, pos] = lg.argmax(-1).cpu()
    return out


def main():
    M = BasinModel()
    cfg = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=LAM,
               commit_steps=2, n_patterns=8, n_ablations=3, pool_rate=POOL)
    wm = SharedMark(M, KEY, **cfg)
    rng = np.random.default_rng(0)
    names = ("smooth", "substitute", "outside", "delete")
    res = {a: {r: [] for r in RATES} for a in names}
    clean = []

    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        y = wm.embed(x, span, MESSAGE)
        clean.append(wm.detect(y, span, MESSAGE)["z"])
        pool, _ = wm._carrier(y, span)
        outside = np.setdiff1d(span, pool)
        for r in RATES:
            k = max(1, int(r * len(span)))
            res["smooth"][r].append(wm.detect(
                redenoise(M, y, rng.choice(span, k, replace=False)), span, MESSAGE)["z"])
            res["substitute"][r].append(wm.detect(
                redenoise(M, y, rng.choice(span, k, replace=False), sample=True),
                span, MESSAGE)["z"])
            ko = min(k, len(outside))
            res["outside"][r].append(wm.detect(
                redenoise(M, y, rng.choice(outside, ko, replace=False), sample=True),
                span, MESSAGE)["z"])
            drop = set(rng.choice(span, k, replace=False).tolist())
            keep = [t for t in range(y.shape[1]) if t not in drop]
            yd = y[:, keep]
            res["delete"][r].append(wm.detect(
                yd, np.arange(span[0], yd.shape[1]), MESSAGE)["z"])
        print(f"[{i:02d}] clean z {clean[-1]:+.2f} | " + " ".join(
            f"{a}@{r:.2f} {res[a][r][-1]:+.2f}" for a in names for r in RATES), flush=True)

    print(f"\n===== ATTACKS on SharedMark (mean z; unwatermarked baseline z ~ 0) =====")
    print(f"{'attack':<12}" + "".join(f"  rho={r:<7.2f}" for r in RATES))
    print(f"{'none':<12}  {np.mean(clean):+.2f}")
    for a in names:
        print(f"{a:<12}" + "".join(f"  {np.mean(res[a][r]):>+11.2f}" for r in RATES))
    json.dump(dict(clean=clean, **{a: {str(r): v for r, v in d.items()}
                                   for a, d in res.items()}),
              open("/ssd1/ming/basinmark/results/attacks_shared.json", "w"), indent=1)


if __name__ == "__main__":
    main()
