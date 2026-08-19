"""Robustness of the frozen config, on exp/19's held-out outputs, LLR detector.
Attacks: random substitution (same-model one-step redenoise at rho of positions) and
deletion. Control arm attacked identically = the null under attack."""
import sys, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel, MASK_ID
from basinmark.resample import ResampleMark
from basinmark.llr import build_cache, llr_pvalue

KEY, BLK, ND = b"retrace-key-A", 32, 20
d = json.load(open("/ssd1/ming/basinmark/results/19_freeze.json"))
GEN, pls = d["gen"], d["pls"]
M = BasinModel()
rng = np.random.default_rng(1)

@torch.no_grad()
def redenoise(ids, span, rho):
    k = max(1, int(rho * len(span)))
    pos = rng.choice(span, k, replace=False)
    x = ids.clone(); x[0, torch.tensor(pos)] = MASK_ID
    lg = M.model(x.to(M.device)).logits[0, torch.tensor(pos)]
    y = ids.clone(); y[0, torch.tensor(pos)] = lg.argmax(-1).cpu()
    return y

def detect(ids, i):
    det = ResampleMark(M, KEY, block_len=BLK, sync_frac=1.0, n_payload_bits=1,
                       s_min=0.5, retries=1, nonce=f"ho-{i}")
    c = build_cache(det, ids, pls[i], GEN)
    return llr_pvalue(c, det.key, pls[i], GEN, R=1, n_mc=50_000)["p_value"]

for arm in ("R1", "control"):
    ids_l = d["ids"][arm][:ND]
    for atk, rho in (("none", 0), ("redenoise", .1), ("redenoise", .2), ("delete", .1)):
        ps = []
        for i, il in enumerate(ids_l):
            y = torch.tensor(il, dtype=torch.long)[None]
            span = np.arange(pls[i], pls[i] + GEN)
            if atk == "redenoise":
                y = redenoise(y, span, rho)
            elif atk == "delete":
                k = max(1, int(rho * len(span)))
                drop = set(rng.choice(span, k, replace=False).tolist())
                keep = [t for t in range(y.shape[1]) if t not in drop]
                y = y[:, keep][:, :pls[i] + GEN]
                if y.shape[1] < pls[i] + GEN:      # pad by repeating final token
                    pad = y[0, -1].repeat(pls[i] + GEN - y.shape[1])
                    y = torch.cat([y, pad[None]], 1)
            ps.append(detect(y, i))
        ps = np.array(ps)
        print(f"{arm:<8} {atk:<10} rho={rho:<4} | TPR@5% {np.mean(ps<.05):.2f} "
              f"@1% {np.mean(ps<.01):.2f} @0.1% {np.mean(ps<.001):.2f}", flush=True)
print("control rows must stay at ~alpha under every attack, or the attack itself is "
      "triggering the detector")
