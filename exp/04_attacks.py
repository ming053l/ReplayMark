"""Robustness. Includes the attack a functional watermark is uniquely exposed to:
the adversary owns the same dLLM and simply re-denoises the text toward its own basin."""
import sys, gzip, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch, torch.nn.functional as F
from basinmark.model import BasinModel, MASK_ID
from basinmark.core import BasinMark

KEY = b"basinmark-key-A"
N_SAMPLES, GEN, PREFIX = 12, 192, 40
CFG = dict(n_probes=24, probe_rate=0.25, ctx_rate=0.20, tau=4.0, lam=2.0, margin=0.15)
MESSAGE = 0xA53C7
RATES = [0.05, 0.10, 0.20, 0.30]


def c4_prefixes(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out
    return out


@torch.no_grad()
def attack_smooth(M, ids, span, rho, rng):
    """Adversary masks rho of the span and re-denoises with the same model, one shot."""
    k = max(1, int(rho * len(span)))
    pos = rng.choice(span, k, replace=False)
    x = M.corrupt(ids, pos).to(M.device)
    logits = M.model(x).logits
    out = ids.clone()
    out[0, pos] = logits[0, torch.tensor(pos)].argmax(-1).cpu()
    return out


@torch.no_grad()
def attack_substitute(M, ids, span, rho, rng):
    """Replace rho of tokens with a plausible alternative (sampled from the model's
    top-5 given the rest) -- a cheap stand-in for light paraphrasing."""
    k = max(1, int(rho * len(span)))
    pos = rng.choice(span, k, replace=False)
    x = M.corrupt(ids, pos).to(M.device)
    logits = M.model(x).logits[0, torch.tensor(pos)].float()
    top = logits.topk(5, -1)
    pick = torch.multinomial(F.softmax(top.values, -1), 1).squeeze(1)
    out = ids.clone()
    out[0, pos] = top.indices.gather(1, pick[:, None]).squeeze(1).cpu()
    return out


def attack_delete(M, ids, span, rho, rng):
    """Delete rho of tokens; the span shifts, stressing absolute-position patterns."""
    k = max(1, int(rho * len(span)))
    drop = set(rng.choice(span, k, replace=False).tolist())
    keep = [i for i in range(ids.shape[1]) if i not in drop]
    out = ids[:, keep]
    new_span = np.arange(span[0], out.shape[1])
    return out, new_span


def main():
    M = BasinModel()
    wm = BasinMark(M, KEY, **CFG)
    prefixes = c4_prefixes(M.tok, N_SAMPLES, PREFIX)
    rng = np.random.default_rng(0)
    res = {a: {r: [] for r in RATES} for a in ("smooth", "substitute", "delete")}
    clean = []
    for i, p in enumerate(prefixes):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32,
                       temperature=0.8, seed=1000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        y = wm.embed(x, span, MESSAGE, rounds=3)
        clean.append(wm.detect(y, span, MESSAGE)["matches"])
        for r in RATES:
            res["smooth"][r].append(wm.detect(attack_smooth(M, y, span, r, rng), span, MESSAGE)["matches"])
            res["substitute"][r].append(wm.detect(attack_substitute(M, y, span, r, rng), span, MESSAGE)["matches"])
            yd, sd = attack_delete(M, y, span, r, rng)
            res["delete"][r].append(wm.detect(yd, sd, MESSAGE)["matches"])
        print(f"[{i:02d}] clean {clean[-1]}/24 | " + " | ".join(
            f"{a}@{r:.2f} {res[a][r][-1]}" for a in res for r in RATES), flush=True)

    print("\n===== ATTACKS (mean sign-matches out of 24; chance = 12) =====")
    print(f"{'attack':<12}" + "".join(f"  rho={r:<6.2f}" for r in RATES))
    print(f"{'none':<12}  {np.mean(clean):.1f}")
    for a in res:
        print(f"{a:<12}" + "".join(f"  {np.mean(res[a][r]):>10.1f}" for r in RATES))
    json.dump(dict(clean=clean, **{a: {str(r): v for r, v in d.items()} for a, d in res.items()}),
              open("/ssd1/ming/basinmark/results/attacks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
