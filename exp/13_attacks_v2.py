"""Robustness, done properly.

Three things the previous attack table got wrong:
  1. `redenoise()` did mask -> ONE forward -> fill everything back at once. That is
     one-step masked reconstruction, not reverse diffusion. Real re-denoising commits
     progressively, so tokens recovered early become context for the rest and can pull
     the text into the model's natural basin. Run at 1/4/8/16/32 steps.
  2. It ran at L=8,R=3 while the chosen operating point is L=8,R=6.
  3. It reported mean z. "Mean signal survives" and "detection survives" are different
     claims; only TPR at a controlled FPR settles the second. Thresholds come from an
     empirical null of unwatermarked text x keys under the same configuration.
"""
import sys, gzip, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch, torch.nn.functional as F
from basinmark.model import BasinModel, MASK_ID
from basinmark.shared import SharedMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NPR = 256, 40, 16
N_WM, N_NULL_TEXT, N_NULL_KEY = 24, 25, 40
RATES = [0.10, 0.20, 0.30]
STEPS = [1, 4, 8, 16, 32]
CFG = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=20.0,
           commit_steps=2, n_patterns=8, n_ablations=6, pool_rate=0.60)


def c4(tok, n, ntok, skip=0):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                if skip > 0:
                    skip -= 1
                    continue
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


@torch.no_grad()
def iterative_redenoise(M, ids, span, rho, steps, rng):
    """Mask rho of the span, then run `steps` of reverse diffusion over exactly those
    positions with low-confidence progressive commitment. steps=1 reproduces the old
    one-shot reconstruction."""
    k = max(1, int(rho * len(span)))
    pos = rng.choice(span, k, replace=False)
    x = M.corrupt(ids, pos).to(M.device)
    todo = torch.tensor(np.sort(pos), device=M.device)
    for t in range(steps):
        n_left = int((x[0, todo] == MASK_ID).sum())
        if n_left == 0:
            break
        n_commit = int(np.ceil(n_left / (steps - t)))
        logits = M.model(x).logits[0]
        x0 = logits.argmax(-1)
        conf = F.softmax(logits.float(), -1).gather(-1, x0[:, None]).squeeze(1)
        del logits
        masked = torch.full_like(conf, -np.inf)
        masked[todo] = conf[todo]
        masked[x[0] != MASK_ID] = -np.inf
        sel = torch.topk(masked, k=min(n_commit, n_left)).indices
        x[0, sel] = x0[sel]
    return x.cpu()


def delete_attack(ids, span, rho, rng):
    k = max(1, int(rho * len(span)))
    drop = set(rng.choice(span, k, replace=False).tolist())
    keep = [t for t in range(ids.shape[1]) if t not in drop]
    y = ids[:, keep]
    return y, np.arange(span[0], y.shape[1])


def main():
    M = BasinModel()
    wm = SharedMark(M, KEY, **CFG)
    rng = np.random.default_rng(0)
    t0 = time.time()

    # ---- empirical null: unwatermarked text x independent keys ----
    null_z = []
    for i, p in enumerate(c4(M.tok, N_NULL_TEXT, PREFIX, skip=200)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=7000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        for k in range(N_NULL_KEY):
            null_z.append(SharedMark(M, f"nk-{k}".encode(), **CFG).detect(x, span, 0)["z"])
        print(f"[null {i:02d}] {len(null_z)} draws  {time.time()-t0:.0f}s", flush=True)
    null_z = np.array(null_z)
    thr = {a: float(np.quantile(null_z, 1 - a)) for a in (0.05, 0.01, 0.001)}
    print(f"\nnull: n={len(null_z)} mean {null_z.mean():+.3f} sd {null_z.std():.3f}  "
          f"thresholds " + "  ".join(f"FPR{a:g}: z>{v:.2f}" for a, v in thr.items()),
          flush=True)

    # ---- watermarked samples, then attacks ----
    names = [f"redenoise_{s}" for s in STEPS] + ["delete"]
    Z = {n: {r: [] for r in RATES} for n in names}
    clean = []
    for i, p in enumerate(c4(M.tok, N_WM, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        y = wm.embed(x, span, MESSAGE)
        clean.append(wm.detect(y, span, MESSAGE)["z"])
        for r in RATES:
            for s in STEPS:
                Z[f"redenoise_{s}"][r].append(
                    wm.detect(iterative_redenoise(M, y, span, r, s, rng), span, MESSAGE)["z"])
            yd, sd = delete_attack(y, span, r, rng)
            Z["delete"][r].append(wm.detect(yd, sd, MESSAGE)["z"])
        print(f"[wm {i:02d}] clean z {clean[-1]:+.2f}  ({time.time()-t0:.0f}s)", flush=True)

    clean = np.array(clean)

    def tpr(zs, a):
        return float(np.mean(np.array(zs) > thr[a]))

    print(f"\n===== ROBUSTNESS  (L=8, R=6, {len(clean)} watermarked samples) =====")
    print(f"null n={len(null_z)}; thresholds from its upper quantiles")
    for a in (0.05, 0.01, 0.001):
        print(f"\n--- TPR @ FPR={a:g}  (threshold z > {thr[a]:.2f}) ---")
        print(f"{'attack':<14}" + "".join(f"  rho={r:<6.2f}" for r in RATES))
        print(f"{'none':<14}  {tpr(clean, a):.2f}")
        for n in names:
            print(f"{n:<14}" + "".join(f"  {tpr(Z[n][r], a):>10.2f}" for r in RATES))
    print(f"\n--- mean z (secondary) ---   clean {clean.mean():+.2f}")
    print(f"{'attack':<14}" + "".join(f"  rho={r:<6.2f}" for r in RATES))
    for n in names:
        print(f"{n:<14}" + "".join(f"  {np.mean(Z[n][r]):>+10.2f}" for r in RATES))
    json.dump(dict(null_z=null_z.tolist(), clean=clean.tolist(), thr=thr,
                   Z={n: {str(r): v for r, v in d.items()} for n, d in Z.items()}),
              open("/ssd1/ming/basinmark/results/attacks_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
