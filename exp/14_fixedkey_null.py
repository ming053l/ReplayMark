"""Does the deployed decision rule control FPR for a FIXED key?

exp/12 drew a fresh key for every text and KS-tested the pooled p-values, which validates
P_{Y,K} -- not the deployment situation, one secret key over many documents. It also
tested strict uniformity, whereas a randomization p-value only owes super-uniformity,
P(p <= alpha) <= alpha. And it used the Gaussian tail, which measured 0.145 FPR at a
nominal 0.10.

Here: one draft corpus, many keys, FPR reported PER KEY and at the worst key, under the
Hoeffding p-value that detection actually uses. Drafts are cached for reuse by exp/13.
"""
import sys, gzip, json, os, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

GEN, PREFIX, NPR = 256, 40, 16
N_TEXT, N_KEY = 120, 20
CACHE = "/ssd1/ming/basinmark/results/draft_corpus.pt"
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


def build_corpus(M):
    if os.path.exists(CACHE):
        d = torch.load(CACHE)
        print(f"[corpus] loaded {len(d['ids'])} cached drafts", flush=True)
        return d["ids"], d["spans"]
    ids, spans, t0 = [], [], time.time()
    for i, p in enumerate(c4(M.tok, N_TEXT, PREFIX, skip=200)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=7000 + i).cpu()
        ids.append(x)
        spans.append(np.arange(p.shape[1], p.shape[1] + GEN))
        if (i + 1) % 20 == 0:
            print(f"[corpus] {i+1}/{N_TEXT}  {time.time()-t0:.0f}s", flush=True)
    torch.save(dict(ids=ids, spans=spans), CACHE)
    return ids, spans


def main():
    M = BasinModel()
    ids, spans = build_corpus(M)
    per_key = {}
    t0 = time.time()
    for k in range(N_KEY):
        wm = SharedMark(M, f"nullkey-{k}".encode(), **CFG)
        pb, zz = [], []
        for x, sp in zip(ids, spans):
            d = wm.detect(x, sp, 0)
            pb.append(d["p_bound"]); zz.append(d["z"])
        per_key[k] = dict(p_bound=pb, z=zz)
        print(f"[key {k:02d}] FPR@0.05 {np.mean(np.array(pb) < 0.05):.3f} "
              f"FPR@0.01 {np.mean(np.array(pb) < 0.01):.3f} "
              f"mean z {np.mean(zz):+.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n===== FIXED-KEY NULL ({N_KEY} keys x {len(ids)} texts) =====")
    for a in (0.10, 0.05, 0.01):
        f = np.array([np.mean(np.array(v["p_bound"]) < a) for v in per_key.values()])
        print(f"nominal {a:<5} | mean FPR {f.mean():.4f}  worst-key {f.max():.4f}  "
              f"{'VALID (<= nominal)' if f.max() <= a else 'VIOLATED at worst key'}")
    allz = np.concatenate([v["z"] for v in per_key.values()])
    print(f"z over all {len(allz)} null draws: mean {allz.mean():+.3f} sd {allz.std():.3f}")
    json.dump({str(k): v for k, v in per_key.items()},
              open("/ssd1/ming/basinmark/results/fixedkey_null.json", "w"))


if __name__ == "__main__":
    main()
