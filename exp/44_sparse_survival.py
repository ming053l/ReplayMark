"""Sparse survival IN PLACE (exp/44): contiguous re-denoise of 50%/75% of positions.

Only 25-50% of the document keeps its watermark, at its original offsets -- the regime
where a per-block min-statistic (Bonferroni) could finally beat the pooled count, without
confounding from position shifts. Same texts/arms/aggregations as exp/33 and exp/33b.

The measured fragility (10% re-denoise: 0.35 -> 0.05 at R=1) was under the POOLED count,
which dilutes intact blocks with damaged ones. Detection already computes per-block
(hits, n); this evaluates two block-local aggregations with exact validity on the same
attacked texts:
  pooled  : Binomial tail over all carriers (deployed detector)
  bonf    : min over blocks of the per-block Binomial tail, times #blocks (Bonferroni --
            exact, conservative; wins when a few blocks stay intact and strong)
  stouffer: sum of per-block z's / sqrt(#blocks) (asymptotic; reported for reference)
No new generation: texts are results/29_clean.json (1024 tok, nonce g2-{i}); attacks are
same-model argmax re-denoising of a random 5%/10% of generated positions. The control arm
goes through the identical pipeline so every aggregation's FPR is checked, not assumed."""
import sys, json, subprocess, os
sys.path.insert(0, "/ssd2/ming/basinmark")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from scipy.stats import binom, norm
from basinmark.model import BasinModel
from basinmark.resample import ResampleMark, MASK_ID

KEY, GEN, BLK = b"retrace-key-A", 1024, 32
D = json.load(open("/ssd2/ming/basinmark/results/29_clean.json"))
pls = D["pls"]
M = BasinModel()
rng = np.random.default_rng(4400)

@torch.no_grad()
def redenoise(ids, p_len, frac):
    ids = ids.clone()
    k = max(1, int(frac * GEN))
    start = p_len + int(rng.integers(0, GEN - k + 1))
    pos = np.arange(start, start + k)
    x = ids.clone()
    x[0, torch.tensor(pos)] = MASK_ID
    lp = M.logprobs_rows(x, torch.tensor(np.sort(pos)), chunk=2)
    ids[0, torch.tensor(np.sort(pos))] = lp[0].argmax(-1).cpu()
    return ids

def aggregate(det):
    bl = [(h, n) for h, n in det["blocks"] if n > 0]
    ps = [float(binom.sf(h - 1, n, 0.5)) for h, n in bl]
    nb = len(bl)
    bonf = min(1.0, min(ps) * nb) if nb else 1.0
    zs = [(h - n / 2) / np.sqrt(n / 4) for h, n in bl]
    stf = float(1 - norm.cdf(sum(zs) / np.sqrt(nb))) if nb else 1.0
    return dict(pooled=det["p_value"], bonf=bonf, stouffer=stf)

out = {}
for arm in ("control", "R16k05"):
    rows = []
    for i, raw in enumerate(D["ids"][arm]):
        ids = torch.tensor([raw])
        det = ResampleMark(M, KEY, nonce=f"g2-{i}", block_len=BLK, sync_frac=1.0,
                           n_payload_bits=1, s_min=0.5, retries=1)
        rec = {}
        for tag, t in (("clean", ids), ("loc50", redenoise(ids, pls[i], 0.50)),
                       ("loc75", redenoise(ids, pls[i], 0.75))):
            rec[tag] = aggregate(det.detect(t, pls[i], GEN, 0))
        rows.append(rec)
        if (i + 1) % 4 == 0:
            print(f"  [{arm}] {i+1}/{len(D['ids'][arm])}", flush=True)
    out[arm] = rows
    for tag in ("clean", "loc50", "loc75"):
        for agg in ("pooled", "bonf", "stouffer"):
            ps = np.array([r[tag][agg] for r in rows])
            print(f"{arm:<8} {tag:<6} {agg:<9} TPR@5% {np.mean(ps<.05):.2f} "
                  f"@1% {np.mean(ps<.01):.2f} @0.1% {np.mean(ps<.001):.2f}", flush=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, source="29_clean", attack="contiguous-majority", rows=out),
          open("/ssd2/ming/basinmark/results/44_sparse.json", "w"))
print(f"saved with git {sha[:8]}")
