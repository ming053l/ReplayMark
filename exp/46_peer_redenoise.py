"""Re-denoising attack on the PEER watermarks (exp/46) — the attack no key-only dLLM
watermark paper reports.

Fresh LLaDA @512 generations with ids kept in-process (retokenization would shift
positions and unfairly break the peers' position-keyed statistics):
  kgw   : kgw_generate delta=1, n=20, seeds 46000+i
  dgmark: WatermarkGenerator.generate_watermark_multinomial top-k 3, n=20
Attack: same-model argmax re-denoising of a random 10% of generated positions (identical
to the attack run against ReplayMark in exp/33). Report each detector's TPR@5%/1% clean
vs attacked, using each method's own statistic (dedup bigram z; parity match z).
"""
import sys, json, os, subprocess
sys.path.insert(0, "/ssd2/ming/basinmark")
sys.path.insert(0, "/ssd2/ming/basinmark/baselines/dgmark-watermarking/src")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.kgw import kgw_generate, kgw_detect
from basinmark.data import c4_prompts
from generation import WatermarkGenerator
from utils import check_watermark_compliance

KEY, GEN, N = b"kgw-key", 512, 20
M = BasinModel()
DG = WatermarkGenerator(M.model, M.tok, "cuda", mask_id=M.mask_id, private_key=None)
prompts = c4_prompts(M.tok, N, skip=2100)
pls = [p.shape[1] for p in prompts]
rng = np.random.default_rng(4600)

@torch.no_grad()
def redenoise(ids, p_len, frac=0.10):
    ids = ids.clone()
    span = np.arange(p_len, p_len + GEN)
    pos = rng.choice(span, max(1, int(frac * GEN)), replace=False)
    x = ids.clone()
    x[0, torch.tensor(pos)] = M.mask_id
    lp = M.logprobs_rows(x, torch.tensor(np.sort(pos)), chunk=2)
    ids[0, torch.tensor(np.sort(pos))] = lp[0].argmax(-1).cpu()
    return ids

def dg_z(y, pl):
    ids = y[0, pl:pl + GEN].tolist()
    try:
        ids = ids[:ids.index(126081) + 1]
    except ValueError:
        pass
    hits = sum(check_watermark_compliance(pl + j + 1, t, None)
               for j, t in enumerate(ids))
    n = len(ids)
    return (hits - n / 2) / np.sqrt(max(n, 1) / 4)

res = {}
for name in ("kgw", "dgmark"):
    zc, za = [], []
    for i, p in enumerate(prompts):
        if name == "kgw":
            y = kgw_generate(M, p, gen_len=GEN, delta=1.0, key=KEY, temperature=0.8,
                             seed=46000 + i)
            span = np.arange(pls[i], pls[i] + GEN)
            zc.append(kgw_detect(M, y, span, key=KEY, dedup=True)["z"])
            za.append(kgw_detect(M, redenoise(y, pls[i]), span, key=KEY,
                                 dedup=True)["z"])
        else:
            y = DG.generate_watermark_multinomial(p.cuda(), steps=GEN, gen_length=GEN,
                                                  block_length=32, top_k=3).cpu()
            zc.append(dg_z(y, pls[i]))
            za.append(dg_z(redenoise(y, pls[i]), pls[i]))
        if (i + 1) % 5 == 0:
            print(f"  [{name}] {i+1}/{N}", flush=True)
    zc, za = np.array(zc), np.array(za)
    res[name] = dict(z_clean=zc.tolist(), z_attacked=za.tolist())
    print(f"{name:<7}| clean  TPR@5% {np.mean(zc>1.645):.2f} @1% {np.mean(zc>2.326):.2f}"
          f" (z {zc.mean():+.2f}) | rd10 TPR@5% {np.mean(za>1.645):.2f} "
          f"@1% {np.mean(za>2.326):.2f} (z {za.mean():+.2f})", flush=True)

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, n=N, gen=GEN, res=res),
          open("/ssd2/ming/basinmark/results/46_peer_redenoise.json", "w"))
print(f"saved with git {sha[:8]}")
