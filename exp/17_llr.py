"""LLR vs indicator, offline on exp/16's saved outputs. Zero new generation."""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.resample import ResampleMark
from basinmark.llr import build_cache, llr_pvalue

KEY, BLK = b"retrace-key-A", 32
CACHES = {}                       # (arm, i) -> per-position arrays, for offline iteration
# 16's in-flight process predated the save patch, so its outputs were never written;
# 19's held-out outputs carry the same three-arm structure minus R2
d = json.load(open("/ssd1/ming/basinmark/results/19_freeze.json"))
pls, GEN, saved = d["pls"], d["gen"], d["ids"]
M = BasinModel()
Rmap = {"control": 1, "R1": 1, "R2": 2}
for name, docs in saved.items():
    t0, pv_llr, pv_ind = time.time(), [], []
    for i, ids_list in enumerate(docs):
        ids = torch.tensor(ids_list, dtype=torch.long)[None]
        w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, retries=1, sync_frac=1.0,
                         n_payload_bits=1, nonce=f"ho-{i}")
        cache = build_cache(w, ids, pls[i], GEN)
        CACHES[f"{name}:{i}"] = cache
        pv_llr.append(llr_pvalue(cache, w.key, pls[i], GEN, R=Rmap[name])["p_value"])
        pv_ind.append(w.detect(ids, pls[i], GEN, 0)["p_value"])
    a, b = np.array(pv_ind), np.array(pv_llr)
    print(f"{name:<8} n={len(docs)} | indicator TPR@5% {np.mean(a<.05):.2f} "
          f"@1% {np.mean(a<.01):.2f} @0.1% {np.mean(a<.001):.2f} | "
          f"LLR TPR@5% {np.mean(b<.05):.2f} @1% {np.mean(b<.01):.2f} "
          f"@0.1% {np.mean(b<.001):.2f} | {(time.time()-t0)/len(docs):.0f}s/doc", flush=True)
json.dump(dict(caches=CACHES, pls=pls, GEN=GEN),
          open("/ssd1/ming/basinmark/results/17_caches.json", "w"))
print("control rows are the null check: both detectors should sit at ~alpha")
print("caches saved: every further detector variant iterates offline, no GPU")
