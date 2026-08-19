"""LLR vs indicator, offline on exp/16's saved outputs. Zero new generation."""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.resample import ResampleMark
from basinmark.llr import build_cache, llr_pvalue

KEY, BLK = b"retrace-key-A", 32
d = json.load(open("/ssd1/ming/basinmark/results/16_outputs.json"))
pls, GEN, saved = d["pls"], d["GEN"], d["saved"]
M = BasinModel()
Rmap = {"control": 1, "R1": 1, "R2": 2}
for name, docs in saved.items():
    t0, pv_llr, pv_ind = time.time(), [], []
    for i, ids_list in enumerate(docs):
        ids = torch.tensor(ids_list, dtype=torch.long)[None]
        w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, retries=1, sync_frac=1.0,
                         n_payload_bits=1, nonce=f"doc-{i}")
        cache = build_cache(w, ids, pls[i], GEN)
        pv_llr.append(llr_pvalue(cache, w.key, pls[i], GEN, R=Rmap[name])["p_value"])
        pv_ind.append(w.detect(ids, pls[i], GEN, 0)["p_value"])
    a, b = np.array(pv_ind), np.array(pv_llr)
    print(f"{name:<8} n={len(docs)} | indicator TPR@5% {np.mean(a<.05):.2f} "
          f"@1% {np.mean(a<.01):.2f} @0.1% {np.mean(a<.001):.2f} | "
          f"LLR TPR@5% {np.mean(b<.05):.2f} @1% {np.mean(b<.01):.2f} "
          f"@0.1% {np.mean(b<.001):.2f} | {(time.time()-t0)/len(docs):.0f}s/doc", flush=True)
print("control rows are the null check: both detectors should sit at ~alpha")
