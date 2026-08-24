"""Wall-clock verification cost (exp/45): seconds per 512-token document, measured.

Shibboleth: full ResampleMark.detect replay (16 blocks x 9 masked evaluations, chunk=2)
on GPU, averaged over 5 saved documents after one warm-up. KGW and dgMARK: their
text-and-key detectors on the same documents (hash lookups only, no model).
"""
import sys, json, time, os
sys.path.insert(0, "/ssd2/ming/basinmark")
sys.path.insert(0, "/ssd2/ming/basinmark/baselines/dgmark-watermarking/src")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.resample import ResampleMark
from basinmark.kgw import kgw_detect
from utils import check_watermark_compliance

KEY, GEN, BLK, N = b"retrace-key-A", 512, 32, 5
D = json.load(open("/ssd2/ming/basinmark/results/23_floor.json"))
M = BasinModel()

docs = [torch.tensor([D["ids"]["control"][i]]) for i in range(N + 1)]
pls = D["pls"]

# warm-up
ResampleMark(M, KEY, nonce="fl-0", block_len=BLK, sync_frac=1.0, n_payload_bits=1,
             s_min=0.5, retries=1).detect(docs[0], pls[0], GEN, 0)
torch.cuda.synchronize()

ours = []
for i in range(1, N + 1):
    t0 = time.time()
    ResampleMark(M, KEY, nonce=f"fl-{i}", block_len=BLK, sync_frac=1.0,
                 n_payload_bits=1, s_min=0.5, retries=1).detect(docs[i], pls[i], GEN, 0)
    torch.cuda.synchronize()
    ours.append(time.time() - t0)

kgw = []
for i in range(1, N + 1):
    t0 = time.time()
    kgw_detect(M, docs[i], np.arange(pls[i], pls[i] + GEN), key=KEY, dedup=True)
    kgw.append(time.time() - t0)

dg = []
for i in range(1, N + 1):
    ids = docs[i][0, pls[i]:pls[i] + GEN].tolist()
    t0 = time.time()
    sum(check_watermark_compliance(pls[i] + j + 1, t, None) for j, t in enumerate(ids))
    dg.append(time.time() - t0)

out = dict(n=N, gen=GEN,
           shibboleth_s=dict(mean=float(np.mean(ours)), sd=float(np.std(ours))),
           kgw_s=dict(mean=float(np.mean(kgw)), sd=float(np.std(kgw))),
           dgmark_s=dict(mean=float(np.mean(dg)), sd=float(np.std(dg))))
print(json.dumps(out, indent=1))
json.dump(out, open("/ssd2/ming/basinmark/results/45_timing.json", "w"))
print(f"shibboleth {np.mean(ours):.2f}s | kgw {np.mean(kgw)*1000:.1f}ms | "
      f"dgmark {np.mean(dg)*1000:.1f}ms per 512-token doc")
