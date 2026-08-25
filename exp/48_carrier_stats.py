"""Per-document carrier counts and confidence intervals (exp/48).

Replays the verifier on saved 512-token outputs and records, per document, the carrier
pool size |P| and the keyed match count T. Arms (ids on disk, none regenerated):
  LLaDA  control + R8k10          <- results/23_floor.json  (n=30 each; R8k10 is the
                                     paper's R=8, kappa=0.1 row)
  Dream  control + R16k05         <- results/36_dream.json  (n=30 / n=20; R16k05 is the
                                     paper's strongest Dream row)
Carrier selection depends only on key+text (not R or kappa), so control documents give the
null's carrier counts under the same probes. Writes results/48_carrier_stats.json;
exp/48_plot.py renders the appendix figure.
"""
import sys, json, os, subprocess
sys.path.insert(0, "/ssd2/ming/basinmark")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.resample import ReplayMark

KEY, GEN, BLK = b"retrace-key-A", 512, 32
out = {}

def replay(M, src, arms, nonce_fmt, tag):
    D = json.load(open(src))
    pls = D["pls"]
    for arm in arms:
        rows = []
        for i, raw in enumerate(D["ids"][arm]):
            ids = torch.tensor([raw])
            det = ReplayMark(M, KEY, nonce=nonce_fmt.format(i=i), block_len=BLK,
                               sync_frac=1.0, n_payload_bits=1, s_min=0.5,
                               retries=1).detect(ids, pls[i], GEN, 0)
            rows.append(dict(n=int(det["n_sync"]), hits=int(det["hits_sync"]),
                             rate=float(det["rate_sync"]), p=float(det["p_value"])))
            if (i + 1) % 10 == 0:
                print(f"  [{tag}/{arm}] {i+1}/{len(D['ids'][arm])}", flush=True)
        ns = np.array([r["n"] for r in rows])
        print(f"{tag:<6} {arm:<8} | carriers/doc mean {ns.mean():.1f} sd {ns.std():.1f} "
              f"min {ns.min()} max {ns.max()}", flush=True)
        out[f"{tag}_{arm}"] = rows

from basinmark.model import BasinModel
M = BasinModel()
replay(M, "/ssd2/ming/basinmark/results/23_floor.json", ["control", "R8k10"],
       "fl-{i}", "llada")
del M
torch.cuda.empty_cache()
from basinmark.dream_model import DreamModel
M = DreamModel()
replay(M, "/ssd2/ming/basinmark/results/36_dream.json", ["control", "R16k05"],
       "dr-{i}", "dream")

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, gen=GEN, rows=out),
          open("/ssd2/ming/basinmark/results/48_carrier_stats.json", "w"))
print(f"saved with git {sha[:8]}")
