"""Carrier-suitability visualization data: where in a real text could the watermark live?

CPU-only replay of the Reproducible Response Bank on saved 512-token CONTROL outputs
(results/23_floor.json, control arm, nonce fl-{i}) — the same computation the verifier
runs, so the map shows exactly what the embedder would have seen. Per position: the
admission score S (two-sided response mass), q+ and q-, and the live base conditional's
entropy. Also records two showcase positions from doc 0 (strongest carrier vs a sharp
one-sided position): top-12 tokens of the block-masked conditional with response signs.

Writes results/43_viz.json. Runs on CPU so it can proceed while the GPU chain is busy.
"""
import sys, json, os
sys.path.insert(0, "/ssd2/ming/basinmark")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
torch.set_num_threads(16)
from basinmark.model import BasinModel
from basinmark.resample import ReplayMark

KEY, GEN, BLK, NDOC = b"retrace-key-A", 512, 32, 2
D = json.load(open("/ssd2/ming/basinmark/results/23_floor.json"))
pls = D["pls"]
M = BasinModel(dtype=torch.float32, device="cpu")

docs = []
showcase = []
for i in range(NDOC):
    ids = torch.tensor([D["ids"]["control"][i]])
    pl = pls[i]
    w = ReplayMark(M, KEY, nonce=f"fl-{i}", block_len=BLK, sync_frac=1.0,
                     n_payload_bits=1, s_min=0.5, retries=1)
    w._p_len = pl
    gen_end = pl + GEN
    rows = []
    for b in range(GEN // BLK):
        lo = pl + b * BLK
        B, g, S, qp, qm = w._build_response_bank(ids, lo, gen_end)
        # the base (block-and-suffix-masked) conditional the gate scores against
        base = ids.clone()
        base[0, lo:gen_end] = M.mask_id
        lp = M.logprobs_rows(base, torch.tensor(B), chunk=1, dtype=torch.float64)
        p = torch.softmax(lp[0] / w.temperature, dim=-1)
        ent = (-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).numpy()
        for k, q in enumerate(B):
            rows.append(dict(pos=int(q - pl), block=b, S=float(S[k]),
                             qp=float(qp[k]), qm=float(qm[k]), H=float(ent[k]),
                             tok=M.tok.decode([int(ids[0, q])])))
        if i == 0 and b == 0:
            pass
        if i == 0:
            # keep candidates for the showcase from every block of doc 0
            for k, q in enumerate(B):
                showcase.append(dict(block=b, k=k, pos=int(q - pl), S=float(S[k]),
                                     probs=[float(x) for x in
                                            p[k].topk(12).values.numpy()],
                                     toks=[M.tok.decode([int(t)]) for t in
                                           p[k].topk(12).indices.numpy()],
                                     gsign=[int(np.sign(float(g[k][int(t)])))
                                            for t in p[k].topk(12).indices.numpy()]))
        print(f"doc{i} block {b+1}/{GEN//BLK}", flush=True)
    docs.append(rows)

# showcase: strongest two-sided position and a near-one-sided position from doc 0
sc = sorted(showcase, key=lambda r: -r["S"])
best = sc[0]
sharp = min(showcase, key=lambda r: r["S"] if r["S"] > 0 else 1e9)
onesided = min(showcase, key=lambda r: r["S"])
json.dump(dict(gen=GEN, blk=BLK, ndoc=NDOC, docs=docs,
               showcase=dict(best=best, onesided=onesided)),
          open("/ssd2/ming/basinmark/results/43_viz.json", "w"))
print("saved results/43_viz.json")
