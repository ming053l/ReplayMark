"""R = 2/4/8 against the MATCHED control, n=30, 512 tokens.

The R>=4 'quality cost' on record (x2.6-x4.7) was measured against the wrong reference;
against the matched scheduler R=1/2 turned out free (0.97x/0.93x, replicated held-out at
0.980x). If R=4/8 stay near 1.0x too, the dev rate curve (0.661->0.697 at R=4->8 on 256
tokens) says 512-token TPR@1% lands near 0.8 -- the frontier. Held-out prompts (skip=600),
outputs saved."""
import sys, json, time, subprocess
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, GEN, BLK, NS = b"retrace-key-A", 512, 32, 30
import os; os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
M = BasinModel()
tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                          torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def nll1(t):
    ids = tk(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
    if ids.shape[1] < 8:
        return (float("nan"), int(ids.shape[1]))
    return (float(gm(ids, labels=ids).loss), int(ids.shape[1]))

prompts = c4_prompts(M.tok, NS, skip=600)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
out = {}
for name, kw in (("control", dict(s_min=2.0, retries=1)),
                 ("R2", dict(s_min=0.5, retries=2)),
                 ("R4", dict(s_min=0.5, retries=4)),
                 ("R8", dict(s_min=0.5, retries=8))):
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"r21-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=8000 + i)
        det = ResampleMark(M, KEY, nonce=f"r21-{i}", **BASE, s_min=0.5, retries=1)
        d = det.detect(y, pls[i], GEN, 0)
        txt = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
        rec.append(dict(ids=y[0].tolist(), p=d["p_value"], rate=d["rate_sync"],
                        nll=nll1(txt)))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{NS}  {(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = rec
    ps = np.array([r["p"] for r in rec])
    print(f"{name:<8} | sync {np.mean([r['rate'] for r in rec]):.3f} | "
          f"TPR@5% {np.mean(ps<.05):.2f} @1% {np.mean(ps<.01):.2f} "
          f"@0.1% {np.mean(ps<.001):.2f}", flush=True)

ctl = out["control"]
for name in ("R2", "R4", "R8"):
    ok = [k for k in range(NS) if ctl[k]["nll"][1] >= 100 and out[name][k]["nll"][1] >= 100]
    dn = np.array([out[name][k]["nll"][0] - ctl[k]["nll"][0] for k in ok])
    print(f"{name}: valid {len(ok)} | paired dNLL median {np.median(dn):+.3f} "
          f"ratio {np.exp(dn.mean()):.3f}", flush=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd1/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, gen=GEN, skip=600, pls=pls,
               ids={k: [r["ids"] for r in v] for k, v in out.items()},
               stats={k: [{kk: vv for kk, vv in r.items() if kk != "ids"} for r in v]
                      for k, v in out.items()}),
          open("/ssd1/ming/basinmark/results/21_R_matched.json", "w"))
print(f"saved with git {sha[:8]}")
