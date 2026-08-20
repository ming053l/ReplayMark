"""Length as the free axis: R=2 at 1024 tokens. n_sync doubles again; if the rate holds
near 0.62, per-doc z ~ 4.1 -> TPR@1% ~0.85 at the R=2 quality point (~1.14x)."""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, GEN, BLK, NS = b"retrace-key-A", 1024, 32, 20
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

prompts = c4_prompts(M.tok, NS, skip=800)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
out = {}
for name, kw in (("control", dict(s_min=2.0, retries=1)),
                 ("R2", dict(s_min=0.5, retries=2))):
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"L-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=9500 + i)
        d = ResampleMark(M, KEY, nonce=f"L-{i}", **BASE, s_min=0.5,
                         retries=1).detect(y, pls[i], GEN, 0)
        rec.append(dict(p=d["p_value"], rate=d["rate_sync"], n=d["n_sync"],
                        nll=nll1(M.tok.decode(y[0, pls[i]:pls[i] + GEN],
                                              skip_special_tokens=True))))
        if (i + 1) % 5 == 0:
            print(f"  [{name}] {i+1}/{NS}  {(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = rec
    ps = np.array([r["p"] for r in rec])
    print(f"{name:<8} | sync {np.mean([r['rate'] for r in rec]):.3f} "
          f"(n {np.mean([r['n'] for r in rec]):.0f}) | TPR@5% {np.mean(ps<.05):.2f} "
          f"@1% {np.mean(ps<.01):.2f} @0.1% {np.mean(ps<.001):.2f}", flush=True)
ctl = out["control"]
ok = [k for k in range(NS) if ctl[k]["nll"][1] >= 150 and out["R2"][k]["nll"][1] >= 150]
dn = np.array([out["R2"][k]["nll"][0] - ctl[k]["nll"][0] for k in ok])
print(f"R2@1024: valid {len(ok)} | dNLL median {np.median(dn):+.3f} "
      f"ratio {np.exp(dn.mean()):.3f}")
json.dump({k: [{kk: vv for kk, vv in r.items()} for r in v] for k, v in out.items()},
          open("/ssd1/ming/basinmark/results/24_len1024.json", "w"))
