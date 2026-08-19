"""Carrier budget vs quality: cap conditioned positions per block."""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, MESSAGE, GEN, BLK, NS = b"retrace-key-A", 0xA5, 256, 32, 12
GRID = [(4, 8), (8, 8), (8, 16), (12, 8)]      # (max_carriers/block, R)

import os; os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
M = BasinModel()
tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                          torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def nll(ts):
    o = []
    for t in ts:
        ids = tk(t, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        o.append(float(gm(ids, labels=ids).loss) if ids.shape[1] >= 8 else np.nan)
    return np.array(o)

prompts = c4_prompts(M.tok, NS)
pls = [p.shape[1] for p in prompts]
ref = [M.generate(p, gen_len=GEN, steps=GEN, block_len=BLK, temperature=0.8,
                  seed=3000 + i).cpu() for i, p in enumerate(prompts)]
rtxt = [M.tok.decode(x[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
        for i, x in enumerate(ref)]
ppl0 = float(np.exp(np.nanmean(nll(rtxt))))
rows = []
for mc, R in GRID:
    nul = [ResampleMark(M, KEY, block_len=BLK, s_min=0.5, max_carriers=mc, retries=R,
                        nonce=f"doc-{i}").detect(x, pls[i], GEN, MESSAGE)
           for i, x in enumerate(ref)]
    nr = float(np.mean([d["rate_sync"] for d in nul]))
    nn = float(np.mean([d["n_sync"] for d in nul]))
    ps, rates, txt, t0 = [], [], [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, max_carriers=mc, retries=R,
                         nonce=f"doc-{i}")
        y = w.generate(p, gen_len=GEN, steps=GEN, message=MESSAGE, seed=3000 + i)
        d = w.detect(y, pls[i], GEN, MESSAGE)
        ps.append(d["p_value"]); rates.append(d["rate_sync"])
        txt.append(M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True))
    pw = float(np.exp(np.nanmean(nll(txt)))); ps = np.array(ps)
    r = dict(mc=mc, R=R, rate=float(np.mean(rates)), rate_ref=nr, n_sync=nn,
             tpr01=float(np.mean(ps < 0.01)), tpr05=float(np.mean(ps < 0.05)),
             ppl=pw, ratio=pw / ppl0)
    rows.append(r)
    print(f"mc={mc:<3} R={R:<3} | sync {r['rate']:.3f} (ref {nr:.3f}, n {nn:.0f}) | "
          f"TPR@5% {r['tpr05']:.2f} @1% {r['tpr01']:.2f} | ppl {pw:.2f} "
          f"(x{r['ratio']:.2f}) | {(time.time()-t0)/NS:.0f}s/doc", flush=True)
    json.dump(dict(rows=rows, ppl_ref=ppl0),
              open("/ssd1/ming/basinmark/results/budget.json", "w"), indent=1)
print(f"ref ppl {ppl0:.2f}; targets: TPR@1% >= 0.8 at ppl <= x1.2-1.3")
