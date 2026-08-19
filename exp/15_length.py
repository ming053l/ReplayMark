"""The last quality-free lever: document length. n_sync grows linearly with length, z with
its square root; dgMARK reports the same curve (their Fig. 4). R stays at 1-2 where the
perplexity was x1.16."""
import sys, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, BLK, NS = b"retrace-key-A", 32, 10
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

prompts = c4_prompts(M.tok, NS)
pls = [p.shape[1] for p in prompts]
CFG = dict(block_len=BLK, s_min=0.5, sync_frac=1.0, n_payload_bits=1)
for GEN in (512,):
    ref = [M.generate(p, gen_len=GEN, steps=GEN, block_len=BLK, temperature=0.8,
                      seed=3000 + i).cpu() for i, p in enumerate(prompts)]
    nr = [nll1(M.tok.decode(x[0, pls[i]:pls[i] + GEN], skip_special_tokens=True))
          for i, x in enumerate(ref)]
    nul = [ResampleMark(M, KEY, retries=1, nonce=f"doc-{i}", **CFG).detect(
        x, pls[i], GEN, 0) for i, x in enumerate(ref)]
    print(f"[GEN={GEN} null] sync {np.mean([d['rate_sync'] for d in nul]):.3f} "
          f"n {np.mean([d['n_sync'] for d in nul]):.0f}", flush=True)
    for R in (1, 2):
        ps, rates, res, t0 = [], [], [], time.time()
        for i, p in enumerate(prompts):
            w = ResampleMark(M, KEY, retries=R, nonce=f"doc-{i}", **CFG)
            y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=3000 + i)
            d = w.detect(y, pls[i], GEN, 0)
            ps.append(d["p_value"]); rates.append(d["rate_sync"])
            res.append(nll1(M.tok.decode(y[0, pls[i]:pls[i] + GEN],
                                         skip_special_tokens=True)))
        ps = np.array(ps)
        ok = [k for k in range(NS) if nr[k][1] >= 50 and res[k][1] >= 50]
        dn = np.array([res[k][0] - nr[k][0] for k in ok])
        print(f"GEN={GEN} R={R} | sync {np.mean(rates):.3f} | TPR@5% {np.mean(ps<.05):.2f} "
              f"@1% {np.mean(ps<.01):.2f} @0.1% {np.mean(ps<.001):.2f} | valid {len(ok)}/{NS} "
              f"| dNLL med {np.median(dn):+.3f} ratio {np.exp(dn.mean()):.2f} | "
              f"{(time.time()-t0)/NS:.0f}s/doc", flush=True)
