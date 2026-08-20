"""Fallback levers if the 1024 combo falls short: recover rate at bounded cost.
Arms at 512 tokens for speed; the winner graduates to 1024. Fresh offsets."""
import sys, json, time, subprocess
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, GEN, BLK, NS = b"retrace-key-A", 512, 32, 20
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

prompts = c4_prompts(M.tok, NS, skip=950)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
for name, kw in (("control", dict(s_min=2.0, retries=1)),
                 ("R16k05", dict(s_min=0.5, retries=16, p_floor=0.05)),
                 ("R8k10w", dict(s_min=0.4, retries=8, p_floor=0.10))):
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"fb-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=10500 + i)
        det = ResampleMark(M, KEY, nonce=f"fb-{i}", **BASE,
                           s_min=(0.4 if name == "R8k10w" else 0.5), retries=1)
        d = det.detect(y, pls[i], GEN, 0)
        rec.append((d["p_value"], d["rate_sync"],
                    nll1(M.tok.decode(y[0, pls[i]:pls[i] + GEN],
                                      skip_special_tokens=True))))
    ps = np.array([r[0] for r in rec])
    if name == "control":
        ctl = rec
    else:
        ok = [k for k in range(NS) if ctl[k][2][1] >= 100 and rec[k][2][1] >= 100]
        dn = np.array([rec[k][2][0] - ctl[k][2][0] for k in ok])
        print(f"{name:<8} | sync {np.mean([r[1] for r in rec]):.3f} | "
              f"TPR@5% {np.mean(ps<.05):.2f} @1% {np.mean(ps<.01):.2f} "
              f"@0.1% {np.mean(ps<.001):.2f} | ratio {np.exp(dn.mean()):.3f} "
              f"(n={len(ok)})", flush=True)
        continue
    print(f"{name:<8} | sync {np.mean([r[1] for r in rec]):.3f} | "
          f"TPR@1% {np.mean(ps<.01):.2f}", flush=True)
