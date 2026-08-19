"""Confirmatory held-out run. FRESH prompts (skip=500: development burned the first
hundreds), n=40, per-document nonces, matched-scheduler control, outputs + git SHA saved.
Detector: whichever exp/17-18 favoured (LLR expected); both p-values recorded per doc so
the choice is auditable. Quality: paired dNLL against the matched control, plus valid
rate under one shared >=100-token rule. Protocol otherwise identical to the baselines:
same checkpoint, C4 300-char prompts, steps = gen_length, GPT-2-large.
"""
import sys, json, time, subprocess
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark
from basinmark.llr import build_cache, llr_pvalue

KEY, GEN, BLK, NS = b"retrace-key-A", 512, 32, 40
SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True, cwd="/ssd1/ming/basinmark").stdout.strip()
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

prompts = c4_prompts(M.tok, NS, skip=500)          # held out from all development
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
ARMS = [("control", dict(s_min=2.0, retries=1)), ("R1", dict(s_min=0.5, retries=1))]
out = {}
for name, kw in ARMS:
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"ho-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=7000 + i)
        det = ResampleMark(M, KEY, nonce=f"ho-{i}", **BASE, s_min=0.5, retries=1)
        cache = build_cache(det, y, pls[i], GEN)
        p_llr = llr_pvalue(cache, det.key, pls[i], GEN, R=1, n_mc=100_000)["p_value"]
        p_ind = det.detect(y, pls[i], GEN, 0)["p_value"]
        txt = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
        rec.append(dict(ids=y[0].tolist(), p_llr=p_llr, p_ind=p_ind, nll=nll1(txt)))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{NS}  {(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = rec
    pl_ = np.array([r["p_llr"] for r in rec]); pi_ = np.array([r["p_ind"] for r in rec])
    print(f"{name:<8} | LLR TPR@5% {np.mean(pl_<.05):.2f} @1% {np.mean(pl_<.01):.2f} "
          f"@0.1% {np.mean(pl_<.001):.2f} | indicator @1% {np.mean(pi_<.01):.2f}", flush=True)

ok = [k for k in range(NS) if out["control"][k]["nll"][1] >= 100
      and out["R1"][k]["nll"][1] >= 100]
dn = np.array([out["R1"][k]["nll"][0] - out["control"][k]["nll"][0] for k in ok])
print(f"\nvalid {len(ok)}/{NS} | paired dNLL vs matched control: median {np.median(dn):+.3f} "
      f"q25 {np.quantile(dn,.25):+.3f} q75 {np.quantile(dn,.75):+.3f} "
      f"ratio {np.exp(dn.mean()):.3f}")
print(f"control ppl {np.exp(np.nanmean([r['nll'][0] for r in out['control']])):.2f}")
json.dump(dict(sha=SHA, gen=GEN, ns=NS, skip=500,
               out={k: [{kk: vv for kk, vv in r.items() if kk != 'ids'} for r in v]
                    for k, v in out.items()},
               ids={k: [r["ids"] for r in v] for k, v in out.items()}, pls=pls),
          open("/ssd1/ming/basinmark/results/19_freeze.json", "w"))
print(f"saved with git {SHA[:8]}")
