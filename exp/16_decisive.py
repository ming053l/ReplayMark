"""The freeze-or-pivot run: n=30 valid-scale, GEN=512, presence-only, three arms.

The quality attribution has been confounded: every dNLL so far compared ResampleMark's
generate against M.generate, which differ in scheduler and sampling path independent of
the watermark -- and R=1 vs R=2 showing the same +1.0 dNLL is the signature of that
confound, not of steering cost. Arm A (s_min=2 -> zero carriers) is the same generator
with steering off; it is both the quality baseline and the null arm. Whatever dNLL
remains between A and R=1/R=2 is attributable to the watermark and nothing else.
"""
import sys, time
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

prompts = c4_prompts(M.tok, NS)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
ARMS = [("control", dict(s_min=2.0, retries=1)),       # same generator, steering off
        ("R1", dict(s_min=0.5, retries=1)),
        ("R2", dict(s_min=0.5, retries=2))]
out = {}
for name, kw in ARMS:
    txts, ps, rates, t0 = [], [], [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"doc-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=3000 + i)
        # detection always uses the deployed carrier rule, whatever the embedder did
        d = ResampleMark(M, KEY, nonce=f"doc-{i}", **BASE, s_min=0.5,
                         retries=1).detect(y, pls[i], GEN, 0)
        ps.append(d["p_value"]); rates.append(d["rate_sync"])
        txts.append(M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{NS}  {(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = dict(nll=[nll1(t) for t in txts], p=np.array(ps), rate=np.array(rates))
    r = out[name]
    print(f"{name:<8} | sync {r['rate'].mean():.3f} | TPR@5% {np.mean(r['p']<.05):.2f} "
          f"@1% {np.mean(r['p']<.01):.2f} @0.1% {np.mean(r['p']<.001):.2f}", flush=True)

ctl = out["control"]
ok = [k for k in range(NS) if ctl["nll"][k][1] >= 100
      and all(out[n]["nll"][k][1] >= 100 for n, _ in ARMS[1:])]
print(f"\nvalid (>=100 gpt2 tokens in every arm): {len(ok)}/{NS}")
for name, _ in ARMS[1:]:
    dn = np.array([out[name]["nll"][k][0] - ctl["nll"][k][0] for k in ok])
    print(f"{name}: dNLL vs matched control  median {np.median(dn):+.3f}  "
          f"q25 {np.quantile(dn,.25):+.3f} q75 {np.quantile(dn,.75):+.3f}  "
          f"ratio(exp-mean) {np.exp(dn.mean()):.2f}")
print(f"control absolute ppl {np.exp(np.nanmean([v[0] for v in ctl['nll']])):.2f}   "
      f"null check: control TPR@1% should be ~0.01: {np.mean(ctl['p']<.01):.2f}")
