"""Dream-7B-Instruct detectability block for Table 1 (512 tokens, locked length).

Mirrors the LLaDA rows: control (matched sampler, zero carriers), R=1, R=8/kappa=0.1,
R=16/kappa=0.05, all at 512 tokens / 32-token blocks / one step per token, C4
continuation prompts, temperature 0.8. n=30 per arm except R16k05 (n=20), matching the
LLaDA sample sizes the caption states. PPL is GPT-2-large paired with the control arm.

PREFLIGHT: one 64-token document is generated and detected before the arms start, so a
broken Dream port (wrong mask id, missing logits shift) dies in minutes, not hours.
"""
import sys, json, time, subprocess, os
sys.path.insert(0, "/ssd2/ming/basinmark")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.dream_model import DreamModel
from basinmark.data import c4_prompts
from basinmark.resample import ReplayMark

KEY, GEN, BLK, NS = b"retrace-key-A", 512, 32, 30
M = DreamModel()
from transformers import AutoModelForCausalLM, AutoTokenizer
tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                          torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def nll1(t):
    ids = tk(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
    if ids.shape[1] < 8:
        return (float("nan"), int(ids.shape[1]))
    return (float(gm(ids, labels=ids).loss), int(ids.shape[1]))

prompts = c4_prompts(M.tok, NS, skip=1500)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)

# ---- preflight ----
w = ReplayMark(M, KEY, nonce="pre", **BASE, s_min=0.5, retries=8, p_floor=0.10)
y = w.generate(prompts[0], gen_len=64, steps=64, message=0, seed=1)
d = ReplayMark(M, KEY, nonce="pre", **BASE, s_min=0.5, retries=1).detect(
    y, pls[0], 64, 0)
txt = M.tok.decode(y[0, pls[0]:pls[0] + 64], skip_special_tokens=True)
print(f"[preflight] carriers {w.stats['carrier']} accepted {w.stats['accepted']} "
      f"sync {d['rate_sync']:.3f} p {d['p_value']:.3f}", flush=True)
print(f"[preflight] text: {txt[:200]!r}", flush=True)
assert w.stats["carrier"] > 0, "no carriers admitted -- Dream port is broken"
assert d["n"] > 0, "detector found no carriers on replay"

ARMS = [("control", dict(s_min=2.0, retries=1), NS),
        ("R1",      dict(s_min=0.5, retries=1), NS),
        ("R8k10",   dict(s_min=0.5, retries=8,  p_floor=0.10), NS),
        ("R16k05",  dict(s_min=0.5, retries=16, p_floor=0.05), 20)]
out = {}
for name, kw, n_arm in ARMS:
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts[:n_arm]):
        w = ReplayMark(M, KEY, nonce=f"dr-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=36000 + i)
        d = ReplayMark(M, KEY, nonce=f"dr-{i}", **BASE, s_min=0.5,
                         retries=1).detect(y, pls[i], GEN, 0)
        rec.append(dict(ids=y[0].tolist(), p=d["p_value"], rate=d["rate_sync"],
                        nll=nll1(M.tok.decode(y[0, pls[i]:pls[i] + GEN],
                                              skip_special_tokens=True))))
        if (i + 1) % 5 == 0:
            print(f"  [{name}] {i+1}/{n_arm}  {(time.time()-t0)/(i+1):.0f}s/doc",
                  flush=True)
    out[name] = rec
    ps = np.array([r["p"] for r in rec])
    print(f"{name:<8} | sync {np.mean([r['rate'] for r in rec]):.3f} | "
          f"TPR@5% {np.mean(ps<.05):.2f} @1% {np.mean(ps<.01):.2f} "
          f"@0.1% {np.mean(ps<.001):.2f}", flush=True)

ctl = out["control"]
for name, _, n_arm in ARMS[1:]:
    ok = [k for k in range(n_arm) if ctl[k]["nll"][1] >= 100 and out[name][k]["nll"][1] >= 100]
    dn = np.array([out[name][k]["nll"][0] - ctl[k]["nll"][0] for k in ok])
    print(f"{name}: valid {len(ok)} | dNLL median {np.median(dn):+.3f} "
          f"ratio {np.exp(dn.mean()):.3f}", flush=True)

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, model="Dream-v0-Instruct-7B", gen=GEN, skip=1500, pls=pls,
               ids={k: [r["ids"] for r in v] for k, v in out.items()},
               stats={k: [{kk: vv for kk, vv in r.items() if kk != "ids"} for r in v]
                      for k, v in out.items()}),
          open("/ssd2/ming/basinmark/results/36_dream.json", "w"))
print(f"saved with git {sha[:8]}")
