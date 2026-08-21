"""Windowed challenge context: buy edit robustness structurally, measure what it costs.

Full-prefix conditioning is why edits propagate: every later block's RRB reads the edited
token. With ctx_window=W the conditional sees only prompt + the last W generated tokens
before the block, so an edit can perturb at most ceil(W/32)+1 blocks' banks. This measures
the whole trade at 512 tokens: clean TPR (does the shorter context still separate?),
attacked TPR under 10% re-denoise (does damage stop propagating?), and paired quality.
Arms share prompts/seeds; the windowed arm uses ctx_window in BOTH generation and
detection (the bank must be the same function on both sides). Fresh prompts (skip=1700,
src_min=700), per-doc nonces w-{i}."""
import sys, json, time, subprocess, os
sys.path.insert(0, "/ssd1/ming/basinmark")
os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark, MASK_ID

KEY, GEN, BLK, NS, W = b"retrace-key-A", 512, 32, 16, 128
M = BasinModel()
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

rng = np.random.default_rng(3400)

@torch.no_grad()
def redenoise(ids, p_len, frac=0.10):
    ids = ids.clone()
    pos = np.sort(rng.choice(np.arange(p_len, p_len + GEN), int(frac * GEN), False))
    x = ids.clone(); x[0, torch.tensor(pos)] = MASK_ID
    lp = M.logprobs_rows(x, torch.tensor(pos), chunk=2)
    ids[0, torch.tensor(pos)] = lp[0].argmax(-1).cpu()
    return ids

prompts = c4_prompts(M.tok, NS, skip=1700, src_min=700)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
ARMS = [("control", dict(s_min=2.0, retries=1), None),
        ("full", dict(s_min=0.5, retries=16, p_floor=0.05), None),
        ("win128", dict(s_min=0.5, retries=16, p_floor=0.05), W)]
out = {}
for name, kw, win in ARMS:
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"w-{i}", **BASE, ctx_window=win, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=34000 + i)
        det = ResampleMark(M, KEY, nonce=f"w-{i}", **BASE, ctx_window=win,
                           s_min=0.5, retries=1)
        d0 = det.detect(y, pls[i], GEN, 0)
        da = det.detect(redenoise(y, pls[i]), pls[i], GEN, 0)
        rec.append(dict(ids=y[0].tolist(), p=d0["p_value"], p_att=da["p_value"],
                        rate=d0["rate_sync"],
                        nll=nll1(M.tok.decode(y[0, pls[i]:pls[i] + GEN],
                                              skip_special_tokens=True))))
        if (i + 1) % 8 == 0:
            print(f"  [{name}] {i+1}/{NS}  {(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = rec
    ps = np.array([r["p"] for r in rec]); pa = np.array([r["p_att"] for r in rec])
    print(f"{name:<8} | sync {np.mean([r['rate'] for r in rec]):.3f} | "
          f"clean TPR@1% {np.mean(ps<.01):.2f} | rd10 TPR@1% {np.mean(pa<.01):.2f}",
          flush=True)
ctl = out["control"]
for name in ("full", "win128"):
    ok = [k for k in range(NS) if ctl[k]["nll"][1] >= 150 and out[name][k]["nll"][1] >= 150]
    dn = np.array([out[name][k]["nll"][0] - ctl[k]["nll"][0] for k in ok])
    print(f"{name}: valid {len(ok)} | ratio {np.exp(dn.mean()):.3f}", flush=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd1/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, gen=GEN, window=W, skip=1700, pls=pls,
               ids={k: [r["ids"] for r in v] for k, v in out.items()},
               stats={k: [{kk: vv for kk, vv in r.items() if kk != "ids"} for r in v]
                      for k, v in out.items()}),
          open("/ssd1/ming/basinmark/results/34_window.json", "w"))
print(f"saved with git {sha[:8]}")
