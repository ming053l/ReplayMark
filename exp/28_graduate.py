"""The combination: plausibility floor x document length.

R8@k=0.1 measured TPR@1% 0.70 at ratio 0.871 with repetition parity on 512 tokens;
n_sync ~600 at 1024 tokens projects z~4.2, i.e. TPR@1% ~0.85 at sub-1.0x quality --
both axes past dgMARK if it holds. Repetition is computed on decoded text with the
shared valid rule, since raw-id bigrams count EOS padding as repetition.

Acceptance now also requires p(v) >= kappa * p_max, capping each carrier's NLL cost at
-log(kappa) beyond the model's own argmax -- an embedder-only, key-free constraint the
detector never sees. If the floored R=8/16 arms hold ratio <= ~1.25 while the rate stays
near the unfloored curve (0.674-0.711), TPR@1% lands at 0.8+ at dgMARK's quality point.
Same protocol as exp/21; fresh offsets (skip=1250); outputs saved."""
import sys, json, time, subprocess
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, GEN, BLK, NS = b"retrace-key-A", 1024, 32, 16
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

prompts = c4_prompts(M.tok, NS, skip=1250)
pls = [p.shape[1] for p in prompts]
BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
# two arms only: 1024-token docs run ~8 min each on this GPU, and R16 bought little
# rate over R8 at the same floor (0.574 vs 0.568 at kappa=0.3)
ARMS = [("control", dict(s_min=2.0, retries=1)),
        ("R16k05", dict(retries=16, p_floor=0.05, s_min=0.5))]
out = {}
for name, kw in ARMS:
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"gr-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=12000 + i)
        d = ResampleMark(M, KEY, nonce=f"gr-{i}", **BASE, s_min=0.5,
                         retries=1).detect(y, pls[i], GEN, 0)
        rec.append(dict(ids=y[0].tolist(), p=d["p_value"], rate=d["rate_sync"],
                        nll=nll1(M.tok.decode(y[0, pls[i]:pls[i] + GEN],
                                              skip_special_tokens=True))))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{NS}  {(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = rec
    ps = np.array([r["p"] for r in rec])
    print(f"{name:<8} | sync {np.mean([r['rate'] for r in rec]):.3f} | "
          f"TPR@5% {np.mean(ps<.05):.2f} @1% {np.mean(ps<.01):.2f} "
          f"@0.1% {np.mean(ps<.001):.2f}", flush=True)
ctl = out["control"]
def rep_of(rec, i):
    txt = M.tok.decode(torch.tensor(rec["ids"])[pls[i]:pls[i] + GEN],
                       skip_special_tokens=True)
    ids = M.tok(txt)["input_ids"]
    if len(ids) < 50:
        return None
    big = [tuple(ids[j:j+2]) for j in range(len(ids)-1)]
    return 1 - len(set(big)) / len(big)

for name, _ in ARMS[1:]:
    ok = [k for k in range(NS) if ctl[k]["nll"][1] >= 150 and out[name][k]["nll"][1] >= 150]
    dn = np.array([out[name][k]["nll"][0] - ctl[k]["nll"][0] for k in ok])
    rc = [rep_of(ctl[k], k) for k in ok]; rw = [rep_of(out[name][k], k) for k in ok]
    rc = [x for x in rc if x is not None]; rw = [x for x in rw if x is not None]
    print(f"{name}: valid {len(ok)} | dNLL median {np.median(dn):+.3f} "
          f"ratio {np.exp(dn.mean()):.3f} | repeat {np.mean(rw):.3f} "
          f"(ctl {np.mean(rc):.3f})", flush=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd1/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, gen=GEN, skip=1250, pls=pls,
               ids={k: [r["ids"] for r in v] for k, v in out.items()},
               stats={k: [{kk: vv for kk, vv in r.items() if kk != "ids"} for r in v]
                      for k, v in out.items()}),
          open("/ssd1/ming/basinmark/results/28_graduate.json", "w"))
print(f"saved with git {sha[:8]}")
