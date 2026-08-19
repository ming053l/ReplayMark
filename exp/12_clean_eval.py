"""Quality, with the two artifacts removed: empty generations counted not dropped, and
degenerate-repetitive references flagged since GPT-2 rewards the repetition itself."""
import sys
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, MESSAGE, GEN, BLK, NS = b"retrace-key-A", 0xA5, 256, 32, 20
import os; os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
M = BasinModel()
tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                          torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def nll1(t):
    ids = tk(t, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
    return (float(gm(ids, labels=ids).loss), int(ids.shape[1]))

def rep2(t):
    ids = M.tok(t)["input_ids"]
    big = [tuple(ids[i:i + 2]) for i in range(len(ids) - 1)]
    return 1 - len(set(big)) / max(len(big), 1)

prompts = c4_prompts(M.tok, NS)
pls = [p.shape[1] for p in prompts]
res = []
for i, p in enumerate(prompts):
    ref = M.generate(p, gen_len=GEN, steps=GEN, block_len=BLK, temperature=0.8,
                     seed=3000 + i).cpu()
    w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, retries=2, nonce=f"doc-{i}")
    y = w.generate(p, gen_len=GEN, steps=GEN, message=MESSAGE, seed=3000 + i)
    d = w.detect(y, pls[i], GEN, MESSAGE)
    tr = M.tok.decode(ref[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
    tw = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
    (nr, lr), (nw_, lw) = nll1(tr), nll1(tw)
    res.append(dict(i=i, nr=nr, nw=nw_, lr=lr, lw=lw, rr=rep2(tr), rw=rep2(tw),
                    p=d["p_value"], rate=d["rate_sync"]))
    print(f"[{i:02d}] len {lr}/{lw}  nll {nr:.2f}/{nw_:.2f}  rep {res[-1]['rr']:.2f}/"
          f"{res[-1]['rw']:.2f}  sync {d['rate_sync']:.2f}  p {d['p_value']:.3f}", flush=True)

ok = [r for r in res if r["lr"] >= 50 and r["lw"] >= 50]
dg = [r for r in ok if r["rr"] > 0.4]                      # degenerate-repetitive refs
cl = [r for r in ok if r["rr"] <= 0.4]
print(f"\nvalid {len(ok)}/{NS}  (empty/short: {NS - len(ok)});  "
      f"degenerate-repetitive refs among valid: {len(dg)}")
for name, grp in (("all valid", ok), ("clean refs only", cl)):
    if not grp:
        continue
    d = np.array([r["nw"] - r["nr"] for r in grp])
    print(f"{name:<17} n={len(grp)}  dNLL median {np.median(d):+.3f} "
          f"(q25 {np.quantile(d,.25):+.3f} q75 {np.quantile(d,.75):+.3f})  "
          f"median doc-ratio {np.median(np.exp(d)):.2f}")
tp = np.array([r["p"] for r in ok])
print(f"detection on valid docs: TPR@5% {np.mean(tp<.05):.2f}  @1% {np.mean(tp<.01):.2f}  "
      f"mean sync {np.mean([r['rate'] for r in ok]):.3f}")
