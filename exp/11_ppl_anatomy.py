"""Where does the R>=2 perplexity damage live: a few degenerate docs, or uniform?"""
import sys
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, MESSAGE, GEN, BLK, NS = b"retrace-key-A", 0xA5, 256, 32, 12
import os; os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
M = BasinModel()
tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                          torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def nll1(t):
    ids = tk(t, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
    return float(gm(ids, labels=ids).loss) if ids.shape[1] >= 8 else float("nan")

prompts = c4_prompts(M.tok, NS)
pls = [p.shape[1] for p in prompts]
rows = []
worst = (None, -1)
for i, p in enumerate(prompts):
    ref = M.generate(p, gen_len=GEN, steps=GEN, block_len=BLK, temperature=0.8,
                     seed=3000 + i).cpu()
    w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, retries=2, nonce=f"doc-{i}")
    y = w.generate(p, gen_len=GEN, steps=GEN, message=MESSAGE, seed=3000 + i)
    tr = M.tok.decode(ref[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
    tw = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
    a, b = nll1(tr), nll1(tw)
    rows.append((a, b))
    print(f"[{i:02d}] ref nll {a:.3f}  wm nll {b:.3f}  delta {b-a:+.3f}", flush=True)
    if b - a > worst[1]:
        worst = (tw, b - a, tr)
a = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows])
d = b - a
print(f"\nper-doc delta-NLL: mean {d.mean():+.3f}  median {np.median(d):+.3f}  "
      f"q25 {np.quantile(d,.25):+.3f}  q75 {np.quantile(d,.75):+.3f}  max {d.max():+.3f}")
print(f"exp(mean) ratio {np.exp(b.mean()-a.mean()):.2f}   "
      f"median-of-doc-ratios {np.median(np.exp(d)):.2f}")
print(f"\n--- worst doc (delta {worst[1]:+.3f}), watermarked text ---\n{worst[0][:500]}")
print(f"\n--- same doc, reference ---\n{worst[2][:500]}")
