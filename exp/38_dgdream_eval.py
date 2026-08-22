"""Length-matched dgMARK evaluation at 512 tokens on Dream-7B-Instruct."""
import sys, csv, glob, json
sys.path.insert(0, "/ssd2/ming/basinmark")
import numpy as np, torch
OUT = "/ssd2/ming/basinmark/results/baselines"
import os; os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                          torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def nll(ts):
    o = []
    for t in ts:
        ids = tk(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.cuda()
        if ids.shape[1] >= 8:
            o.append(float(gm(ids, labels=ids).loss))
    return np.array(o)

def load(tag):
    f = sorted(glob.glob(f"{OUT}/dgdream_{tag}*.csv"))[0]
    return list(csv.DictReader(open(f, newline="", encoding="utf-8")))

og = load("original")
n_o = nll([r["generated"] for r in og])
p0 = float(np.exp(np.nanmean(n_o)))
ro = np.array([float(r["match_ratio"]) for r in og])
no = np.array([int(r["trimmed_length"]) for r in og])
zo = (ro - 0.5) * np.sqrt(no) / 0.5
print(f"original n={len(og)} ppl {p0:.2f} z {zo.mean():+.2f} (sd {zo.std():.2f})")
for tag in ("watermark", "beam3"):
    try:
        rows = load(tag)
    except IndexError:
        print(f"{tag}: missing"); continue
    r = np.array([float(x["match_ratio"]) for x in rows])
    n = np.array([int(x["trimmed_length"]) for x in rows])
    z = (r - 0.5) * np.sqrt(n) / 0.5
    pw = float(np.exp(np.nanmean(nll([x["generated"] for x in rows]))))
    print(f"{tag:<10} n={len(rows)} | TPR@5% {np.mean(z>1.645):.2f} "
          f"@1% {np.mean(z>2.326):.2f} @0.1% {np.mean(z>3.090):.2f} | "
          f"ppl {pw:.2f} (x{pw/p0:.3f})")
