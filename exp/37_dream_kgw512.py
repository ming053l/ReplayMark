"""KGW at 512 tokens on Dream-7B-Instruct (n=30, deltas 0/1), vocab 152064.

Reported on the same axes as BasinMark and the local dgMARK run: TPR at an analytic FPR
threshold, GPT-2-large perplexity ratio against its OWN no-watermark control (left-to-right
decoding, same sampler), and detection cost.
"""
import sys, gzip, json, time
sys.path.insert(0, "/ssd2/ming/basinmark")
import numpy as np, torch
from basinmark.dream_model import DreamModel
from basinmark.kgw import kgw_generate, kgw_detect

GEN, NS = 512, 30
DELTAS = [0.0, 1.0]
KEY = b"kgw-key"


def c4_prompts(tok, n, max_chars=300):
    """Same construction dgMARK uses: C4 documents truncated to 300 characters."""
    out = []
    with gzip.open("/ssd2/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            t = json.loads(line)["text"]
            w, cur = t.split(), ""
            for x in w:
                if len(cur) + len(x) + 1 > max_chars:
                    break
                cur = x if not cur else cur + " " + x
            ids = tok(cur)["input_ids"]
            if len(ids) >= 20:
                out.append(torch.tensor(ids, dtype=torch.long)[None])
                if len(out) == n:
                    return out


class Nll:
    def __init__(self):
        import os
        os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
        self.m = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2-large", torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def __call__(self, texts):
        o = []
        for t in texts:
            ids = self.tk(t, return_tensors="pt", truncation=True,
                          max_length=1024).input_ids.cuda()
            o.append(float(self.m(ids, labels=ids).loss) if ids.shape[1] >= 8 else np.nan)
        return np.array(o)


def main():
    M = DreamModel()
    nll = Nll()
    prompts = c4_prompts(M.tok, NS)
    print(f"[prompts] {len(prompts)} C4 documents, 300-char truncation", flush=True)

    rows, ppl0 = [], None
    for d in DELTAS:
        zs, zr, dup, txt, t0 = [], [], [], [], time.time()
        for i, p in enumerate(prompts):
            span = np.arange(p.shape[1], p.shape[1] + GEN)
            y = kgw_generate(M, p, gen_len=GEN, delta=d, key=KEY, temperature=0.8,
                             seed=5000 + i)
            dd = kgw_detect(M, y, span, key=KEY, dedup=True, vocab=152064)
            zs.append(dd["z"]); dup.append(dd["dup_frac"])
            zr.append(kgw_detect(M, y, span, key=KEY, dedup=False, vocab=152064)["z"])
            txt.append(M.tok.decode(y[0, span], skip_special_tokens=True))
        n = nll(txt); ppl = float(np.exp(np.nanmean(n)))
        if d == 0.0:
            ppl0 = ppl
        zs = np.array(zs)
        r = dict(delta=d, z=float(zs.mean()), z_nodedup=float(np.mean(zr)),
                 dup_frac=float(np.mean(dup)), ppl=ppl, ratio=ppl / ppl0,
                 tpr05=float(np.mean(zs > 1.645)), tpr01=float(np.mean(zs > 2.326)),
                 tpr001=float(np.mean(zs > 3.090)), z_all=zs.tolist(),
                 text0=txt[0][:300])
        rows.append(r)
        print(f"delta={d:<4} | z {r['z']:+.2f} (no-dedup {r['z_nodedup']:+.2f}) | "
              f"TPR@5% {r['tpr05']:.2f} @1% {r['tpr01']:.2f} @0.1% {r['tpr001']:.2f} | "
              f"ppl {ppl:.2f} (x{r['ratio']:.2f}) | dup bigrams {r['dup_frac']:.2f} | "
              f"{(time.time()-t0)/len(prompts):.0f}s/sample", flush=True)
        json.dump(rows, open("/ssd2/ming/basinmark/results/kgw512_dream.json", "w"), indent=1)

    print("\n===== KGW on LLaDA (left-to-right), same pipeline =====")
    print(f"{'delta':<8}{'ppl ratio':>11}{'TPR@1%':>9}{'z':>8}")
    for r in rows:
        print(f"{r['delta']:<8}{r['ratio']:>11.2f}{r['tpr01']:>9.2f}{r['z']:>+8.2f}")
    print("detection cost: 0 model forwards (green-list hashing only)")


if __name__ == "__main__":
    main()
