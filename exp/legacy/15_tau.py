"""Quality is the binding constraint, not detection. Sweep the per-token budget tau.

exp/11 measured GPT-2-large perplexity for the first time and it invalidates the
operating point chosen so far: draft 22.2, gate=0.6 lam=20 -> 107.3. tau=6 nats admits a
token the denoiser rates ~0.25% as likely as its best, and at 15% of positions that
compounds. Detection was tuned with quality unmeasured; this sweep does it the right way
round -- fix lam high so tau binds, and find whether ANY setting reaches usable
detection at an acceptable perplexity cost.
"""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 12, 16
GRID = list(itertools.product([1.0, 2.0, 3.0, 6.0], [None, 0.60]))
LAM = 20.0


def c4(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


class Ppl:
    def __init__(self):
        import os
        os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
        self.m = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2-large", torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def __call__(self, texts):
        o = []
        for t in texts:
            ids = self.tk(t, return_tensors="pt", truncation=True,
                          max_length=512).input_ids.cuda()
            o.append(float(torch.exp(self.m(ids, labels=ids).loss))
                     if ids.shape[1] >= 8 else float("nan"))
        return np.array(o)


def main():
    M = BasinModel()
    ppl = Ppl()
    drafts = []
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        drafts.append((x, np.arange(p.shape[1], p.shape[1] + GEN)))
    d0 = np.nanmedian(ppl([M.tok.decode(x[0, s], skip_special_tokens=True)
                           for x, s in drafts]))
    print(f"[drafts] n={NS} ppl {d0:.1f}  (reference sampler)", flush=True)

    rows = []
    for tau, pool in GRID:
        cfg = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=tau, lam=LAM,
                   commit_steps=2, n_patterns=8, n_ablations=6, pool_rate=pool)
        wm = SharedMark(M, KEY, **cfg)
        zs, pb, acc, ch, txt = [], [], [], [], []
        for x, span in drafts:
            y = wm.embed(x, span, MESSAGE)
            d = wm.detect(y, span, MESSAGE)
            zs.append(d["z"]); pb.append(d["p_bound"]); acc.append(d["matches"] / NPR)
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            txt.append(M.tok.decode(y[0, span], skip_special_tokens=True))
        pw = np.nanmedian(ppl(txt))
        rows.append(dict(tau=tau, gate=pool, z=float(np.mean(zs)),
                         p_bound=float(np.median(pb)), acc=float(np.mean(acc)),
                         changed=float(np.mean(ch)), ppl=float(pw), dppl=float(pw - d0)))
        r = rows[-1]
        print(f"tau={tau:<4} gate={str(pool):<5} | z {r['z']:+.2f} | median p_bound "
              f"{r['p_bound']:.1e} | bit acc {r['acc']:.3f} | changed {r['changed']:.3f} "
              f"| ppl {r['ppl']:.1f} ({r['dppl']:+.1f}, x{r['ppl']/d0:.2f})", flush=True)
    json.dump(dict(rows=rows, ppl_draft=float(d0)),
              open("/ssd1/ming/basinmark/results/tau.json", "w"), indent=1)


if __name__ == "__main__":
    main()
