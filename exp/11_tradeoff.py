"""Detection strength vs text quality -- the curve watermarks are actually compared on.

The entropy gate buys ~24x in p-value but doubles the edit rate, and the internal
substitution cost cannot settle whether that is worth it (gating to high-entropy
positions raises mean cost mechanically, since those are the positions with a large
admissible set). Perplexity under an independent model is the arbiter.
"""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.shared import SharedMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 12, 16
GRID = list(itertools.product([None, 0.60], [3.0, 8.0, 20.0]))


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
        out = []
        for t in texts:
            ids = self.tk(t, return_tensors="pt", truncation=True,
                          max_length=512).input_ids.cuda()
            out.append(float(torch.exp(self.m(ids, labels=ids).loss))
                       if ids.shape[1] >= 8 else float("nan"))
        return np.array(out)


def main():
    M = BasinModel()
    ppl = Ppl()
    drafts = []
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        drafts.append((x, np.arange(p.shape[1], p.shape[1] + GEN)))
    base_txt = [M.tok.decode(x[0, s], skip_special_tokens=True) for x, s in drafts]
    pd = ppl(base_txt)
    print(f"[drafts] n={NS}  GPT-2-large ppl median {np.nanmedian(pd):.1f}", flush=True)

    rows = []
    for pool, lam in GRID:
        cfg = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=lam,
                   commit_steps=2, n_patterns=8, n_ablations=3, pool_rate=pool)
        wm = SharedMark(M, KEY, **cfg)
        zs, pv, acc, ch, txt = [], [], [], [], []
        for x, span in drafts:
            y = wm.embed(x, span, MESSAGE)
            d = wm.detect(y, span, MESSAGE)
            zs.append(d["z"]); pv.append(d["p_value"]); acc.append(d["matches"] / NPR)
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            txt.append(M.tok.decode(y[0, span], skip_special_tokens=True))
        pw = ppl(txt)
        rows.append(dict(gate=pool, lam=lam, z=float(np.mean(zs)),
                         p=float(np.median(pv)), acc=float(np.mean(acc)),
                         changed=float(np.mean(ch)), ppl=float(np.nanmedian(pw)),
                         dppl=float(np.nanmedian(pw) - np.nanmedian(pd))))
        r = rows[-1]
        print(f"gate={str(pool):<5} lam={lam:<5} | z {r['z']:+.2f} | median p {r['p']:.1e} "
              f"| bit acc {r['acc']:.3f} | changed {r['changed']:.3f} "
              f"| ppl {r['ppl']:.1f} (draft {np.nanmedian(pd):.1f}, "
              f"{r['dppl']:+.1f})", flush=True)
    json.dump(dict(rows=rows, ppl_draft=pd.tolist()),
              open("/ssd1/ming/basinmark/results/tradeoff.json", "w"), indent=1)


if __name__ == "__main__":
    main()
