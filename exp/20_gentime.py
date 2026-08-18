"""Post-hoc vs generation-time, at matched tau/lambda.

The hypothesis is that the post-hoc perplexity blow-up comes from refilling 30 % of the
span from near-independent marginals, not from the per-token substitution price (which is
identical either way). So the decisive control is the middle arm below: two-phase
generation with NO guidance. It isolates the cost of the forced decoding order --
non-pool positions first, pool last -- from the cost of the watermark itself.

  reference     normal LLaDA generation                 -> quality floor
  two-phase     same, but pool positions filled last    -> cost of the ORDER alone
  guided        two-phase + watermark objective          -> total cost
"""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.select import LeverageMark
from basinmark.gentime import GenMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR, STEPS = 256, 40, 8, 16, 128
GRID = list(itertools.product([2.0, 3.0, 6.0], [8.0, 20.0]))
SHAPE = dict(n_probes=NPR, pool_rate=0.50, carrier_rate=0.30, ctx_rate=0.15,
             n_patterns=8, n_ablations=6, select="leverage")


def c4(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


class Nll:
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
            o.append(float(self.m(ids, labels=ids).loss) if ids.shape[1] >= 8 else np.nan)
        return np.array(o)


def main():
    M = BasinModel()
    nll = Nll()
    prompts = c4(M.tok, NS, PREFIX)
    spans = [np.arange(p.shape[1], p.shape[1] + GEN) for p in prompts]

    ref = [M.generate(p, gen_len=GEN, steps=STEPS, block_len=32, temperature=0.8,
                      seed=3000 + i).cpu() for i, p in enumerate(prompts)]
    n_ref = nll([M.tok.decode(x[0, s], skip_special_tokens=True) for x, s in zip(ref, spans)])
    ppl0 = float(np.exp(np.nanmean(n_ref)))
    print(f"[reference generation] ppl {ppl0:.2f}", flush=True)

    # control: the two-phase order alone, no watermark objective
    ctrl, zc = [], []
    for i, p in enumerate(prompts):
        g = GenMark(M, KEY, tau=6.0, lam=0.0, nonce=f"doc-{i}", **SHAPE)
        y = g.generate(p, gen_len=GEN, steps=STEPS, temperature=0.8, message=MESSAGE,
                       seed=3000 + i)
        ctrl.append(y); zc.append(g.detect(y, spans[i], MESSAGE)["z"])
    n_ctrl = nll([M.tok.decode(x[0, s], skip_special_tokens=True)
                  for x, s in zip(ctrl, spans)])
    ppl_c = float(np.exp(np.nanmean(n_ctrl)))
    R_order = ppl_c / ppl0
    print(f"[two-phase order, lam=0] ppl {ppl_c:.2f} (x{R_order:.2f})  "
          f"z {np.mean(zc):+.2f}  <- cost of the decoding order alone", flush=True)

    # This number decides whether the rest of the sweep is worth any GPU time. If the
    # forced schedule -- all non-pool positions, then all pool positions -- already costs
    # more than the whole dgMARK watermark does (x1.10 for 92% TPR), then no lambda/tau
    # can rescue it and the formulation, not the tuning, is wrong. One configuration is
    # still run for the record.
    grid = GRID
    if R_order > 1.35:
        print(f"\n*** ABORT SWEEP: the two-phase schedule alone costs x{R_order:.2f}, "
              f"before any watermark. dgMARK's ENTIRE cost is x1.10. Global two-phase "
              f"generation is the wrong formulation; block-local is the next step. "
              f"Running one config for the record only. ***\n", flush=True)
        grid = [(6.0, 20.0)]

    rows = []
    for tau, lam in grid:
        for mode in ("posthoc", "gentime"):
            zs, pb, txt = [], [], []
            for i, p in enumerate(prompts):
                if mode == "gentime":
                    w = GenMark(M, KEY, tau=tau, lam=lam, commit_steps=2,
                                nonce=f"doc-{i}", **SHAPE)
                    y = w.generate(p, gen_len=GEN, steps=STEPS, temperature=0.8,
                                   message=MESSAGE, seed=3000 + i)
                else:
                    w = LeverageMark(M, KEY, tau=tau, lam=lam, commit_steps=2,
                                     nonce=f"doc-{i}", **SHAPE)
                    y = w.embed(ref[i], spans[i], MESSAGE)
                d = w.detect(y, spans[i], MESSAGE)
                zs.append(d["z"]); pb.append(d["p_bound"])
                txt.append(M.tok.decode(y[0, spans[i]], skip_special_tokens=True))
            nw = nll(txt); pb = np.array(pb)
            r = dict(mode=mode, tau=tau, lam=lam, z=float(np.mean(zs)),
                     tpr01=float(np.mean(pb < 0.01)), tpr05=float(np.mean(pb < 0.05)),
                     ppl=float(np.exp(np.nanmean(nw))),
                     ratio=float(np.exp(np.nanmean(nw)) / ppl0))
            rows.append(r)
            print(f"{mode:<8} tau={tau:<4} lam={lam:<5} | z {r['z']:+.2f} | "
                  f"TPR@1% {r['tpr01']:.2f} @5% {r['tpr05']:.2f} | "
                  f"ppl {r['ppl']:.2f} (x{r['ratio']:.2f})", flush=True)
            json.dump(dict(rows=rows, ppl_ref=ppl0, ppl_ctrl=ppl_c,
                           z_ctrl=float(np.mean(zc))),
                      open("/ssd1/ming/basinmark/results/gentime.json", "w"), indent=1)

    print("\n===== post-hoc vs generation-time =====")
    print(f"reference ppl {ppl0:.2f}; two-phase order alone x{R_order:.2f}")
    for tau, lam in grid:
        a = [r for r in rows if r["tau"] == tau and r["lam"] == lam]
        p_, g_ = a[0], a[1]
        print(f"  tau={tau} lam={lam}:  post-hoc z {p_['z']:+.2f} x{p_['ratio']:.2f}"
              f"   ->  gen-time z {g_['z']:+.2f} x{g_['ratio']:.2f}")
    print("\ndgMARK reference point: x1.10 ppl -> TPR@1% 0.92 ; x1.25 -> 0.99")


if __name__ == "__main__":
    main()
