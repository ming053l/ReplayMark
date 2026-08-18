"""Does better generation quality reduce post-hoc watermark capacity?

Fixing the sampler took draft perplexity 22.2 -> 9.5 and made the watermark much harder
to embed. The proposed mechanism -- better text sits nearer the model's mode, so the
denoiser's conditional at each position is sharper and the admissible set smaller -- is
so far only an inference from those two facts. Measure it directly: denoiser entropy and
|A_i(tau)| at pool positions, for drafts from the buggy and the reference sampler.
"""
import sys, gzip, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.select import pool_patterns

GEN, PREFIX, NS = 256, 40, 16
TAUS = [1.0, 2.0, 3.0, 6.0]


def main():
    M = BasinModel()
    import os
    os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    gm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                              torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def ppl(t):
        ids = tk(t, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
        return float(torch.exp(gm(ids, labels=ids).loss))

    prompts = c4_prompts(M.tok, NS)
    out = {}
    for tag, legacy in (("reference", False), ("buggy", True)):
        P, H, A = [], [], {t: [] for t in TAUS}
        for i, p in enumerate(prompts):
            x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                           seed=3000 + i, legacy_conf=legacy).cpu()
            span = np.arange(p.shape[1], p.shape[1] + GEN)
            P.append(ppl(M.tok.decode(x[0, span], skip_special_tokens=True)))
            Q, _, _, _ = pool_patterns(b"cap", GEN, 16, 0.50, 0.15, 8, 6)
            Q = span[Q]
            base = M.logprobs_rows(M.corrupt(x, Q), torch.tensor(Q), chunk=1)[0]
            H.append(float((-(base.exp() * base).sum(1)).mean()))
            top = base.max(1, keepdim=True).values
            for t in TAUS:
                A[t].append(float((base >= (top - t)).sum(1).float().mean()))
        out[tag] = dict(ppl=float(np.median(P)), H=float(np.mean(H)),
                        A={str(t): float(np.mean(v)) for t, v in A.items()})
        print(f"{tag:<10} ppl {out[tag]['ppl']:6.1f}  denoiser entropy {out[tag]['H']:.3f}  "
              + "  ".join(f"|A({t})| {out[tag]['A'][str(t)]:.1f}" for t in TAUS), flush=True)

    r, b = out["reference"], out["buggy"]
    print(f"\nreference vs buggy: ppl {r['ppl']:.1f} vs {b['ppl']:.1f}, "
          f"entropy {r['H']:.3f} vs {b['H']:.3f} "
          f"({100*(r['H']-b['H'])/b['H']:+.1f} %)")
    for t in TAUS:
        print(f"  |A({t})|: {r['A'][str(t)]:.1f} vs {b['A'][str(t)]:.1f} "
              f"({100*(r['A'][str(t)]-b['A'][str(t)])/b['A'][str(t)]:+.1f} %)")
    json.dump(out, open("/ssd1/ming/basinmark/results/capacity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
