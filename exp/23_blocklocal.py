"""Block-local BasinMark: does steering commit order only stay near the quality floor?

The two things this has to beat are measured, not assumed:
  * post-hoc substitution -- best TPR@1% inside x1.35 perplexity was 0.10;
  * global two-phase generation -- x1.98 perplexity before any watermark at all.

The claim under test is that steering *which* position commits, while never changing a
candidate token, costs close to nothing -- the reason dgMARK pays only x1.10. The control
that makes this falsifiable is the middle arm: the same block-local schedule with the
watermark preference switched off (`tau_conf` raised so nothing is eligible early and the
scheduled top-k branch does all the work), which recovers reference decoding. If the
watermarked arm's perplexity sits on top of it and z is meaningfully positive, the channel
works; if z stays at zero, the model's own preferred tokens are simply never compatible
often enough and order steering cannot carry this statistic.
"""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.blocklocal import BlockMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, NS, STEPS, BLK = 256, 10, 128, 32
GRID = [dict(tau_conf=0.5, holes=2), dict(tau_conf=0.9, holes=2),
        dict(tau_conf=0.9, holes=8), dict(tau_conf=0.99, holes=2)]


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
    prompts = c4_prompts(M.tok, NS)
    spans = [(p.shape[1], GEN) for p in prompts]

    ref = [M.generate(p, gen_len=GEN, steps=STEPS, block_len=BLK, temperature=0.8,
                      seed=3000 + i).cpu() for i, p in enumerate(prompts)]
    n_ref = nll([M.tok.decode(x[0, pl:pl + GEN], skip_special_tokens=True)
                 for x, (pl, _) in zip(ref, spans)])
    ppl0 = float(np.exp(np.nanmean(n_ref)))
    # what the reference text already scores under the same detector, i.e. the null
    z_ref = [BlockMark(M, KEY, block_len=BLK, nonce=f"doc-{i}").detect(
        x, spans[i][0], GEN, MESSAGE)["z"] for i, x in enumerate(ref)]
    print(f"[reference] ppl {ppl0:.2f}   z on unwatermarked text {np.mean(z_ref):+.2f}",
          flush=True)

    rows = []
    for cfg in GRID:
        zs, pb, txt, st, t0 = [], [], [], [], time.time()
        for i, p in enumerate(prompts):
            w = BlockMark(M, KEY, block_len=BLK, probes_per_block=2, n_patterns=6,
                          n_ablations=3, nonce=f"doc-{i}", **cfg)
            y = w.generate(p, gen_len=GEN, steps=STEPS, temperature=0.8,
                           message=MESSAGE, seed=3000 + i)
            d = w.detect(y, spans[i][0], GEN, MESSAGE)
            zs.append(d["z"]); pb.append(d["p_bound"]); st.append(w.stats)
            txt.append(M.tok.decode(y[0, spans[i][0]:spans[i][0] + GEN],
                                    skip_special_tokens=True))
        nw = nll(txt); pb = np.array(pb)
        r = dict(**cfg, z=float(np.mean(zs)), ppl=float(np.exp(np.nanmean(nw))),
                 ratio=float(np.exp(np.nanmean(nw)) / ppl0),
                 tpr05=float(np.mean(pb < 0.05)), tpr01=float(np.mean(pb < 0.01)),
                 deferred=float(np.mean([s["deferred"] for s in st])),
                 compatible=float(np.mean([s["compatible"] for s in st])),
                 fallback=float(np.mean([s["fallback"] for s in st])),
                 n_forwards=int(GEN // BLK * 6))
        rows.append(r)
        print(f"tau_conf={cfg['tau_conf']:<5} H={cfg['holes']:<2} | z {r['z']:+.2f} | "
              f"TPR@5% {r['tpr05']:.2f} @1% {r['tpr01']:.2f} | ppl {r['ppl']:.2f} "
              f"(x{r['ratio']:.2f}) | compat {r['compatible']:.0f} defer "
              f"{r['deferred']:.0f} fallback {r['fallback']:.0f} | "
              f"{(time.time()-t0)/NS:.0f}s/sample", flush=True)
        json.dump(dict(rows=rows, ppl_ref=ppl0, z_ref=float(np.mean(z_ref))),
                  open("/ssd1/ming/basinmark/results/blocklocal.json", "w"), indent=1)

    print("\n===== BLOCK-LOCAL BasinMark =====")
    print(f"reference ppl {ppl0:.2f}; detector on unwatermarked text z "
          f"{np.mean(z_ref):+.2f}")
    print(f"{'tau_conf':<10}{'H':<4}{'ppl ratio':>11}{'z':>8}{'TPR@1%':>9}"
          f"{'det.fwd':>9}")
    for r in rows:
        print(f"{r['tau_conf']:<10}{r['holes']:<4}{r['ratio']:>11.2f}{r['z']:>+8.2f}"
              f"{r['tpr01']:>9.2f}{r['n_forwards']:>9}")
    print("\nfor reference: post-hoc best inside x1.35 was TPR@1% 0.10; global two-phase "
          "cost x1.98 with no watermark; local dgMARK 3-beam is 0.86 at x1.23")


if __name__ == "__main__":
    main()
