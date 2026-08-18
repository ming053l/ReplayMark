"""Block-local BasinMark: how far does deferral-as-rejection-sampling get us?

Deferring an incompatible position is not merely a delay. Each step redraws the candidate
with fresh Gumbel noise, so a deferred position is re-sampled from p_theta next step. With
compatibility roughly a coin flip per draw, k retries reach a compatible token with
probability 1 - 2^-k, and every accepted token was drawn from the model's own distribution
-- no token is ever forced. That makes the step budget, not tau or H, the primary lever:
more steps per block means more retries.

Axes:
  steps       128 (2 commits/step) / 256 (1 per step, dgMARK's sequential regime) / 384
  probes/blk  1 (|S_j|=32, low noise per bit) / 2 (|S_j|=16, twice the null blocks)
  tau_conf    0.5 / 0.9   how confident a position must be before it may be committed

Protocol is the one in the paper's table: LLaDA-8B-Instruct, C4 prompts truncated to 300
characters, 256 generated tokens, GPT-2-large perplexity against the reference-decoding
control, TPR at the Hoeffding threshold, detection cost in forward passes.
"""
import sys, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.blocklocal import BlockMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, NS, BLK = 256, 10, 32
GRID = list(itertools.product([128, 256, 384], [1, 2], [0.5, 0.9]))


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
    pl = [p.shape[1] for p in prompts]

    # reference control per step budget: the schedule itself must not be charged to the
    # watermark, and LLaDA's own quality depends on how many steps it is given
    ref_ppl, ref_z = {}, {}
    for steps in sorted(set(s for s, _, _ in GRID)):
        ref = [M.generate(p, gen_len=GEN, steps=steps, block_len=BLK, temperature=0.8,
                          seed=3000 + i).cpu() for i, p in enumerate(prompts)]
        n = nll([M.tok.decode(x[0, pl[i]:pl[i] + GEN], skip_special_tokens=True)
                 for i, x in enumerate(ref)])
        ref_ppl[steps] = float(np.exp(np.nanmean(n)))
        zz = [BlockMark(M, KEY, block_len=BLK, probes_per_block=2,
                        nonce=f"doc-{i}").detect(x, pl[i], GEN, MESSAGE)["z"]
              for i, x in enumerate(ref)]
        ref_z[steps] = float(np.mean(zz))
        print(f"[reference steps={steps}] ppl {ref_ppl[steps]:.2f}  "
              f"z on unwatermarked {ref_z[steps]:+.2f}", flush=True)

    rows = []
    for steps, ppb, tc in GRID:
        zs, pb, txt, st, t0 = [], [], [], [], time.time()
        for i, p in enumerate(prompts):
            w = BlockMark(M, KEY, block_len=BLK, probes_per_block=ppb, n_patterns=6,
                          n_ablations=3, tau_conf=tc, holes=2, nonce=f"doc-{i}")
            y = w.generate(p, gen_len=GEN, steps=steps, temperature=0.8,
                           message=MESSAGE, seed=3000 + i)
            d = w.detect(y, pl[i], GEN, MESSAGE)
            zs.append(d["z"]); pb.append(d["p_bound"]); st.append(w.stats)
            txt.append(M.tok.decode(y[0, pl[i]:pl[i] + GEN], skip_special_tokens=True))
        nw = nll(txt); pb = np.array(pb)
        cm = float(np.mean([s["compatible"] for s in st]))
        fb = float(np.mean([s["fallback"] for s in st]))
        co = float(np.mean([s["committed"] for s in st]))
        r = dict(steps=steps, probes_per_block=ppb, tau_conf=tc,
                 z=float(np.mean(zs)), z_ref=ref_z[steps],
                 ppl=float(np.exp(np.nanmean(nw))),
                 ratio=float(np.exp(np.nanmean(nw)) / ref_ppl[steps]),
                 tpr05=float(np.mean(pb < 0.05)), tpr01=float(np.mean(pb < 0.01)),
                 compatible=cm, fallback=fb, committed=co,
                 forced_frac=fb / max(co, 1),
                 n_blocks=(GEN // BLK) * ppb * 3,
                 n_forwards=(GEN // BLK) * 6)
        rows.append(r)
        print(f"steps={steps:<4} probes/blk={ppb} tau={tc:<4} | z {r['z']:+.2f} "
              f"(ref {r['z_ref']:+.2f}) | TPR@5% {r['tpr05']:.2f} @1% {r['tpr01']:.2f} | "
              f"ppl {r['ppl']:.2f} (x{r['ratio']:.2f}) | forced {r['forced_frac']:.2f} "
              f"| {(time.time()-t0)/NS:.0f}s/sample", flush=True)
        json.dump(dict(rows=rows, ref_ppl=ref_ppl, ref_z=ref_z),
                  open("/ssd1/ming/basinmark/results/blocklocal_sweep.json", "w"), indent=1)

    print("\n===== BLOCK-LOCAL SWEEP =====")
    print(f"{'steps':<7}{'p/blk':<7}{'tau':<6}{'ppl ratio':>11}{'z':>8}{'TPR@1%':>9}"
          f"{'forced':>8}")
    for r in rows:
        print(f"{r['steps']:<7}{r['probes_per_block']:<7}{r['tau_conf']:<6}"
              f"{r['ratio']:>11.2f}{r['z']:>+8.2f}{r['tpr01']:>9.2f}"
              f"{r['forced_frac']:>8.2f}")
    good = [r for r in rows if r["ratio"] <= 1.20]
    if good:
        b = max(good, key=lambda r: r["tpr01"])
        print(f"\nbest inside ppl x1.20: TPR@1% {b['tpr01']:.2f} z {b['z']:+.2f} at "
              f"steps={b['steps']} probes/blk={b['probes_per_block']} tau={b['tau_conf']}")
    else:
        print("\nNO configuration stays inside ppl x1.20")
    print("targets: local dgMARK 3-beam 0.86 at x1.23; post-hoc BasinMark 0.10 at x1.17")
    print("'forced' is the fraction of commits made by the scheduled top-k branch rather "
          "than by watermark preference -- if it is near 1 the channel never gets to act")


if __name__ == "__main__":
    main()
