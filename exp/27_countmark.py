"""CountMark under the paper's protocol, against the local dgMARK and KGW numbers.

Targets to beat, all measured on this machine, same checkpoint, same C4 prompts, same 256
generated tokens, same GPT-2-large perplexity:

    KGW delta=1        TPR@1% 0.93   x1.03 (x1.20 against block decoding)   0 forwards
    dgMARK 3-beam      TPR@1% 0.86   x1.23                                  0 forwards
    BasinMark post-hoc TPR@1% 0.10   x1.17                                  9 forwards

Axes: the challenge construction (random pair vs near-context-versus-far-context, which
should widen g's dynamic range), the step budget (more steps means more chances for a
deferred position to be re-drawn), and the confidence gate.
"""
import sys, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.countmark import CountMark

KEY, MESSAGE = b"basinmark-key-A", 0xA5
GEN, NS, BLK = 256, 10, 32
GRID = list(itertools.product(["contrast", "random"], [128, 256], [0.3, 0.7]))


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

    ref_ppl, ref_stat = {}, {}
    for steps in sorted(set(s for _, s, _ in GRID)):
        ref = [M.generate(p, gen_len=GEN, steps=steps, block_len=BLK, temperature=0.8,
                          seed=3000 + i).cpu() for i, p in enumerate(prompts)]
        n = nll([M.tok.decode(x[0, pl[i]:pl[i] + GEN], skip_special_tokens=True)
                 for i, x in enumerate(ref)])
        ref_ppl[steps] = float(np.exp(np.nanmean(n)))
        for ch in ("contrast", "random"):
            zz = [CountMark(M, KEY, block_len=BLK, challenge=ch, nonce=f"doc-{i}").detect(
                x, pl[i], GEN, MESSAGE) for i, x in enumerate(ref)]
            ref_stat[(steps, ch)] = (float(np.mean([d["z"] for d in zz])),
                                     float(np.mean([d["rate"] for d in zz])))
            print(f"[reference steps={steps} {ch}] ppl {ref_ppl[steps]:.2f}  "
                  f"z {ref_stat[(steps, ch)][0]:+.2f}  match rate "
                  f"{ref_stat[(steps, ch)][1]:.3f}  (want 0.500)  ties "
                  f"{np.mean([d['tie_frac'] for d in zz]):.3f}", flush=True)

    rows = []
    for ch, steps, tc in GRID:
        zs, ps, rate, acc, txt, st, ties, t0 = [], [], [], [], [], [], [], time.time()
        for i, p in enumerate(prompts):
            w = CountMark(M, KEY, block_len=BLK, n_patterns=4, tau_conf=tc, holes=4,
                          n_bits=8, challenge=ch, nonce=f"doc-{i}")
            y = w.generate(p, gen_len=GEN, steps=steps, temperature=0.8,
                           message=MESSAGE, seed=3000 + i)
            d = w.detect(y, pl[i], GEN, MESSAGE)
            zs.append(d["z"]); ps.append(d["p_value"]); rate.append(d["rate"])
            acc.append(d["bit_acc"]); st.append(w.stats); ties.append(d["tie_frac"])
            txt.append(M.tok.decode(y[0, pl[i]:pl[i] + GEN], skip_special_tokens=True))
        nw = nll(txt); ps = np.array(ps)
        wm = float(np.mean([s["wm"] for s in st]))
        cm = float(np.mean([s["committed"] for s in st]))
        r = dict(challenge=ch, steps=steps, tau_conf=tc,
                 z=float(np.mean(zs)), z_ref=ref_stat[(steps, ch)][0],
                 rate=float(np.mean(rate)), rate_ref=ref_stat[(steps, ch)][1],
                 bit_acc=float(np.mean(acc)),
                 ppl=float(np.exp(np.nanmean(nw))),
                 ratio=float(np.exp(np.nanmean(nw)) / ref_ppl[steps]),
                 tpr05=float(np.mean(ps < 0.05)), tpr01=float(np.mean(ps < 0.01)),
                 tpr001=float(np.mean(ps < 0.001)),
                 wm_frac=wm / max(cm, 1), tie_frac=float(np.mean(ties)),
                 n_forwards=(GEN // BLK) * 4)
        rows.append(r)
        print(f"{ch:<9} steps={steps:<4} tau={tc:<4} | match {r['rate']:.3f} "
              f"(ref {r['rate_ref']:.3f}) | z {r['z']:+.2f} | TPR@5% {r['tpr05']:.2f} "
              f"@1% {r['tpr01']:.2f} @0.1% {r['tpr001']:.2f} | bits {r['bit_acc']:.2f} | "
              f"ppl {r['ppl']:.2f} (x{r['ratio']:.2f}) | wm-driven {r['wm_frac']:.2f} "
              f"ties {r['tie_frac']:.2f} | "
              f"{(time.time()-t0)/NS:.0f}s/sample", flush=True)
        json.dump(dict(rows=rows, ref_ppl=ref_ppl,
                       ref_stat={f"{k[0]}_{k[1]}": v for k, v in ref_stat.items()}),
                  open("/ssd1/ming/basinmark/results/countmark.json", "w"), indent=1)

    print("\n===== CountMark =====")
    print(f"{'challenge':<10}{'steps':<7}{'tau':<6}{'match':>8}{'z':>8}{'TPR@1%':>9}"
          f"{'ppl':>9}{'fwd':>6}")
    for r in rows:
        print(f"{r['challenge']:<10}{r['steps']:<7}{r['tau_conf']:<6}{r['rate']:>8.3f}"
              f"{r['z']:>+8.2f}{r['tpr01']:>9.2f}{r['ratio']:>9.2f}{r['n_forwards']:>6}")
    ok = [r for r in rows if r["ratio"] <= 1.20]
    if ok:
        b = max(ok, key=lambda r: r["tpr01"])
        print(f"\nbest inside ppl x1.20: TPR@1% {b['tpr01']:.2f} at ppl x{b['ratio']:.2f} "
              f"({b['challenge']}, steps={b['steps']}, tau={b['tau_conf']})")
    else:
        print("\nno configuration inside ppl x1.20")
    print("targets: KGW d=1 0.93 @ x1.03 (x1.20 vs block dec.); dgMARK 3-beam 0.86 @ x1.23")


if __name__ == "__main__":
    main()
