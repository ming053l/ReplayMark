"""ReTrace V3 MVP: watermark strength against retries, on the baselines' protocol.

R is the only knob that matters now, and a retry costs no forward pass, so the curve to
report is strength against R. Also logged: the acceptance mass actually available at the
selected carriers, which is what makes the theoretical curve 1 - (1 - q)^R attainable or
not, and every quality axis a reviewer will ask for once the sampling distribution is
conditioned -- perplexity, distinct-n and repetition, not perplexity alone.

Protocol matches the baseline table: LLaDA-8B-Instruct, C4 prompts truncated to 300
characters, 256 generated tokens, GPT-2-large perplexity against a reference-decoded
control, per-document nonces, and a null arm printed for every configuration.
"""
import sys, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch

from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark

KEY, MESSAGE = b"retrace-key-A", 0xA5
GEN, BLK, STEPS, NS = 256, 32, 128, 12
RETRIES = (1, 2, 3, 4, 8)
CARRIER = 0.5


class Ppl:
    def __init__(self):
        import os
        os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
        self.m = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2-large", torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def nll(self, texts):
        o = []
        for t in texts:
            ids = self.tk(t, return_tensors="pt", truncation=True,
                          max_length=512).input_ids.cuda()
            o.append(float(self.m(ids, labels=ids).loss) if ids.shape[1] >= 8 else np.nan)
        return np.array(o)


def diversity(tok, texts):
    """distinct-n and a repetition rate; perplexity alone cannot see a sampler that has
    been conditioned into repeating itself."""
    d2, d3, rep = [], [], []
    for t in texts:
        ids = tok(t)["input_ids"]
        for n, acc in ((2, d2), (3, d3)):
            grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
            acc.append(len(set(grams)) / max(len(grams), 1))
        big = [tuple(ids[i:i + 2]) for i in range(len(ids) - 1)]
        rep.append(1 - len(set(big)) / max(len(big), 1))
    return float(np.mean(d2)), float(np.mean(d3)), float(np.mean(rep))


def main():
    M = BasinModel()
    ppl = Ppl()
    prompts = c4_prompts(M.tok, NS)
    pls = [p.shape[1] for p in prompts]

    ref = [M.generate(p, gen_len=GEN, steps=STEPS, block_len=BLK, temperature=0.8,
                      seed=3000 + i).cpu() for i, p in enumerate(prompts)]
    rtxt = [M.tok.decode(x[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
            for i, x in enumerate(ref)]
    n0 = ppl.nll(rtxt)
    ppl0 = float(np.exp(np.nanmean(n0)))
    rd2, rd3, rrep = diversity(M.tok, rtxt)

    nulls, qs = [], []
    for i, x in enumerate(ref):
        w = ResampleMark(M, KEY, block_len=BLK, carrier_frac=CARRIER, retries=1,
                         nonce=f"doc-{i}")
        nulls.append(w.detect(x, pls[i], GEN, MESSAGE))
        for b in range(GEN // BLK):
            _, _, S, _, _ = w._table(x, pls[i] + b * BLK, pls[i] + GEN)
            qs.append(np.sort(S)[-int(CARRIER * BLK):])
    q = np.concatenate(qs)
    print(f"[reference] ppl {ppl0:.2f}  distinct-2 {rd2:.3f}  distinct-3 {rd3:.3f}  "
          f"repeat {rrep:.3f}", flush=True)
    print(f"[null]      sync rate {np.mean([d['rate_sync'] for d in nulls]):.3f} "
          f"(want 0.500)  n_sync {np.mean([d['n_sync'] for d in nulls]):.0f}", flush=True)
    print(f"[carriers]  two-sided acceptance mass S: median {np.median(q):.3f}, "
          f"mean {q.mean():.3f}  ->  a single draw accepts with prob ~S/2", flush=True)
    for R in RETRIES:
        pred = float(np.mean(1 - (1 - q / 2) ** R))
        print(f"            predicted acceptance at R={R}: {pred:.3f}", flush=True)

    rows = []
    for R in RETRIES:
        ps, rates, accs, txt, st, t0 = [], [], [], [], [], time.time()
        for i, p in enumerate(prompts):
            w = ResampleMark(M, KEY, block_len=BLK, carrier_frac=CARRIER, retries=R,
                             nonce=f"doc-{i}")
            y = w.generate(p, gen_len=GEN, steps=STEPS, message=MESSAGE, seed=3000 + i)
            d = w.detect(y, pls[i], GEN, MESSAGE)
            ps.append(d["p_value"]); rates.append(d["rate_sync"]); accs.append(d["bit_acc"])
            st.append(w.stats)
            txt.append(M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True))
        nw = ppl.nll(txt); ps = np.array(ps)
        d2, d3, rep = diversity(M.tok, txt)
        acc = float(np.mean([s["accepted"] / max(s["carrier"], 1) for s in st]))
        dr = float(np.mean([s["draws"] / max(s["committed"], 1) for s in st]))
        r = dict(R=R, rate=float(np.mean(rates)),
                 rate_ref=float(np.mean([d["rate_sync"] for d in nulls])),
                 tpr01=float(np.mean(ps < 0.01)), tpr05=float(np.mean(ps < 0.05)),
                 tpr001=float(np.mean(ps < 0.001)), bit_acc=float(np.mean(accs)),
                 ppl=float(np.exp(np.nanmean(nw))),
                 ratio=float(np.exp(np.nanmean(nw)) / ppl0),
                 accept=acc, draws_per_token=dr, d2=d2, d3=d3, rep=rep,
                 gen_forwards=STEPS, det_seqs=int((GEN // BLK) * 5),
                 secs=(time.time() - t0) / len(prompts))
        rows.append(r)
        print(f"R={R:<3} | sync {r['rate']:.3f} (ref {r['rate_ref']:.3f}) | "
              f"TPR@5% {r['tpr05']:.2f} @1% {r['tpr01']:.2f} @0.1% {r['tpr001']:.2f} | "
              f"bits {r['bit_acc']:.2f} | ppl {r['ppl']:.2f} (x{r['ratio']:.2f}) | "
              f"accepted {r['accept']:.2f} | draws/token {r['draws_per_token']:.2f} | "
              f"d2 {d2:.3f} rep {rep:.3f} | {r['secs']:.0f}s/sample", flush=True)
        json.dump(dict(rows=rows, ppl_ref=ppl0, d2_ref=rd2, d3_ref=rd3, rep_ref=rrep),
                  open("/ssd1/ming/basinmark/results/resample_mvp.json", "w"), indent=1)

    print("\n===== ReTrace V3: retries vs strength =====")
    print(f"{'R':<4}{'ppl ratio':>11}{'TPR@1%':>9}{'accepted':>10}{'draws/tok':>11}"
          f"{'distinct-2':>12}{'gen fwd':>9}{'det seq':>9}")
    for r in rows:
        print(f"{r['R']:<4}{r['ratio']:>11.2f}{r['tpr01']:>9.2f}{r['accept']:>10.2f}"
              f"{r['draws_per_token']:>11.2f}{r['d2']:>12.3f}{r['gen_forwards']:>9}"
              f"{r['det_seqs']:>9}")
    print(f"reference distinct-2 {rd2:.3f}, repeat {rrep:.3f}, ppl {ppl0:.2f}")
    print("targets: KGW d=1 TPR@1% 0.93 at x1.03; dgMARK 3-beam 0.86 at x1.23, both with "
          "zero detection forwards")


if __name__ == "__main__":
    main()
