"""Evaluate the dgMARK reproduction on exactly the axes BasinMark is reported on.

The point of this run is an anchor, not a leaderboard: we do not currently know whether
"TPR is low at ppl x1.2" is a BasinMark problem or the price every dLLM watermark pays.

dgMARK's detector standardises the match ratio against the empirical mean/sd of the
non-watermarked set, so it supplies its own null. Two thresholds are reported:
  * analytic -- parity gives a Binomial(n, 1/2) null, z = (r - 1/2) * sqrt(n) / (1/2),
    threshold 2.326 for FPR 1%. This is the form the literature reports and it does not
    run out of resolution.
  * empirical -- quantile of the non-watermarked z. Resolution is 1/n_null, so with 50
    null samples FPR=1% is NOT resolvable; reported with that caveat, never rounded to 0.
"""
import sys, csv, json, glob
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch

OUT = "/ssd1/ming/basinmark/results/baselines"


def find(tag):
    """generate.py appends its method to --output_prefix, so the file is
    dgmark_<tag>_<tag>.csv rather than dgmark_<tag>.csv."""
    hits = sorted(glob.glob(f"{OUT}/dgmark_{tag}*.csv"))
    if not hits:
        raise FileNotFoundError(f"no dgMARK {tag} csv under {OUT}")
    return hits[0]


def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["prompt"], r["generated"], float(r["match_ratio"]),
                         int(r["trimmed_length"])))
    return rows


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


def arm(tag, og_rows, n_o, nll):
    try:
        f = find(tag)
    except FileNotFoundError:
        print(f"[{tag}] not present, skipped", flush=True)
        return None
    rows = load(f)
    r = np.array([x[2] for x in rows]); n = np.array([x[3] for x in rows])
    ro = np.array([x[2] for x in og_rows]); no = np.array([x[3] for x in og_rows])
    z = (r - 0.5) * np.sqrt(n) / 0.5
    zo = (ro - 0.5) * np.sqrt(no) / 0.5
    nw = nll([x[1] for x in rows])
    k = min(len(nw), len(n_o))
    dn = nw[:k] - n_o[:k]                        # paired: same prompt list, same order
    out = dict(tag=tag, file=f.split("/")[-1], n=len(rows), rate=float(r.mean()),
               z=float(z.mean()), ppl=float(np.exp(np.nanmean(nw))),
               ratio=float(np.exp(np.nanmean(nw)) / np.exp(np.nanmean(n_o))),
               dnll=float(np.nanmean(dn)),
               dnll_q=[float(np.nanquantile(dn, q)) for q in (.25, .5, .75)],
               tpr={str(a): float(np.mean(z > t))
                    for a, t in ((0.05, 1.645), (0.01, 2.326), (0.001, 3.090))},
               fpr={str(a): float(np.mean(zo > t))
                    for a, t in ((0.05, 1.645), (0.01, 2.326), (0.001, 3.090))},
               z_all=z.tolist(), zo_all=zo.tolist(), nll=nw.tolist())
    print(f"[{tag:<10}] n={out['n']:<3} match {out['rate']:.4f}  z {out['z']:+.2f}  "
          f"TPR@5% {out['tpr']['0.05']:.2f} @1% {out['tpr']['0.01']:.2f} "
          f"@0.1% {out['tpr']['0.001']:.2f}  ppl {out['ppl']:.2f} "
          f"(x{out['ratio']:.3f})  dNLL {out['dnll']:+.4f}", flush=True)
    return out


def main():
    fo = find("original")
    og = load(fo)
    nll = Nll()
    n_o = nll([r[1] for r in og])
    ro = np.array([r[2] for r in og]); no = np.array([r[3] for r in og])
    zo = (ro - 0.5) * np.sqrt(no) / 0.5
    print(f"[original  ] n={len(og)} match {ro.mean():.4f}  z {zo.mean():+.2f} "
          f"(sd {zo.std():.2f})  ppl {np.exp(np.nanmean(n_o)):.2f}", flush=True)

    arms = [a for a in (arm("watermark", og, n_o, nll), arm("beam3", og, n_o, nll))
            if a is not None]

    print("\n===== dgMARK @ LLaDA-8B-Instruct, C4 (300-char prompts), 256 tokens =====")
    print(f"{'arm':<12}{'ppl ratio':>11}{'TPR@5%':>9}{'TPR@1%':>9}{'TPR@0.1%':>10}{'z':>8}")
    for a in arms:
        print(f"{a['tag']:<12}{a['ratio']:>11.3f}{a['tpr']['0.05']:>9.2f}"
              f"{a['tpr']['0.01']:>9.2f}{a['tpr']['0.001']:>10.2f}{a['z']:>+8.2f}")
    print(f"observed FPR on the non-watermarked arm at the same thresholds: "
          + "  ".join(f"{a}: {np.mean(zo > t):.2f}"
                      for a, t in ((0.05, 1.645), (0.01, 2.326), (0.001, 3.090))))
    print(f"empirical-null resolution is 1/{len(og)} = {1/len(og):.3f}; FPR=0.1% is "
          f"below it and the analytic threshold is what carries that column")
    print("detection cost: 0 model forwards (token-id parity only)")
    json.dump(dict(original=dict(n=len(og), rate=float(ro.mean()), z=float(zo.mean()),
                                 ppl=float(np.exp(np.nanmean(n_o))), nll=n_o.tolist(),
                                 z_all=zo.tolist()), arms=arms),
              open("/ssd1/ming/basinmark/results/dgmark_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
