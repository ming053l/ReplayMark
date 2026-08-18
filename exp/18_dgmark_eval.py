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


def main():
    fw, fo = find("watermark"), find("original")
    wm, og = load(fw), load(fo)
    print(f"loaded {len(wm)} watermarked ({fw.split('/')[-1]}), "
          f"{len(og)} original ({fo.split('/')[-1]})", flush=True)

    rw = np.array([r[2] for r in wm]); nw_ = np.array([r[3] for r in wm])
    ro = np.array([r[2] for r in og]); no_ = np.array([r[3] for r in og])
    zw_a = (rw - 0.5) * np.sqrt(nw_) / 0.5
    zo_a = (ro - 0.5) * np.sqrt(no_) / 0.5
    mu0, sd0 = ro.mean(), ro.std()
    zw_e, zo_e = (rw - mu0) / sd0, (ro - mu0) / sd0

    nll = Nll()
    n_w = nll([r[1] for r in wm])
    n_o = nll([r[1] for r in og])
    k = min(len(n_w), len(n_o))
    dn = n_w[:k] - n_o[:k]                       # paired: same prompt list, same order

    print("\n===== dgMARK @ LLaDA-8B-Instruct, C4, 256 tokens =====")
    print(f"match ratio   watermarked {rw.mean():.4f}   original {ro.mean():.4f}")
    print(f"analytic z    watermarked {zw_a.mean():+.2f}   original {zo_a.mean():+.2f} "
          f"(sd {zo_a.std():.2f})")
    for a, t in ((0.05, 1.645), (0.01, 2.326), (0.001, 3.090)):
        print(f"  TPR @ FPR={a:<6g} (analytic z>{t:.3f}) = {np.mean(zw_a > t):.2f}"
              f"   [observed FPR {np.mean(zo_a > t):.2f}]")
    print(f"empirical null n={len(zo_e)} -> FPR resolution {1/len(zo_e):.3f}; "
          f"1% is below resolution, shown only for completeness")
    for a in (0.05, 0.01):
        thr = float(np.quantile(zo_e, 1 - a))
        print(f"  TPR @ empirical FPR={a:<5g} (z>{thr:.2f}) = {np.mean(zw_e > thr):.2f}")
    print(f"\nGPT-2-large  ppl original {np.exp(np.nanmean(n_o)):.2f}   "
          f"watermarked {np.exp(np.nanmean(n_w)):.2f}   "
          f"ratio x{np.exp(np.nanmean(n_w))/np.exp(np.nanmean(n_o)):.3f}")
    print(f"paired dNLL  mean {np.nanmean(dn):+.4f}  q25 {np.nanquantile(dn,.25):+.4f}  "
          f"q50 {np.nanquantile(dn,.5):+.4f}  q75 {np.nanquantile(dn,.75):+.4f}")
    print("detection cost: 0 model forwards (token-id parity only)")
    json.dump(dict(zw_a=zw_a.tolist(), zo_a=zo_a.tolist(), zw_e=zw_e.tolist(),
                   zo_e=zo_e.tolist(), nll_wm=n_w.tolist(), nll_og=n_o.tolist()),
              open("/ssd1/ming/basinmark/results/dgmark_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
