"""Go/no-go: TPR at a controlled FPR against the perplexity actually paid.

Fixed since the previous version:
  * `none` is now a keyed random pick inside each probe's pool share, not the first
    positions of a position-sorted array;
  * every document gets its own nonce, K_d = HMAC(K, nonce_d), so the Hoeffding
    guarantee P(p_bound <= alpha) <= alpha applies per document and the fraction below
    alpha is a TPR at a controlled FPR rather than 12 draws sharing one key;
  * the leverage score is probe-conditioned -- the range of the aggregate guidance the
    embedder will actually push along, not the mean of individual pair ranges.

Also reports how often the top-K candidate slate truncates the admissible set, since a
truncated slate would silently bias U and C.
"""
import sys, gzip, json, time, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.select import LeverageMark

KEY, MESSAGE = b"basinmark-key-A", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 10, 16
SELECTS = ["none", "entropy", "leverage"]
TAUS, LAMS = [1.0, 2.0, 3.0], [8.0, 20.0]
SHAPES = [(0.50, 0.30), (0.60, 0.15)]        # (pool_rate, carrier_rate)


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
    drafts = []
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        drafts.append((x, np.arange(p.shape[1], p.shape[1] + GEN)))
    n0 = nll([M.tok.decode(x[0, s], skip_special_tokens=True) for x, s in drafts])
    print(f"[drafts] n={NS}  ppl {np.exp(np.nanmean(n0)):.1f}", flush=True)

    rows = []
    for (pool, carr), sel, tau, lam in itertools.product(SHAPES, SELECTS, TAUS, LAMS):
        zs, pb, ch, txt, adm = [], [], [], [], []
        for i, (x, span) in enumerate(drafts):
            wm = LeverageMark(M, KEY, n_probes=NPR, pool_rate=pool, carrier_rate=carr,
                              ctx_rate=0.15, tau=tau, lam=lam, commit_steps=2,
                              n_patterns=8, n_ablations=6, select=sel,
                              nonce=f"doc-{i}")            # per-document key
            y = wm.embed(x, span, MESSAGE)
            d = wm.detect(y, span, MESSAGE)
            zs.append(d["z"]); pb.append(d["p_bound"])
            ch.append(float((y[0, span] != x[0, span]).float().mean()))
            txt.append(M.tok.decode(y[0, span], skip_special_tokens=True))
            adm.append(wm.adm_stats)
        nw = nll(txt); dn = nw - n0; pb = np.array(pb)
        r = dict(pool=pool, carrier=carr, select=sel, tau=tau, lam=lam,
                 z=float(np.mean(zs)), tpr01=float(np.mean(pb < 0.01)),
                 tpr05=float(np.mean(pb < 0.05)), tpr001=float(np.mean(pb < 0.001)),
                 dnll_mean=float(np.nanmean(dn)),
                 dnll_q=[float(np.nanquantile(dn, q)) for q in (0.25, 0.5, 0.75)],
                 ppl_ratio=float(np.exp(np.nanmean(nw)) / np.exp(np.nanmean(n0))),
                 changed=float(np.mean(ch)),
                 adm_median=float(np.mean([a["median"] for a in adm])),
                 adm_capped=float(np.mean([a["frac_capped"] for a in adm])))
        rows.append(r)
        print(f"pool={pool} carr={carr} {sel:<9} tau={tau:<4} lam={lam:<5} | "
              f"TPR@1% {r['tpr01']:.2f} @5% {r['tpr05']:.2f} | z {r['z']:+.2f} | "
              f"ppl x{r['ppl_ratio']:.2f} (dNLL {r['dnll_mean']:+.3f}) | "
              f"chg {r['changed']:.3f} | adm med {r['adm_median']:.0f} "
              f"capped {r['adm_capped']:.2f}", flush=True)
        json.dump(dict(rows=rows, nll_draft=n0.tolist()),
                  open("/ssd1/ming/basinmark/results/pareto.json", "w"), indent=1)

    print("\n===== PARETO (TPR@1% FPR vs perplexity ratio) =====")
    for cap in (1.20, 1.35, 1.60):
        ok = [r for r in rows if r["ppl_ratio"] <= cap]
        if ok:
            b = max(ok, key=lambda r: r["tpr01"])
            print(f"  ppl <= x{cap}: best TPR@1% {b['tpr01']:.2f}  "
                  f"({b['select']} pool={b['pool']} carr={b['carrier']} "
                  f"tau={b['tau']} lam={b['lam']}, ppl x{b['ppl_ratio']:.2f})")
        else:
            print(f"  ppl <= x{cap}: NO configuration")
    for sel in SELECTS:
        sub = [r for r in rows if r["select"] == sel and r["ppl_ratio"] <= 1.35]
        best = max(sub, key=lambda r: r["tpr01"]) if sub else None
        print(f"  {sel:<9} best TPR@1% under ppl x1.35: "
              + (f"{best['tpr01']:.2f}" if best else "no config qualifies"))


if __name__ == "__main__":
    main()
