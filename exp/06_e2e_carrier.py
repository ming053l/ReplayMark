"""Full end-to-end for BasinMark-C: detection p-values, negative controls, quality."""
import sys, gzip, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.carrier import CarrierMark

KEY, WRONG, MESSAGE = b"basinmark-key-A", b"basinmark-key-B", 0xA53C7
GEN, PREFIX, NS, NPR = 256, 40, 24, 16
CFG = dict(n_probes=NPR, carrier_rate=0.30, ctx_rate=0.20, tau=6.0, lam=3.0,
           commit_steps=8, n_ablations=3)


def c4(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


def ppl_scores(texts):
    """Perplexity of each completion under an independent AR model (GPT-2 large)."""
    import os
    os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large",
                                             torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tk(t, return_tensors="pt", truncation=True, max_length=512).input_ids.cuda()
            if ids.shape[1] < 8:
                out.append(float("nan")); continue
            out.append(float(torch.exp(m(ids, labels=ids).loss)))
    del m; torch.cuda.empty_cache()
    return out


def main():
    M = BasinModel()
    wm, wrong = CarrierMark(M, KEY, **CFG), CarrierMark(M, WRONG, **CFG)
    rows, draft_txt, wm_txt = [], [], []
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        t0 = time.time()
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        neg = wm.detect(x, span, MESSAGE)
        y = wm.embed(x, span, MESSAGE)
        pos = wm.detect(y, span, MESSAGE)
        bad = wrong.detect(y, span, MESSAGE)
        draft_txt.append(M.tok.decode(x[0, span], skip_special_tokens=True))
        wm_txt.append(M.tok.decode(y[0, span], skip_special_tokens=True))
        rows.append(dict(i=i, pos=pos["matches"], neg=neg["matches"], bad=bad["matches"],
                         p_pos=pos["p_value"], p_neg=neg["p_value"], p_bad=bad["p_value"],
                         z_pos=pos["z"], z_neg=neg["z"], z_bad=bad["z"],
                         changed=float((y[0, span] != x[0, span]).float().mean()),
                         cost=wm.last_cost, t=time.time() - t0))
        print(f"[{i:02d}] wm {pos['matches']}/{NPR} z={pos['z']:+.2f} p={pos['p_value']:.1e} | "
              f"no-wm z={neg['z']:+.2f} | wrong-key z={bad['z']:+.2f} | "
              f"changed {rows[-1]['changed']:.3f} cost {rows[-1]['cost']:.2f} | "
              f"{rows[-1]['t']:.0f}s", flush=True)
        if i == 0:
            print("  --- draft ---\n  " + draft_txt[0][:350])
            print("  --- watermarked ---\n  " + wm_txt[0][:350], flush=True)

    del M.model; torch.cuda.empty_cache()
    pd, pw = ppl_scores(draft_txt), ppl_scores(wm_txt)
    a = {k: np.array([r[k] for r in rows]) for k in ("pos", "neg", "bad", "changed", "cost")}
    print("\n===== BasinMark-C END TO END =====")
    print(f"{NS} samples, {NPR}-bit payload, {GEN}-token span")
    print(f"sign-matches/{NPR}   watermarked {a['pos'].mean():.1f}   "
          f"no-watermark {a['neg'].mean():.1f}   wrong-key {a['bad'].mean():.1f}   "
          f"(chance {NPR/2:.1f})")
    print(f"bit accuracy {a['pos'].mean()/NPR:.3f}")
    for tag, k in (("wm", "p_pos"), ("no-wm", "p_neg"), ("wrong-key", "p_bad")):
        print(f"  TPR/FPR @ p<0.01   {tag:<10} {np.mean([r[k] < 0.01 for r in rows]):.2f}"
              f"   @ p<1e-6 {np.mean([r[k] < 1e-6 for r in rows]):.2f}")
    from sklearn.metrics import roc_auc_score
    zp = [r["z_pos"] for r in rows]
    for tag, k in (("vs no-watermark", "z_neg"), ("vs wrong-key", "z_bad")):
        zn = [r[k] for r in rows]
        print(f"  AUC {tag:<18} {roc_auc_score([1]*len(zp)+[0]*len(zn), zp+zn):.3f}")
    print(f"  median z   wm {np.median(zp):+.2f}   no-wm {np.median([r['z_neg'] for r in rows]):+.2f}")
    print(f"tokens changed {a['changed'].mean():.3f}   cost {a['cost'].mean():.2f} nats/carrier")
    print(f"GPT-2-large ppl  draft {np.nanmedian(pd):.1f}  watermarked {np.nanmedian(pw):.1f}")
    json.dump(dict(rows=rows, ppl_draft=pd, ppl_wm=pw, cfg=CFG),
              open("/ssd1/ming/basinmark/results/e2e_carrier.json", "w"), indent=1)


if __name__ == "__main__":
    main()
