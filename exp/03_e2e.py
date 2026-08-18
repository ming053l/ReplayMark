"""End-to-end: embed a 24-bit payload into LLaDA completions of C4 prompts, then detect."""
import sys, gzip, json, time
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.core import BasinMark

KEY, WRONG = b"basinmark-key-A", b"basinmark-key-B"
N_SAMPLES, GEN, PREFIX = 24, 192, 40
CFG = dict(n_probes=24, probe_rate=0.25, ctx_rate=0.20, tau=4.0, lam=2.0, margin=0.15)
MESSAGE = 0xA53C7


def c4_prefixes(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            t = json.loads(line)["text"]
            ids = tok(t)["input_ids"]
            if len(ids) < ntok + 60:
                continue
            out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
            if len(out) == n:
                return out
    return out


def main():
    M = BasinModel()
    wm = BasinMark(M, KEY, **CFG)
    wm_wrong = BasinMark(M, WRONG, **CFG)
    prefixes = c4_prefixes(M.tok, N_SAMPLES, PREFIX)
    rows = []
    for i, p in enumerate(prefixes):
        t0 = time.time()
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32,
                       temperature=0.8, seed=1000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        d_neg = wm.detect(x, span, MESSAGE)                       # unwatermarked, right key
        y = wm.embed(x, span, MESSAGE, rounds=3, verbose=(i == 0))
        d_pos = wm.detect(y, span, MESSAGE)                       # watermarked, right key
        d_bad = wm_wrong.detect(y, span, MESSAGE)                 # watermarked, wrong key
        changed = float((y[0, span] != x[0, span]).float().mean())
        rows.append(dict(i=i, neg=d_neg["matches"], pos=d_pos["matches"], bad=d_bad["matches"],
                         p_pos=d_pos["p_value"], p_neg=d_neg["p_value"], p_bad=d_bad["p_value"],
                         changed=changed, cost=getattr(wm, "last_cost", 0.0),
                         t=time.time() - t0))
        print(f"[{i:02d}] matches wm {d_pos['matches']}/24 (p={d_pos['p_value']:.2e}) | "
              f"no-wm {d_neg['matches']}/24 | wrong-key {d_bad['matches']}/24 | "
              f"changed {changed:.2f} cost {rows[-1]['cost']:.2f} nats | {rows[-1]['t']:.0f}s",
              flush=True)
        if i == 0:
            P = p.shape[1]
            print("  --- draft ---\n  " + M.tok.decode(x[0, P:], skip_special_tokens=True)[:400])
            print("  --- watermarked ---\n  " + M.tok.decode(y[0, P:], skip_special_tokens=True)[:400], flush=True)

    a = {k: np.array([r[k] for r in rows]) for k in ("neg", "pos", "bad", "changed", "cost")}
    print("\n===== E2E =====")
    print(f"samples {len(rows)}   payload {CFG['n_probes']} bits")
    print(f"matches/24  watermarked {a['pos'].mean():.1f}  no-wm {a['neg'].mean():.1f}  "
          f"wrong-key {a['bad'].mean():.1f}   (chance 12.0)")
    print(f"bit accuracy {a['pos'].mean()/24:.3f}")
    print(f"detected @ p<1e-4: wm {np.mean([r['p_pos']<1e-4 for r in rows]):.2f}  "
          f"no-wm {np.mean([r['p_neg']<1e-4 for r in rows]):.2f}  "
          f"wrong-key {np.mean([r['p_bad']<1e-4 for r in rows]):.2f}")
    print(f"tokens changed {a['changed'].mean():.3f}   substitution cost "
          f"{a['cost'].mean():.2f} nats/edited-pos")
    json.dump(rows, open("/ssd1/ming/basinmark/results/e2e.json", "w"), indent=1)


if __name__ == "__main__":
    main()
