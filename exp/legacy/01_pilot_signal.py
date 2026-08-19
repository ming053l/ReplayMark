"""Go/no-go: is the keyed re-denoising contrast controllable, and is the null symmetric?"""
import sys, time, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel, MASK_ID
from basinmark.prng import probe_pattern

KEY = b"basinmark-pilot"
PROBE_RATE, CTX_RATE = 0.12, 0.20
TAU = 1.5          # nats of per-token quality budget
N_PROBES = 24

QUESTIONS = [
    "Explain why diffusion language models can generate tokens in parallel.",
    "Describe the water cycle in a short paragraph.",
    "What makes a good scientific hypothesis? Answer in a few sentences.",
    "Summarise the causes of the industrial revolution.",
]


def main():
    t0 = time.time()
    M = BasinModel()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    records = []
    for qi, q in enumerate(QUESTIONS):
        p = M.build_prompt(q)
        P, G = p.shape[1], 128
        t = time.time()
        x = M.generate(p, gen_len=G, steps=64, block_len=32, temperature=0.0, seed=qi)
        print(f"[gen {qi}] {time.time()-t:.1f}s :: "
              f"{M.tok.decode(x[0, P:], skip_special_tokens=True)[:110]!r}", flush=True)
        ids = x.cpu()
        ans = np.arange(P, P + G)          # probes live in the answer span only

        for j in range(N_PROBES):
            S, D0, D1 = probe_pattern(KEY, j, G, PROBE_RATE, CTX_RATE)
            S, D0, D1 = ans[S], ans[D0], ans[D1]
            rows = torch.tensor(S)
            batch = torch.cat([
                M.corrupt(ids, np.concatenate([S, D0])),
                M.corrupt(ids, np.concatenate([S, D1])),
                M.corrupt(ids, S),                       # clean base (fluency model)
            ], 0)
            lp = M.logprobs_rows(batch, rows, chunk=3)   # [3, |S|, V]
            y = ids[0, rows]
            l0 = lp[0].gather(1, y[:, None]).squeeze(1)
            l1 = lp[1].gather(1, y[:, None]).squeeze(1)
            delta = (l1 - l0).numpy()

            # controllability: among tokens the *unwatermarked* model rates within TAU
            # nats of its own best, how far can we push g = l1 - l0 ?
            base = lp[2]
            adm = base >= (base.max(1, keepdim=True).values - TAU)
            g = lp[1] - lp[0]
            neg = torch.finfo(g.dtype).min
            gmax = torch.where(adm, g, torch.full_like(g, neg)).max(1).values.numpy()
            gmin = torch.where(adm, g, torch.full_like(g, -neg)).min(1).values.numpy()
            n_adm = adm.sum(1).numpy()

            records.append(dict(
                q=qi, j=j, nS=len(S),
                Delta=float(delta.mean()), sd_delta=float(delta.std()),
                gain_up=float((gmax - delta).mean()), gain_dn=float((delta - gmin).mean()),
                n_adm=float(np.median(n_adm)), frac_free=float((n_adm > 1).mean()),
                H_base=float((-(base.exp() * base).sum(1)).mean()),
            ))
        print(f"  probes done, {time.time()-t:.1f}s", flush=True)

    R = records
    nS = np.mean([r["nS"] for r in R])
    sd = np.mean([r["sd_delta"] for r in R])
    noise = sd / np.sqrt(nS)                       # std of Delta_j under resampling
    gain = np.mean([min(r["gain_up"], r["gain_dn"]) for r in R])
    D = np.array([r["Delta"] for r in R])
    print("\n===== PILOT =====")
    print(f"probes                 {len(R)}   |S| = {nS:.0f}")
    print(f"per-position sd(delta) {sd:.3f} nats")
    print(f"noise sd(Delta_j)      {noise:.3f} nats      <- what we must overcome")
    print(f"achievable |shift|     {gain:.3f} nats/pos   (tau={TAU}, worst direction)")
    print(f"SNR  gain/noise        {gain/noise:.1f}x")
    print(f"admissible tokens      median {np.median([r['n_adm'] for r in R]):.0f}, "
          f"frac positions with >1 choice {np.mean([r['frac_free'] for r in R]):.2f}")
    print(f"null Delta_j: mean {D.mean():+.4f}  sd {D.std():.4f}  "
          f"P(sign>0) {np.mean(D>0):.3f}  (want ~0, ~0.5)")
    json.dump(R, open("/ssd1/ming/basinmark/results/pilot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
