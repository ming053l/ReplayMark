"""Why controllability failed, and what fixes it: sweep temperature x probe_rate x tau."""
import sys, time, json, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.prng import probe_pattern

KEY = b"basinmark-pilot"
N_PROBES, CTX_RATE = 24, 0.20
TAUS = [1.5, 3.0, 6.0]
TEMPS = [0.0, 1.0]
RATES = [0.12, 0.25, 0.40]
QS = ["Explain why diffusion language models can generate tokens in parallel.",
      "Describe the water cycle in a short paragraph.",
      "What makes a good scientific hypothesis? Answer in a few sentences.",
      "Summarise the causes of the industrial revolution."]


def main():
    M = BasinModel()
    rows = []
    for temp in TEMPS:
        texts = []
        for qi, q in enumerate(QS):
            p = M.build_prompt(q)
            x = M.generate(p, gen_len=128, steps=64, block_len=32, temperature=temp, seed=qi)
            texts.append((x.cpu(), p.shape[1]))
        for rate in RATES:
            acc = {t: [] for t in TAUS}
            nadm = {t: [] for t in TAUS}
            noise = []
            for ids, P in texts:
                span = np.arange(P, P + 128)
                for j in range(N_PROBES):
                    S, D0, D1 = probe_pattern(KEY, j, 128, rate, CTX_RATE)
                    S, D0, D1 = span[S], span[D0], span[D1]
                    r = torch.tensor(S)
                    batch = torch.cat([M.corrupt(ids, np.concatenate([S, D0])),
                                       M.corrupt(ids, np.concatenate([S, D1])),
                                       M.corrupt(ids, S)], 0)
                    lp = M.logprobs_rows(batch, r, chunk=3)
                    g, base = lp[1] - lp[0], lp[2]
                    y = ids[0, r]
                    d = (lp[1].gather(1, y[:, None]) - lp[0].gather(1, y[:, None])).squeeze(1)
                    noise.append(float(d.std(unbiased=True) / np.sqrt(len(S))))
                    for tau in TAUS:
                        adm = base >= (base.max(1, keepdim=True).values - tau)
                        big = torch.finfo(g.dtype).max
                        Ap = torch.where(adm, g, torch.full_like(g, -big)).max(1).values.mean()
                        An = torch.where(adm, g, torch.full_like(g, big)).min(1).values.mean()
                        acc[tau].append((float(Ap), float(An)))
                        nadm[tau].append(float((adm.sum(1) > 1).float().mean()))
            for tau in TAUS:
                a = np.array(acc[tau])
                rows.append(dict(temp=temp, rate=rate, tau=tau,
                                 set_pos=float((a[:, 0] > 0).mean()),
                                 set_neg=float((a[:, 1] < 0).mean()),
                                 both=float(((a[:, 0] > 0) & (a[:, 1] < 0)).mean()),
                                 Ap=float(a[:, 0].mean()), An=float(a[:, 1].mean()),
                                 free=float(np.mean(nadm[tau])), noise=float(np.mean(noise))))
                print(f"T={temp} rate={rate:.2f} tau={tau:<4} | settable {rows[-1]['both']:.2f} "
                      f"(+{rows[-1]['set_pos']:.2f}/-{rows[-1]['set_neg']:.2f}) "
                      f"A+={rows[-1]['Ap']:+.3f} A-={rows[-1]['An']:+.3f} "
                      f"free={rows[-1]['free']:.2f} noise={rows[-1]['noise']:.3f}", flush=True)
    json.dump(rows, open("/ssd1/ming/basinmark/results/sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
