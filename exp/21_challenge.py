"""How much signal do random challenge pairs waste?

g_i(v) = log p(v | C_1) - log p(v | C_0) is a context log-likelihood contrast. What
matters for embedding is its dynamic range inside the fluency-preserving candidate set,

    G_i(tau; u,v) = max_{w in A_i(tau)} g - min_{w in A_i(tau)} g

If the best of the L(L-1)/2 available pattern pairs has a far larger range than the
average pair, then drawing challenges at random throws away most of the available signal,
and selecting the pair by content is worth doing. Validity survives that: the exact
sign-flip test only requires the ORIENTATION bit to be a fair keyed coin, not the
unordered pair to be content-independent -- provided the selection is invariant to
swapping the two members, never looks at the payload sign, and is reproducible by the
detector.
"""
import sys, gzip, json
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.select import LeverageMark

KEY = b"basinmark-key-A"
GEN, PREFIX, NS = 256, 40, 6
TAUS = [1.0, 2.0, 3.0, 6.0]
SHAPE = dict(n_probes=16, pool_rate=0.50, carrier_rate=0.30, ctx_rate=0.15,
             n_patterns=8, n_ablations=6, select="leverage")


def c4(tok, n, ntok):
    out = []
    with gzip.open("/ssd1/ming/basinmark/data/c4-validation.json.gz", "rt") as f:
        for line in f:
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= ntok + 60:
                out.append(torch.tensor(ids[:ntok], dtype=torch.long)[None])
                if len(out) == n:
                    return out


def main():
    M = BasinModel()
    res = {t: {"mean": [], "max": [], "cost": []} for t in TAUS}
    for i, p in enumerate(c4(M.tok, NS, PREFIX)):
        x = M.generate(p, gen_len=GEN, steps=GEN // 2, block_len=32, temperature=0.8,
                       seed=3000 + i).cpu()
        span = np.arange(p.shape[1], p.shape[1] + GEN)
        w = LeverageMark(M, KEY, tau=6.0, lam=0.0, nonce=f"doc-{i}", **SHAPE)
        Q, _, pats, _ = w._pat(span)
        rows = torch.tensor(Q)
        arms = M.logprobs_rows(
            torch.cat([M.corrupt(x, np.concatenate([Q, d])) for d in pats], 0),
            rows, chunk=2)
        base = M.logprobs_rows(M.corrupt(x, Q), rows, chunk=1)[0]
        cand = base.topk(256, dim=1).indices
        bsel = base.gather(1, cand)
        top = bsel.max(1, keepdim=True).values
        armk = torch.stack([a.gather(1, cand) for a in arms])
        L = len(arms)
        for t in TAUS:
            adm = bsel >= (top - t)
            big = torch.finfo(armk.dtype).max
            G = []
            for u in range(L):
                for v in range(u + 1, L):
                    d = armk[v] - armk[u]
                    hi = torch.where(adm, d, torch.full_like(d, -big)).max(1).values
                    lo = torch.where(adm, d, torch.full_like(d, big)).min(1).values
                    G.append((hi - lo).numpy())
            G = np.stack(G)                       # [pairs, |Q|]
            res[t]["mean"].append(float(G.mean()))
            res[t]["max"].append(float(G.max(0).mean()))
        print(f"[{i}] done", flush=True)

    print("\n===== CHALLENGE-PAIR HEADROOM (dynamic range of g inside A_i(tau)) =====")
    print(f"{'tau':<6}{'mean pair':>12}{'best pair':>12}{'ratio':>9}")
    for t in TAUS:
        m, x_ = np.mean(res[t]["mean"]), np.mean(res[t]["max"])
        print(f"{t:<6}{m:>12.3f}{x_:>12.3f}{x_/m:>9.2f}")
    print("\nratio >> 1 means random challenges waste most of the available signal")
    json.dump({str(k): v for k, v in res.items()},
              open("/ssd1/ming/basinmark/results/challenge.json", "w"), indent=1)


if __name__ == "__main__":
    main()
