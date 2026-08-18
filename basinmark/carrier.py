"""BasinMark-C: carrier-set formulation with an exactly consistent guidance table.

The iterative embedder in core.py suffers from table staleness: g is computed on the
current text, then many tokens are rewritten at once, which changes every probe's
conditioning context and invalidates g. Empirically bit accuracy is non-monotonic in
the guidance weight and collapses below chance at lam=30 (logs_clean/lam.log).

Fix: reserve a keyed *carrier set* P (a fraction of the span) and mask ALL of P in both
arms of every probe. The conditioning context of every arm is then span \\ (P u D_j^b),
which contains no carrier position -- so nothing we write into P can change any arm's
output. The guidance table is exact, fixed, and computed once:

    2*M forwards total, versus 3*M per round with no consistency guarantee.

Carrier tokens are then committed progressively, low-confidence-remasking style, with a
fresh fluency forward each step (which does see previously committed carriers). That
refines quality without ever touching g's validity.

D_j^0 / D_j^1 remain exchangeable, so the exact Binomial(M, 1/2) null is unchanged.
"""
import numpy as np, torch
from .prng import stream, payload_bits
from .model import MASK_ID


def carrier_patterns(key, n, n_probes, carrier_rate, ctx_rate):
    rng = stream(key, "carrier", n, n_probes, carrier_rate)
    perm = rng.permutation(n)
    n_p = max(n_probes, int(round(carrier_rate * n)))
    P, rest = perm[:n_p], perm[n_p:]
    n_d = max(1, int(round(ctx_rate * n)))
    if 2 * n_d > len(rest):
        raise ValueError("ctx_rate too large for the non-carrier region")
    S_list = [np.sort(c) for c in np.array_split(P, n_probes)]
    D = []
    for j in range(n_probes):
        q = stream(key, "ctx2", j, n).permutation(len(rest))
        D.append((np.sort(rest[q[:n_d]]), np.sort(rest[q[n_d:2 * n_d]])))
    return np.sort(P), S_list, D


class CarrierMark:
    def __init__(self, model, key: bytes, n_probes=16, carrier_rate=0.30, ctx_rate=0.20,
                 tau=6.0, lam=3.0, commit_steps=8):
        self.M, self.key = model, key
        self.n_probes, self.carrier_rate, self.ctx_rate = n_probes, carrier_rate, ctx_rate
        self.tau, self.lam, self.commit_steps = tau, lam, commit_steps
        self._cn, self._cp = -1, None

    def _pat(self, span):
        n = len(span)
        if self._cn != n:
            self._cn, self._cp = n, carrier_patterns(
                self.key, n, self.n_probes, self.carrier_rate, self.ctx_rate)
        P, S_list, D = self._cp
        return span[P], [span[S] for S in S_list], [(span[a], span[b]) for a, b in D]

    @torch.no_grad()
    def _arms(self, ids, span):
        """Per-probe (lp0, lp1) at S_j. Independent of the carrier tokens by construction."""
        P, S_list, D = self._pat(span)
        out = []
        for S, (D0, D1) in zip(S_list, D):
            batch = torch.cat([self.M.corrupt(ids, np.concatenate([P, D0])),
                               self.M.corrupt(ids, np.concatenate([P, D1]))], 0)
            out.append(self.M.logprobs_rows(batch, torch.tensor(S), chunk=2))
        return out

    def deltas(self, ids, span):
        P, S_list, _ = self._pat(span)
        D, T = [], []
        for S, lp in zip(S_list, self._arms(ids, span)):
            y = ids[0, torch.tensor(S)]
            d = (lp[1].gather(1, y[:, None]) - lp[0].gather(1, y[:, None])).squeeze(1).numpy()
            D.append(d.mean())
            T.append(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-9))
        return np.array(D), np.array(T)

    def detect(self, ids, span, message=0):
        from scipy.stats import binom
        D, T = self.deltas(ids, span)
        signs = payload_bits(self.key, self.n_probes, message)
        k = int((np.sign(D) == signs).sum())
        return dict(matches=k, n=self.n_probes,
                    p_value=float(binom.sf(k - 1, self.n_probes, 0.5)),
                    Delta=D.tolist(), t=T.tolist())

    @torch.no_grad()
    def embed(self, ids, span, message=0, verbose=False):
        P, S_list, _ = self._pat(span)
        signs = payload_bits(self.key, self.n_probes, message)
        arms = self._arms(ids, span)                       # fixed for the whole embedding

        gvec = {}                                          # position -> signed guidance row
        for j, (S, lp) in enumerate(zip(S_list, arms)):
            g = signs[j] * (lp[1] - lp[0])
            for u, i in enumerate(S):
                gvec[i] = g[u]

        work = self.M.corrupt(ids, P)                      # blank the carrier, then refill
        todo = list(P)
        per = max(1, int(np.ceil(len(P) / self.commit_steps)))
        cost = []
        while todo:
            rows = torch.tensor(todo)
            base = self.M.logprobs_rows(work, rows, chunk=1)[0]     # sees committed carriers
            top = base.max(1, keepdim=True).values
            adm = base >= (top - self.tau)
            obj = base + self.lam * torch.stack([gvec[i] for i in todo])
            neg = torch.finfo(obj.dtype).min
            obj = torch.where(adm, obj, torch.full_like(obj, neg))
            pick = obj.argmax(1)
            # commit the positions the fluency model is most sure about, as in
            # low-confidence remasking, so the joint text stays in-distribution
            conf = base.gather(1, pick[:, None]).squeeze(1)
            order = conf.argsort(descending=True)[:per]
            for o in order.tolist():
                work[0, todo[o]] = pick[o]
                cost.append(float(top[o, 0] - base[o, pick[o]]))
            todo = [p for k, p in enumerate(todo) if k not in set(order.tolist())]
        self.last_cost = float(np.mean(cost)) if cost else 0.0
        if verbose:
            print(f"    carrier {len(P)} positions, mean cost {self.last_cost:.2f} nats",
                  flush=True)
        return work
