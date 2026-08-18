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


def carrier_patterns(key, n, n_probes, carrier_rate, ctx_rate, n_ablations=1):
    """P (carrier), the M probe sets partitioning it, and R ablation PAIRS per probe.

    Each pair (D_j^{0,r}, D_j^{1,r}) is an independent exchangeable draw, so swapping
    within any one pair negates only that contrast. The null therefore has M*R
    independent symmetric blocks rather than M -- more detection power at the same
    carrier size, and sqrt(R) less noise per bit.
    """
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
        pairs = []
        for r in range(n_ablations):
            q = stream(key, "ctx2", j, r, n).permutation(len(rest))
            pairs.append((np.sort(rest[q[:n_d]]), np.sort(rest[q[n_d:2 * n_d]])))
        D.append(pairs)
    return np.sort(P), S_list, D


class CarrierMark:
    def __init__(self, model, key: bytes, n_probes=16, carrier_rate=0.30, ctx_rate=0.20,
                 tau=6.0, lam=3.0, commit_steps=8, n_ablations=3):
        self.M, self.key = model, key
        self.n_probes, self.carrier_rate, self.ctx_rate = n_probes, carrier_rate, ctx_rate
        self.tau, self.lam, self.commit_steps = tau, lam, commit_steps
        self.n_ablations = n_ablations
        self._cn, self._cp = -1, None

    def _pat(self, span):
        n = len(span)
        if self._cn != n:
            self._cn, self._cp = n, carrier_patterns(
                self.key, n, self.n_probes, self.carrier_rate, self.ctx_rate,
                self.n_ablations)
        P, S_list, D = self._cp
        return (span[P], [span[S] for S in S_list],
                [[(span[a], span[b]) for a, b in pairs] for pairs in D])

    @torch.no_grad()
    def _arms(self, ids, span):
        """[probe][ablation] -> (lp0, lp1) at S_j. By construction these never depend on
        the carrier tokens, so they are fixed for the whole embedding."""
        P, S_list, D = self._pat(span)
        out = []
        for S, pairs in zip(S_list, D):
            batch = torch.cat([self.M.corrupt(ids, np.concatenate([P, d]))
                               for D0, D1 in pairs for d in (D0, D1)], 0)
            lp = self.M.logprobs_rows(batch, torch.tensor(S), chunk=2)
            out.append([(lp[2 * r], lp[2 * r + 1]) for r in range(len(pairs))])
        return out

    def deltas(self, ids, span, per_ablation=False):
        """Delta_j (mean over ablations). With per_ablation, also the M x R raw grid,
        whose signs are independently symmetric under H0."""
        _, S_list, _ = self._pat(span)
        grid = []
        for S, pairs in zip(S_list, self._arms(ids, span)):
            y = ids[0, torch.tensor(S)]
            grid.append([float((lp1.gather(1, y[:, None]) - lp0.gather(1, y[:, None])).mean())
                         for lp0, lp1 in pairs])
        grid = np.array(grid)                                   # [M, R]
        D = grid.mean(1)
        T = D / (grid.std(1, ddof=1) / np.sqrt(grid.shape[1]) + 1e-9) if grid.shape[1] > 1 \
            else np.zeros_like(D)
        return (D, T, grid) if per_ablation else (D, T)

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
        for j, (S, pairs) in enumerate(zip(S_list, arms)):
            g = signs[j] * torch.stack([lp1 - lp0 for lp0, lp1 in pairs]).mean(0)
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
            # Order by the model's INTRINSIC certainty at the position (top of base),
            # not by the log-prob of the chosen token. Using the chosen token would
            # systematically defer every watermark-driven pick to the last steps, where
            # the context is richest and the admissible set collapses to a singleton --
            # a feedback loop that starves exactly the positions meant to carry payload.
            order = top.squeeze(1).argsort(descending=True)[:per]
            for o in order.tolist():
                work[0, todo[o]] = pick[o]
                cost.append(float(top[o, 0] - base[o, pick[o]]))

            todo = [p for k, p in enumerate(todo) if k not in set(order.tolist())]
        self.last_cost = float(np.mean(cost)) if cost else 0.0
        if verbose:
            print(f"    carrier {len(P)} positions, mean cost {self.last_cost:.2f} nats",
                  flush=True)
        return work


def signflip_pvalue(a: np.ndarray):
    """Exact conditional p-value for T = sum_j a_j, where a_j = s_j * Delta_j.

    Counting signs throws away the magnitude of each Delta_j. Exchangeability of
    D_j^0/D_j^1 makes every a_j symmetric about 0 *independently*, so conditional on the
    magnitudes |a_j| the null is a Rademacher mixture: enumerate all 2^M sign patterns
    for an exact one-sided p-value. Same assumption as the sign test, strictly more power.
    """
    m = np.abs(a)
    M = len(m)
    T = float(a.sum())
    if M <= 20:
        signs = ((np.arange(1 << M)[:, None] >> np.arange(M)) & 1) * 2 - 1
        return float((signs @ m >= T).mean())
    from scipy.stats import norm
    return float(norm.sf(T / (np.sqrt((m ** 2).sum()) + 1e-12)))


def _detect_full(self, ids, span, message=0):
    from scipy.stats import binom
    D, T, grid = self.deltas(ids, span, per_ablation=True)
    signs = payload_bits(self.key, self.n_probes, message)
    a = (signs[:, None] * grid).ravel()               # M*R independent symmetric blocks
    k = int((np.sign(D) == signs).sum())
    return dict(matches=k, n=self.n_probes, n_blocks=int(a.size),
                p_sign=float(binom.sf(k - 1, self.n_probes, 0.5)),
                p_value=signflip_pvalue(a),           # the reported p-value
                z=float(a.sum() / (np.sqrt((a ** 2).sum()) + 1e-12)),
                stat=float(a.sum()), Delta=D.tolist(), t=T.tolist())


CarrierMark.detect = _detect_full
