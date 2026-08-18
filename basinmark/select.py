"""Probe-conditioned carrier selection.

Entropy says a position is *editable*. What the embedder needs is a position where a
large guidance swing is available cheaply. Those differ, and perplexity said so: the
entropy gate bought z 2.90 -> 3.78 while taking GPT-2-large perplexity 41 -> 107.

Two things this file gets right that an earlier attempt did not.

**The selector must score the guidance the embedder actually uses.** The embedder pushes
along the R-pair *average*

    g_{j,i}(v) = (1/R) sum_r [ l_{v_jr,i}(v) - l_{u_jr,i}(v) ]

so scoring the mean of each pair's individual range is the wrong objective: a position
where pair 1 favours token A by +5 and pair 2 favours token B by +5 has a large range
under both pairs and almost no swing in their average. Selection therefore happens
*after* the pool is partitioned into probes (a keyed, text-independent split), so every
candidate already knows which probe it would serve and its true aggregate guidance can
be evaluated. Sign-free, since orientation swaps leave a range invariant and the
randomization null must stay clean:

    U_{j,i} = ( max_{v in A_i} g_{j,i}(v) - min_{v in A_i} g_{j,i}(v) ) / 2
    C_{j,i} = mean fluency cost of reaching those two extremes
    W_{j,i} = U_{j,i} / (C_{j,i} + eps)

**All three selectors must draw from the same candidate structure.** `none` is a keyed
random pick *within each probe's share of the pool* -- not the first positions of a
position-sorted array, which is what an earlier version silently did, making the
keyed-only baseline a left-of-text carrier and the comparison meaningless.

The arms mask the whole pool rather than the selected carrier, so guidance exists at
every candidate before selection and the selection stays reproducible at detection.
"""
import hashlib, hmac
import numpy as np, torch
from .prng import stream, payload_bits
from .carrier import signflip_pvalue

EPS = 0.25          # nats


def pool_patterns(key, n, n_probes, pool_rate, ctx_rate, n_patterns, n_ablations):
    rng = stream(key, "pool", n, n_probes, pool_rate)
    perm = rng.permutation(n)
    n_q = max(n_probes, int(round(pool_rate * n)))
    Q = np.sort(perm[:n_q])
    rest = perm[n_q:]
    n_d = max(1, int(round(ctx_rate * n)))
    if n_d > len(rest):
        raise ValueError("pool_rate + ctx_rate too large")
    # keyed, text-independent partition of the pool into probe shares
    share = stream(key, "share", n, n_probes).permutation(len(Q))
    parts = [np.sort(c) for c in np.array_split(share, n_probes)]
    pats = [np.sort(stream(key, "pat", l, n).choice(rest, n_d, replace=False))
            for l in range(n_patterns)]
    pairs = []
    for j in range(n_probes):
        r_ = stream(key, "pair", j, n)
        row = []
        for r in range(n_ablations):
            u, v = r_.choice(n_patterns, 2, replace=False)
            if r_.integers(0, 2):
                u, v = v, u
            row.append((int(u), int(v)))
        pairs.append(row)
    return Q, parts, pats, pairs


class LeverageMark:
    def __init__(self, model, key: bytes, n_probes=16, pool_rate=0.50, carrier_rate=0.30,
                 ctx_rate=0.15, tau=2.0, lam=8.0, commit_steps=2, n_patterns=8,
                 n_ablations=6, select="leverage", topk=256, nonce=None):
        self.M = model
        self.base_key, self.nonce = key, nonce
        self.key = key if nonce is None else hmac.new(
            key, nonce if isinstance(nonce, bytes) else str(nonce).encode(),
            hashlib.sha256).digest()
        self.n_probes, self.pool_rate, self.carrier_rate = n_probes, pool_rate, carrier_rate
        self.ctx_rate, self.tau, self.lam = ctx_rate, tau, lam
        self.commit_steps = commit_steps
        self.n_patterns, self.n_ablations = n_patterns, n_ablations
        self.select, self.topk = select, topk
        self._cn, self._cp, self._cache = -1, None, {}
        self.adm_stats = None

    def _pat(self, span):
        n = len(span)
        if self._cn != n:
            self._cn, self._cp = n, pool_patterns(
                self.key, n, self.n_probes, self.pool_rate, self.ctx_rate,
                self.n_patterns, self.n_ablations)
        Q, parts, pats, pairs = self._cp
        return span[Q], parts, [span[p] for p in pats], pairs

    @torch.no_grad()
    def _prepare(self, ids, span):
        """(probe sets, guidance rows, index map). Depends only on tokens OUTSIDE the
        pool, which the embedder never writes -- identical before and after embedding."""
        Q, parts, pats, pairs = self._pat(span)
        ctx = self.M.corrupt(ids, Q)
        ck = hash(ctx.numpy().tobytes())
        if ck in self._cache:
            self.adm_stats = self._cache[ck][-1]
            return self._cache[ck]
        rows = torch.tensor(Q)
        arms = self.M.logprobs_rows(
            torch.cat([self.M.corrupt(ids, np.concatenate([Q, d])) for d in pats], 0),
            rows, chunk=2)                                        # [L, |Q|, V]
        base = self.M.logprobs_rows(ctx, rows, chunk=1)[0]        # [|Q|, V]

        cand = base.topk(self.topk, dim=1).indices
        bsel = base.gather(1, cand)
        top = bsel.max(1, keepdim=True).values
        adm = bsel >= (top - self.tau)
        n_adm = adm.sum(1).numpy()
        stats = dict(median=float(np.median(n_adm)), p90=float(np.quantile(n_adm, 0.9)),
                     max=int(n_adm.max()), frac_capped=float((n_adm >= self.topk).mean()))

        armk = torch.stack([a.gather(1, cand) for a in arms])     # [L, |Q|, K]
        big = torch.finfo(armk.dtype).max
        n_per = max(1, int(round(self.carrier_rate * len(span))) // self.n_probes)
        rnd = stream(self.key, "sel-none", len(span))
        H = -(base.exp() * base).sum(1).numpy()

        S_list, keep_all = [], []
        for j, part in enumerate(parts):
            if self.select == "none":
                score = rnd.random(len(part))
            elif self.select == "entropy":
                score = H[part]
            else:
                # the aggregate guidance this probe will actually push along
                g = torch.stack([armk[v, part] - armk[u, part]
                                 for u, v in pairs[j]]).mean(0)   # [|part|, K]
                a = adm[part]
                hi = torch.where(a, g, torch.full_like(g, -big))
                lo = torch.where(a, g, torch.full_like(g, big))
                ih, il = hi.argmax(1), lo.argmin(1)
                U = 0.5 * (g.gather(1, ih[:, None]) - g.gather(1, il[:, None])).squeeze(1)
                t, b = top[part].squeeze(1), bsel[part]
                C = 0.5 * ((t - b.gather(1, ih[:, None]).squeeze(1))
                           + (t - b.gather(1, il[:, None]).squeeze(1)))
                score = (U / (C + EPS)).numpy()
            keep = part[np.argsort(-score)[:n_per]]
            S_list.append(np.sort(Q[keep]))
            keep_all.append(keep)
        P = np.sort(np.concatenate([Q[k] for k in keep_all]))
        pos_of = {int(p): k for k, p in enumerate(Q)}
        out = (P, S_list, arms, pos_of, pairs, stats)
        self._cache[ck] = out
        self.adm_stats = stats
        return out

    def _grid(self, ids, span):
        P, S_list, arms, pos_of, pairs, _ = self._prepare(ids, span)
        return np.array([[float(np.mean([
            float(arms[v][pos_of[int(i)], int(ids[0, i])]
                  - arms[u][pos_of[int(i)], int(ids[0, i])]) for i in S]))
            for u, v in pairs[j]] for j, S in enumerate(S_list)])

    def detect(self, ids, span, message=0):
        grid = self._grid(ids, span)
        signs = payload_bits(self.key, self.n_probes, message)
        a = (signs[:, None] * grid).ravel()
        r = signflip_pvalue(a)
        D = grid.mean(1)
        return dict(matches=int((np.sign(D) == signs).sum()), n=self.n_probes,
                    n_blocks=int(a.size), n_forwards=self.n_patterns + 1,
                    p_value=r["p_exact"] if r["p_exact"] is not None else r["p_bound"],
                    p_bound=r["p_bound"], p_mc=r["p_mc"], z=r["z"], Delta=D.tolist())

    @torch.no_grad()
    def embed(self, ids, span, message=0):
        P, S_list, arms, pos_of, pairs, _ = self._prepare(ids, span)
        signs = payload_bits(self.key, self.n_probes, message)
        gvec = {}
        for j, S in enumerate(S_list):
            for i in S:
                row = pos_of[int(i)]
                gvec[int(i)] = signs[j] * torch.stack(
                    [arms[v][row] - arms[u][row] for u, v in pairs[j]]).mean(0)

        work = self.M.corrupt(ids, P)
        todo, cost = list(P), []
        per = max(1, int(np.ceil(len(P) / self.commit_steps)))
        while todo:
            base = self.M.logprobs_rows(work, torch.tensor(todo), chunk=1)[0]
            top = base.max(1, keepdim=True).values
            obj = torch.where(base >= (top - self.tau),
                              base + self.lam * torch.stack([gvec[int(i)] for i in todo]),
                              torch.full_like(base, torch.finfo(base.dtype).min))
            pick = obj.argmax(1)
            order = top.squeeze(1).argsort(descending=True)[:per]
            for o in order.tolist():
                work[0, todo[o]] = pick[o]
                cost.append(float(top[o, 0] - base[o, pick[o]]))
            drop = set(order.tolist())
            todo = [p for k, p in enumerate(todo) if k not in drop]
        self.last_cost = float(np.mean(cost)) if cost else 0.0
        return work
