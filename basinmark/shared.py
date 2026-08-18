"""Detection in L forwards, independent of payload size.

CarrierMark runs 2 forwards per (probe, ablation) pair -- 2*M*R = 96 forwards of an 8B
model per detection, against zero model calls for every baseline (dgMARK checks token
parity; KGW/eth-sri/KTH/Unigram/AAR hash the context). That gap is the method's main
structural cost.

It is also unnecessary. Every arm masks the *whole* carrier set P, so a single forward on
mask(P u D) already returns log-probs at every carrier position -- i.e. for all M probe
sets at once. So draw L *shared* ablation patterns, run L forwards, and let the key
decide which ordered pair of patterns each (probe, repetition) block contrasts.

    detection cost: L forwards, independent of M and R.

The exact null survives. Each block carries an independent keyed orientation bit
eps_{j,r}; flipping it negates that block's contrast and nothing else, so conditional on
the magnitudes the signs are independent Rademacher -- the same exact sign-flip test.
Blocks are correlated through the shared patterns, but the conditional test never
required independent magnitudes.
"""
import numpy as np, torch
from .prng import stream, payload_bits
from .carrier import signflip_pvalue


def shared_patterns(key, n, n_probes, carrier_rate, ctx_rate, n_patterns, n_ablations,
                    pool_rate=None):
    """With pool_rate set, P is not returned -- a keyed POOL Q is, and the carrier is
    chosen inside it by entropy at detection/embedding time (see SharedMark._carrier).

    Key-blind carrier positions waste payload: wherever the denoiser is certain there is
    no admissible alternative, so the position contributes noise and no signal. Observed
    as extreme sample-to-sample variance -- 3.9 % of tokens moved and z=+4.1 on one C4
    continuation, 0.8 % and z=+1.4 on the next.

    The entropy gate is exactly reproducible by the detector: entropy is read from a
    forward with the WHOLE pool masked, whose conditioning context contains no pool
    position, so it cannot be changed by anything the embedder writes.
    """
    rng = stream(key, "shared", n, n_probes, carrier_rate)
    perm = rng.permutation(n)
    n_p = max(n_probes, int(round((pool_rate or carrier_rate) * n)))
    P, rest = perm[:n_p], perm[n_p:]
    n_d = max(1, int(round(ctx_rate * n)))
    if n_d > len(rest):
        raise ValueError("ctx_rate too large for the non-carrier region")
    S_list = None if pool_rate else [np.sort(c) for c in np.array_split(P, n_probes)]
    pats = [np.sort(stream(key, "pat", l, n).choice(rest, n_d, replace=False))
            for l in range(n_patterns)]
    pairs = []                                   # [M][R] -> (u, v) ordered pattern indices
    for j in range(n_probes):
        r_ = stream(key, "pair", j, n)
        row = []
        for r in range(n_ablations):
            u, v = r_.choice(n_patterns, 2, replace=False)
            if r_.integers(0, 2):                # independent orientation bit per block
                u, v = v, u
            row.append((int(u), int(v)))
        pairs.append(row)
    return np.sort(P), S_list, pats, pairs


class SharedMark:
    def __init__(self, model, key: bytes, n_probes=16, carrier_rate=0.30, ctx_rate=0.20,
                 tau=6.0, lam=8.0, commit_steps=4, n_patterns=8, n_ablations=3,
                 pool_rate=None):
        self.M, self.key = model, key
        self.n_probes, self.carrier_rate, self.ctx_rate = n_probes, carrier_rate, ctx_rate
        self.tau, self.lam, self.commit_steps = tau, lam, commit_steps
        self.n_patterns, self.n_ablations = n_patterns, n_ablations
        self.pool_rate = pool_rate
        self._cn, self._cp = -1, None
        self._gate = {}

    def _pat(self, span):
        n = len(span)
        if self._cn != n:
            self._cn, self._cp = n, shared_patterns(
                self.key, n, self.n_probes, self.carrier_rate, self.ctx_rate,
                self.n_patterns, self.n_ablations, self.pool_rate)
        P, S_list, pats, pairs = self._cp
        return (span[P], None if S_list is None else [span[S] for S in S_list],
                [span[p] for p in pats], pairs)

    @torch.no_grad()
    def _carrier(self, ids, span):
        """(carrier positions, probe sets). Without a pool these are purely keyed; with
        one, the highest-entropy positions of the pool are kept -- reproducibly, since
        the entropy forward masks the entire pool."""
        Q, S_list, _, _ = self._pat(span)
        if S_list is not None:
            return Q, S_list
        # The selection depends only on tokens OUTSIDE the pool, which the embedder never
        # writes to -- so cache on exactly those, and the same forward is never repeated
        # between _carrier, _logp and _grid.
        ctx = self.M.corrupt(ids, Q)
        ck = hash(ctx.numpy().tobytes())
        if ck in self._gate:
            return self._gate[ck]
        base = self.M.logprobs_rows(ctx, torch.tensor(Q), chunk=1)[0]
        H = -(base.exp() * base).sum(1).numpy()
        n_c = max(self.n_probes, int(round(self.carrier_rate * len(span))))
        keep = np.sort(Q[np.argsort(-H)[:n_c]])
        order = stream(self.key, "split", len(span)).permutation(len(keep))
        out = keep, [np.sort(keep[c]) for c in np.array_split(order, self.n_probes)]
        self._gate[ck] = out
        return out

    @torch.no_grad()
    def _logp(self, ids, span):
        """L forwards -> log-probs at every carrier position under each shared pattern."""
        P, _ = self._carrier(ids, span)
        _, _, pats, _ = self._pat(span)
        batch = torch.cat([self.M.corrupt(ids, np.concatenate([P, d])) for d in pats], 0)
        return self.M.logprobs_rows(batch, torch.tensor(P), chunk=2)   # [L, |P|, V]

    def _grid(self, ids, span, lp=None):
        P, S_list = self._carrier(ids, span)
        _, _, _, pairs = self._pat(span)
        lp = self._logp(ids, span) if lp is None else lp
        idx = {int(p): k for k, p in enumerate(P)}
        y = ids[0, torch.tensor(P)]
        at = lp.gather(2, y[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(2)  # [L,|P|]
        return np.array([[float((at[v, [idx[int(i)] for i in S]]
                                - at[u, [idx[int(i)] for i in S]]).mean())
                          for u, v in pairs[j]] for j, S in enumerate(S_list)])

    def detect(self, ids, span, message=0):
        from scipy.stats import binom
        grid = self._grid(ids, span)
        signs = payload_bits(self.key, self.n_probes, message)
        a = (signs[:, None] * grid).ravel()
        D = grid.mean(1)
        k = int((np.sign(D) == signs).sum())
        return dict(matches=k, n=self.n_probes, n_blocks=int(a.size),
                    n_forwards=self.n_patterns,
                    p_sign=float(binom.sf(k - 1, self.n_probes, 0.5)),
                    p_value=signflip_pvalue(a),
                    z=float(a.sum() / (np.sqrt((a ** 2).sum()) + 1e-12)),
                    Delta=D.tolist())

    @torch.no_grad()
    def embed(self, ids, span, message=0):
        P, S_list = self._carrier(ids, span)
        _, _, pats, pairs = self._pat(span)
        signs = payload_bits(self.key, self.n_probes, message)
        lp = self._logp(ids, span)                      # fixed: no carrier token enters it
        idx = {int(p): k for k, p in enumerate(P)}
        gvec = {}
        for j, S in enumerate(S_list):
            for i in S:
                row = idx[int(i)]
                g = torch.stack([lp[v, row] - lp[u, row] for u, v in pairs[j]]).mean(0)
                gvec[int(i)] = signs[j] * g

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
