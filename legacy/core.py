"""BasinMark: embed / detect via keyed re-denoising contrast."""
import numpy as np, torch
from .prng import probe_pattern, partition_patterns, payload_bits


class BasinMark:
    def __init__(self, model, key: bytes, n_probes=24, probe_rate=0.25, ctx_rate=0.20,
                 tau=4.0, lam=1.0, topk=64, margin=0.15, disjoint=True):
        self.M, self.key = model, key
        self.n_probes, self.probe_rate, self.ctx_rate = n_probes, probe_rate, ctx_rate
        # tau: hard per-token cap (nats) on how far below the denoiser's own argmax a
        # substitution may fall. lam: weight of the watermark term against fluency.
        # The realised cost is far below tau -- it is reported by embed() as `cost`.
        self.tau, self.lam, self.topk, self.margin = tau, lam, topk, margin
        self.disjoint = disjoint
        self._cache_n, self._cache = -1, None

    def _patterns(self, span):
        n = len(span)
        if self._cache_n != n:
            if self.disjoint:
                pats = partition_patterns(self.key, n, self.n_probes, self.ctx_rate)
            else:
                pats = [probe_pattern(self.key, j, n, self.probe_rate, self.ctx_rate)
                        for j in range(self.n_probes)]
            self._cache_n, self._cache = n, pats
        return [(span[S], span[D0], span[D1]) for S, D0, D1 in self._cache]

    def _pattern(self, j, span):
        return self._patterns(span)[j]

    # ---------------- detection ----------------
    @torch.no_grad()
    def deltas(self, ids, span):
        """Per-probe contrast Delta_j and its t-statistic. 2 forwards per probe."""
        out = []
        for j in range(self.n_probes):
            S, D0, D1 = self._pattern(j, span)
            rows = torch.tensor(S)
            batch = torch.cat([self.M.corrupt(ids, np.concatenate([S, D0])),
                               self.M.corrupt(ids, np.concatenate([S, D1]))], 0)
            lp = self.M.logprobs_rows(batch, rows, chunk=2)
            y = ids[0, rows]
            d = (lp[1].gather(1, y[:, None]) - lp[0].gather(1, y[:, None])).squeeze(1).numpy()
            out.append((d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-9)))
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    def detect(self, ids, span, message=0):
        """Sign-match count against the keyed codeword. Exact Binomial(M, 1/2) null."""
        from scipy.stats import binom
        D, T = self.deltas(ids, span)
        signs = payload_bits(self.key, self.n_probes, message)
        k = int((np.sign(D) == signs).sum())
        return dict(matches=k, n=self.n_probes,
                    p_value=float(binom.sf(k - 1, self.n_probes, 0.5)),
                    bits=(np.sign(D) > 0).astype(int).tolist(),
                    Delta=D.tolist(), t=T.tolist())

    # ---------------- embedding ----------------
    @torch.no_grad()
    def _probe_tables(self, ids, span):
        """For each probe: candidate ids, admissibility, guidance g, base logprobs.

        Every probe position is masked in BOTH arms, so y_i never enters either
        conditioning context -- the guidance g_i(v) = l1_i(v) - l0_i(v) is a table
        lookup obtained from 2 forwards, not |V| lookaheads. (DESIGN.md sec. 2)
        """
        tabs = []
        for j in range(self.n_probes):
            S, D0, D1 = self._pattern(j, span)
            rows = torch.tensor(S)
            batch = torch.cat([self.M.corrupt(ids, np.concatenate([S, D0])),
                               self.M.corrupt(ids, np.concatenate([S, D1])),
                               self.M.corrupt(ids, S)], 0)
            lp = self.M.logprobs_rows(batch, rows, chunk=3)
            base = lp[2]
            cand = base.topk(self.topk, dim=1).indices                  # [|S|, K]
            bsel = base.gather(1, cand)
            adm = bsel >= (base.max(1, keepdim=True).values - self.tau)
            g = (lp[1] - lp[0]).gather(1, cand)
            y = ids[0, rows]
            cur = (lp[1].gather(1, y[:, None]) - lp[0].gather(1, y[:, None])).squeeze(1)
            tabs.append(dict(S=S, cand=cand, adm=adm, g=g, base=bsel, Delta=float(cur.mean())))
        return tabs

    def embed(self, ids, span, message=0, rounds=3, verbose=False):
        """generate -> keyed re-mask -> biased re-denoise (DESIGN.md sec. 3)."""
        ids = ids.clone()
        signs = payload_bits(self.key, self.n_probes, message)
        for r in range(rounds):
            tabs = self._probe_tables(ids, span)
            D = np.array([t["Delta"] for t in tabs])
            ok = (np.sign(D) == signs) & (np.abs(D) > self.margin)
            if verbose:
                print(f"  round {r}: {ok.sum()}/{self.n_probes} bits set, "
                      f"mean|Delta| {np.abs(D).mean():.3f}", flush=True)
            if ok.all():
                break
            # Coordinate ascent over positions. A position can belong to several probe
            # sets, which pull it in different directions; the joint objective resolves
            # the contention instead of letting the last probe overwrite the others.
            score, cands, admit, basel = {}, {}, {}, {}
            for j, t in enumerate(tabs):
                w = 1.0 if not ok[j] else 0.05          # spend effort on unset bits
                for u, i in enumerate(t["S"]):
                    v = t["cand"][u]
                    contrib = w * signs[j] * t["g"][u]
                    if i not in score:
                        score[i] = torch.zeros(self.topk)
                        cands[i], admit[i], basel[i] = v, t["adm"][u].clone(), t["base"][u].clone()
                    else:                                # align onto the first probe's slate
                        pos = _align(v, cands[i])
                        contrib = _scatter(contrib, pos, self.topk)
                        admit[i] &= _scatter_bool(t["adm"][u], pos, self.topk)
                    score[i] += contrib
            cost = []
            for i, sc in score.items():
                a = admit[i]
                if a.sum() == 0:
                    continue
                # Soft guidance: trade fluency against watermark instead of taking the
                # worst admissible token. tau only bounds the tail.
                obj = basel[i] + self.lam * sc
                pick = torch.where(a, obj, torch.full_like(obj, torch.finfo(obj.dtype).min)).argmax()
                cost.append(float(basel[i].max() - basel[i][pick]))
                ids[0, i] = cands[i][pick]
            self.last_cost = float(np.mean(cost)) if cost else 0.0
            if verbose:
                print(f"    mean substitution cost {self.last_cost:.3f} nats/edited-pos",
                      flush=True)
        return ids


def _align(src_ids, dst_ids):
    """index of each src token inside dst slate, -1 if absent."""
    eq = src_ids[:, None] == dst_ids[None, :]
    has = eq.any(1)
    return torch.where(has, eq.float().argmax(1), torch.full_like(has, -1, dtype=torch.long))


def _scatter(vals, pos, k):
    out = torch.zeros(k, dtype=vals.dtype)
    m = pos >= 0
    out[pos[m]] = vals[m]
    return out


def _scatter_bool(vals, pos, k):
    out = torch.ones(k, dtype=torch.bool)     # tokens absent from the slate stay allowed
    m = pos >= 0
    out[pos[m]] = vals[m]
    return out
