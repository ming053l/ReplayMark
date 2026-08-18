"""Block-local BasinMark: commit locally, verify globally.

Two measured failures motivate this file.

*Post-hoc substitution is priced out.* The median carrier position admits exactly one
token inside the fluency budget, so a watermark that must place specific tokens has almost
nothing to spend (`exp/16_pareto.py`, every row prints `adm med 1`).

*Global two-phase generation is worse, not better.* Deferring the whole carrier to a second
phase costs x1.98 perplexity **before any watermark** -- more than dgMARK's entire cost
(`exp/20_gentime.py`). That schedule was not a design choice; it was forced by needing every
non-carrier token final before the guidance table could be computed.

The fix comes from how LLaDA actually decodes: blocks are autoregressive, and only *within*
a block is the process diffusion. So make the watermark block-local.

For block B_b, with B_<b already final and B_b entirely masked, build the challenge table
once. Because the arms mask **all** of B_b, nothing committed inside the block can enter
their conditioning, so the table is exact for the whole block -- the same isolation
argument as the carrier construction, but now costing one block instead of half the
sequence.

Then decode the block normally and **never change a candidate token**. The watermark rides
on commit *order* only, which is what makes dgMARK nearly free:

    s_j * g_{b,i}(x_hat_i) > 0  ->  commit this position first
    otherwise                  ->  defer it; with more context the model may itself
                                   propose a token that is compatible

The compatibility test is not a hash of the token identity but whether the model's own
preferred token already points the right way under a keyed re-denoising challenge, so the
observable stays BasinMark's.

Commit safety follows C4's CCTC (arXiv C4, Eq. 8): eligibility by confidence, then only the
*frontier-anchored prefix* of the eligible set -- keeping the eligible run that survives at
most H still-masked-but-ineligible positions to its left -- because a confident token whose
left context is unresolved is not safe to freeze. Withheld positions are confirmed one step
later rather than dropped, and a scheduled top-k branch guarantees the block still finishes
on time.

Detection needs no trajectory: for each block take B_<b from the final text, mask B_b,
re-apply the keyed challenges, and read g at the observed tokens. Local commitment, global
verification.
"""
import numpy as np, torch
import torch.nn.functional as F
from .model import MASK_ID
from .prng import stream, payload_bits
from .carrier import signflip_pvalue


def block_patterns(key, prompt_len, block_lo, block_len, probes, n_patterns,
                   n_ablations, ctx_frac=0.20, min_ctx=4):
    """Challenge structure for one block.

    Ablation patterns are drawn from the already-final region (prompt + earlier blocks),
    which is exactly the region the arms condition on. Early blocks therefore have a
    smaller pool to challenge with -- for block 0 it is the prompt alone.
    """
    B = np.arange(block_lo, block_lo + block_len)
    region = np.arange(0, block_lo)                       # prompt + finished blocks
    n_d = max(min_ctx, int(round(ctx_frac * len(region))))
    n_d = min(n_d, max(1, len(region) - 1))
    rng = stream(key, "blk", block_lo, block_len, probes)
    order = rng.permutation(block_len)
    S_list = [np.sort(B[c]) for c in np.array_split(order, probes)]
    pats = [np.sort(stream(key, "blkpat", block_lo, l).choice(region, n_d, replace=False))
            for l in range(n_patterns)]
    pairs = []
    for j in range(probes):
        r_ = stream(key, "blkpair", block_lo, j)
        row = []
        for r in range(n_ablations):
            u, v = r_.choice(n_patterns, 2, replace=False)
            if r_.integers(0, 2):                         # independent orientation bit
                u, v = v, u
            row.append((int(u), int(v)))
        pairs.append(row)
    return B, S_list, pats, pairs


class BlockMark:
    def __init__(self, model, key: bytes, block_len=32, probes_per_block=2,
                 n_patterns=6, n_ablations=3, ctx_frac=0.20, tau_conf=0.9, holes=2,
                 nonce=None):
        import hashlib, hmac
        self.M = model
        self.key = key if nonce is None else hmac.new(
            key, str(nonce).encode(), hashlib.sha256).digest()
        self.block_len, self.probes_per_block = block_len, probes_per_block
        self.n_patterns, self.n_ablations = n_patterns, n_ablations
        self.ctx_frac, self.tau_conf, self.holes = ctx_frac, tau_conf, holes

    # ---------- challenge table for one block ----------
    @torch.no_grad()
    def _arms(self, x, prompt_len, lo):
        """L forwards, block fully masked. Independent of anything committed inside it."""
        B, S_list, pats, pairs = block_patterns(
            self.key, prompt_len, lo, self.block_len, self.probes_per_block,
            self.n_patterns, self.n_ablations, self.ctx_frac)
        base = x.clone()
        base[0, B] = MASK_ID
        batch = torch.cat([self._mask(base, d) for d in pats], 0)
        lp = self.M.logprobs_rows(batch, torch.tensor(B), chunk=2)      # [L, |B|, V]
        return B, S_list, pairs, lp

    @staticmethod
    def _mask(ids, pos):
        out = ids.clone()
        out[0, torch.tensor(pos)] = MASK_ID
        return out

    def _guidance(self, B, S_list, pairs, lp, signs_off):
        """position -> signed guidance row, and position -> probe index."""
        loc = {int(p): k for k, p in enumerate(B)}
        g, owner = {}, {}
        for j, S in enumerate(S_list):
            s = signs_off[j]
            for i in S:
                r = loc[int(i)]
                g[int(i)] = s * torch.stack(
                    [lp[v, r] - lp[u, r] for u, v in pairs[j]]).mean(0)
                owner[int(i)] = j
        return g, owner

    # ---------- generation ----------
    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=256, steps=128, temperature=0.8, message=0,
                 seed=0):
        gen = torch.Generator(device=self.M.device).manual_seed(seed)
        Pn = prompt_ids.shape[1]
        n_blocks = max(1, gen_len // self.block_len)
        n_probes = n_blocks * self.probes_per_block
        signs = payload_bits(self.key, n_probes, message)
        steps_pb = max(1, steps // n_blocks)

        x = torch.full((1, Pn + gen_len), MASK_ID, dtype=torch.long, device=self.M.device)
        x[:, :Pn] = prompt_ids.to(self.M.device)
        self.stats = dict(deferred=0, compatible=0, committed=0, fallback=0)

        for b in range(n_blocks):
            lo = Pn + b * self.block_len
            B, S_list, pairs, lp = self._arms(x.cpu(), Pn, lo)
            off = b * self.probes_per_block
            g, owner = self._guidance(B, S_list, pairs, lp,
                                      signs[off:off + self.probes_per_block])
            Bt = torch.tensor(B, device=x.device)
            pending = {}                                   # deferred position -> token
            for t in range(steps_pb):
                live = Bt[x[0, Bt] == MASK_ID]
                if live.numel() == 0:
                    break
                k = int(np.ceil(live.numel() / (steps_pb - t)))
                logits = self.M.model(x).logits[0]
                if temperature > 0:
                    u = torch.rand(logits.shape, device=logits.device,
                                   dtype=torch.float64, generator=gen)
                    xh = (logits.double() / temperature
                          - torch.log(-torch.log(u))).argmax(-1)
                else:
                    xh = logits.argmax(-1)
                conf = F.softmax(logits.double(), -1).gather(
                    -1, xh[:, None]).squeeze(1)
                del logits

                liv = live.tolist()
                cand = xh[live].tolist()
                cf = conf[live].tolist()
                # watermark compatibility of the model's OWN choice -- no token is changed
                w = [float(g[int(i)][int(v)]) for i, v in zip(liv, cand)]

                # CCTC eligibility, then the frontier-anchored prefix
                elig = [n for n, c in enumerate(cf) if c >= self.tau_conf]
                eset = set(liv[n] for n in elig)
                safe = []
                for n, i in enumerate(liv):
                    if n not in elig:
                        continue
                    h = sum(1 for jj in liv if jj < i and jj not in eset)
                    if h <= self.holes:
                        safe.append(n)
                # confirmed deferrals: still predicting the same token one step later
                conf_ok = [n for n, i in enumerate(liv)
                           if pending.get(i) is not None and pending[i] == cand[n]]

                pool = sorted(set(safe) | set(conf_ok))
                compat = [n for n in pool if w[n] > 0]
                self.stats["compatible"] += len(compat)
                pick = compat[:k]
                if len(pick) < k:                          # branch (c): keep the schedule
                    rest = sorted((n for n in range(len(liv)) if n not in pick),
                                  key=lambda n: -cf[n])
                    add = rest[:k - len(pick)]
                    self.stats["fallback"] += len(add)
                    pick += add
                mism = [n for n in pool if w[n] <= 0 and n not in pick]
                self.stats["deferred"] += len(mism)
                pending = {liv[n]: cand[n] for n in mism}

                for n in pick:
                    x[0, liv[n]] = cand[n]
                self.stats["committed"] += len(pick)
        return x.cpu()

    # ---------- detection ----------
    @torch.no_grad()
    def deltas(self, ids, prompt_len, gen_len):
        n_blocks = max(1, gen_len // self.block_len)
        grid = []
        for b in range(n_blocks):
            lo = prompt_len + b * self.block_len
            B, S_list, pairs, lp = self._arms(ids, prompt_len, lo)
            loc = {int(p): k for k, p in enumerate(B)}
            y = ids[0, torch.tensor(B)]
            at = lp.gather(2, y[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(2)
            for j, S in enumerate(S_list):
                rows = [loc[int(i)] for i in S]
                grid.append([float((at[v, rows] - at[u, rows]).mean())
                             for u, v in pairs[j]])
        return np.array(grid)                              # [n_blocks*probes, R]

    def detect(self, ids, prompt_len, gen_len, message=0):
        grid = self.deltas(ids, prompt_len, gen_len)
        signs = payload_bits(self.key, grid.shape[0], message)
        a = (signs[:, None] * grid).ravel()
        r = signflip_pvalue(a)
        D = grid.mean(1)
        n_blocks = max(1, gen_len // self.block_len)
        return dict(matches=int((np.sign(D) == signs).sum()), n=int(grid.shape[0]),
                    n_blocks=int(a.size), z=r["z"], p_bound=r["p_bound"],
                    p_mc=r["p_mc"], p_value=r["p_bound"],
                    n_forwards=n_blocks * self.n_patterns, Delta=D.tolist())
