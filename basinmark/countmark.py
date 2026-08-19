"""CountMark: a re-denoising watermark whose statistic is a sum of bounded indicators.

Why the previous statistic could not be driven, measured rather than argued
(`exp/26_diag.py`): the per-position quantity s*g is heavy-tailed. Averaging sixteen of
them into one Delta_j leaves the mean dominated by a few large values, so controlling the
*sign* of the small ones -- which is all the commit-order channel can do -- barely moves
it. On unwatermarked text the fraction of positions with s*g > 0 measured 0.18-0.52 across
samples while the mean sat at ~0, which is exactly what a right-skewed distribution looks
like.

dgMARK does not have this problem because its statistic is a count: every controlled
position contributes exactly one unit, so a modest per-position bias over many positions
becomes a large z. The fix is to adopt that *shape* while keeping the observable a keyed
re-denoising response rather than a hash of the token identity.

Give every position its own keyed orientation bit eps_i in {-1, +1} and define

    m_i = 1[ eps_i * g_i(y_i) > 0 ],        T = sum_i m_i

where g_i is the same challenge contrast as before. Under H0 the eps_i are independent
fair coins, so the m_i are i.i.d. Bernoulli(1/2) and

    T ~ Binomial(n, 1/2)      exactly.

This is stronger than what the previous design could claim: exact enumeration rather than
Hoeffding's bound, and no resolution floor. Per-probe orientation bits could not give this
-- one bit flips every position in the probe at once, so a probe carried one bit of
randomness no matter how many positions it held.

Payload: positions are partitioned into M keyed groups and each group's count carries one
bit, the same construction dgMARK uses.

Embedding is unchanged in spirit and never alters a token: during block decoding, prefer
committing positions whose *model-preferred* token already satisfies eps_i * g_i > 0, and
defer the others.
"""
import hashlib, hmac
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import binom
from .model import MASK_ID
from .prng import stream


def orientation_bits(key: bytes, positions, salt="eps"):
    """One independent keyed coin per position -- the source of the exact binomial null."""
    out = {}
    for i in positions:
        h = hmac.new(key, f"{salt}:{int(i)}".encode(), hashlib.sha256).digest()
        out[int(i)] = 1 if (h[0] & 1) else -1
    return out


def tie_bits(key: bytes, positions):
    """A second keyed coin, used only where g is exactly zero.

    Ties are not rare here and ignoring them silently breaks the null. LLaDA is confident
    enough that at many positions the emitted token has probability indistinguishable from
    one, so log_softmax rounds to 0.0 under BOTH arms and their difference is exactly zero.
    Scoring `eps * g > 0` then counts every such position as a miss, which drags the
    unwatermarked match rate to P(g != 0)/2 -- measured at 0.32-0.37 rather than 0.50,
    thirteen standard deviations off. Resolving ties by an independent keyed coin restores
    Bernoulli(1/2) by construction; the positions stay uncontrollable during embedding, in
    the same way dgMARK gains nothing at a position where no candidate matches its parity.
    """
    out = {}
    for i in positions:
        h = hmac.new(key, f"tie:{int(i)}".encode(), hashlib.sha256).digest()
        out[int(i)] = 1 if (h[1] & 1) else 0
    return out


def score(gv, eps, tie):
    """m_i, with exact ties broken by the keyed coin rather than counted as misses."""
    gv = np.asarray(gv, dtype=np.float64)
    m = (eps * gv > 0).astype(np.int64)
    z = gv == 0.0
    m[z] = tie[z]
    return m, int(z.sum())


def block_challenges(key, block_lo, block_len, n_patterns, ctx_frac=0.20, min_ctx=4,
                     mode="random"):
    """L ablation patterns over the already-final region, and the keyed pairing.

    mode="contrast" builds the pair from opposite ends of the context instead of at
    random: one pattern ablates the positions nearest the block, the other the furthest.
    Recency dominates next-token prediction, so the two arms then disagree far more, which
    widens the dynamic range of g -- the quantity `exp/21_challenge.py` measured a
    1.8-2.2x headroom on when merely picking the best of 28 random pairs. Validity is
    unaffected: the exact null needs the *orientation* bit to be a keyed coin, not the
    unordered pair to be content-independent.
    """
    region = np.arange(0, block_lo)
    n_d = max(min_ctx, int(round(ctx_frac * len(region))))
    n_d = min(n_d, max(1, len(region) // 2))
    if mode == "contrast":
        near, far = np.sort(region[-n_d:]), np.sort(region[:n_d])
        pats = [near, far]
        for l in range(2, n_patterns):
            pats.append(np.sort(stream(key, "cpat", block_lo, l)
                                .choice(region, n_d, replace=False)))
        pairs = [(0, 1)]
    else:
        pats = [np.sort(stream(key, "rpat", block_lo, l).choice(region, n_d, replace=False))
                for l in range(n_patterns)]
        r_ = stream(key, "rpair", block_lo)
        u, v = r_.choice(n_patterns, 2, replace=False)
        pairs = [(int(u), int(v))]
    return pats, pairs


class CountMark:
    def __init__(self, model, key: bytes, block_len=32, n_patterns=4, ctx_frac=0.20,
                 tau_conf=0.5, holes=4, n_bits=8, challenge="contrast", nonce=None):
        self.M = model
        self.key = key if nonce is None else hmac.new(
            key, str(nonce).encode(), hashlib.sha256).digest()
        self.block_len, self.n_patterns, self.ctx_frac = block_len, n_patterns, ctx_frac
        self.tau_conf, self.holes, self.n_bits = tau_conf, holes, n_bits
        self.challenge = challenge

    # ---------- challenge table for one block ----------
    @torch.no_grad()
    def _table(self, x, lo, gen_end):
        """g_i(.) for every position of the block, as a [|B|, V] tensor.

        The block AND everything after it are masked, which is the state the generator is
        in while the block is decoded -- so the detector, recomputing this on finished
        text, gets the identical table. Omitting the tail cost the previous version its
        entire signal.
        """
        B = np.arange(lo, lo + self.block_len)
        pats, pairs = block_challenges(self.key, lo, self.block_len, self.n_patterns,
                                       self.ctx_frac, mode=self.challenge)
        base = x.clone()
        base[0, lo:gen_end] = MASK_ID
        batch = []
        for d in pats:
            m = base.clone()
            m[0, torch.tensor(d)] = MASK_ID
            batch.append(m)
        # float64 must be requested INSIDE log_softmax; casting afterwards recovers
        # nothing, since the rounding that creates the ties has already happened
        lp = self.M.logprobs_rows(torch.cat(batch, 0), torch.tensor(B), chunk=2,
                                  dtype=torch.float64)
        u, v = pairs[0]
        return B, (lp[v] - lp[u])                     # [|B|, V], float64

    def _eps(self, span):
        return orientation_bits(self.key, span)

    # ---------- detection ----------
    @torch.no_grad()
    def detect(self, ids, prompt_len, gen_len, message=0):
        span = np.arange(prompt_len, prompt_len + gen_len)
        eps = self._eps(span)
        tie = tie_bits(self.key, span)
        gen_end = prompt_len + gen_len
        hits = np.zeros(self.n_bits, dtype=np.int64)
        tot = np.zeros(self.n_bits, dtype=np.int64)
        n_tie = 0
        grp = stream(self.key, "grp", gen_len, self.n_bits).integers(0, self.n_bits,
                                                                    size=gen_len)
        for b in range(max(1, gen_len // self.block_len)):
            lo = prompt_len + b * self.block_len
            B, g = self._table(ids, lo, gen_end)
            y = ids[0, torch.tensor(B)]
            gv = g.gather(1, y[:, None]).squeeze(1).numpy()
            e = np.array([eps[int(i)] for i in B], dtype=np.float64)
            tb = np.array([tie[int(i)] for i in B], dtype=np.int64)
            m, nz = score(gv, e, tb)
            n_tie += nz
            for k, i in enumerate(B):
                j = int(grp[int(i) - prompt_len])
                tot[j] += 1
                hits[j] += int(m[k])
        n, h = int(tot.sum()), int(hits.sum())
        bits = (hits > tot / 2).astype(int)
        target = np.array([(message >> t) & 1 for t in range(self.n_bits)])
        return dict(green=h, n=n, rate=h / max(n, 1), tie_frac=n_tie / max(n, 1),
                    z=float((h - n / 2) / np.sqrt(n / 4)),
                    p_value=float(binom.sf(h - 1, n, 0.5)),
                    bits=bits.tolist(), bit_acc=float((bits == target).mean()),
                    n_forwards=int(max(1, gen_len // self.block_len) * self.n_patterns))

    # ---------- generation ----------
    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=256, steps=128, temperature=0.8, message=0,
                 seed=0):
        gen = torch.Generator(device=self.M.device).manual_seed(seed)
        Pn = prompt_ids.shape[1]
        span = np.arange(Pn, Pn + gen_len)
        eps = self._eps(span)
        tie = tie_bits(self.key, span)
        grp = stream(self.key, "grp", gen_len, self.n_bits).integers(0, self.n_bits,
                                                                    size=gen_len)
        want = {int(i): (1 if ((message >> int(grp[int(i) - Pn])) & 1) else -1)
                for i in span}
        n_blocks = max(1, gen_len // self.block_len)
        steps_pb = max(1, steps // n_blocks)
        gen_end = Pn + gen_len

        x = torch.full((1, Pn + gen_len), MASK_ID, dtype=torch.long, device=self.M.device)
        x[:, :Pn] = prompt_ids.to(self.M.device)
        self.stats = dict(committed=0, wm=0, fallback=0, flipped=0)

        for b in range(n_blocks):
            lo = Pn + b * self.block_len
            B, g = self._table(x.cpu(), lo, gen_end)
            gmap = {int(p): g[k] for k, p in enumerate(B)}
            Bt = torch.tensor(B, device=x.device)
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
                conf = F.softmax(logits.double(), -1).gather(-1, xh[:, None]).squeeze(1)
                del logits
                liv, cand = live.tolist(), xh[live].tolist()
                cf = conf[live].tolist()
                # does the model's OWN choice already answer the challenge correctly?
                # a tie is not steerable: its score is fixed by the keyed coin
                ok = [(eps[i] * want[i] * float(gmap[i][v]) > 0)
                      if float(gmap[i][v]) != 0.0 else False
                      for i, v in zip(liv, cand)]

                elig = [n for n, c in enumerate(cf) if c >= self.tau_conf]
                eset = set(liv[n] for n in elig)
                safe = [n for n in elig
                        if sum(1 for jj in liv if jj < liv[n] and jj not in eset)
                        <= self.holes]
                pick = [n for n in safe if ok[n]][:k]
                self.stats["wm"] += len(pick)
                if len(pick) < k:
                    rest = sorted((n for n in range(len(liv)) if n not in pick),
                                  key=lambda n: -cf[n])[:k - len(pick)]
                    self.stats["fallback"] += len(rest)
                    pick += rest
                for n in pick:
                    x[0, liv[n]] = cand[n]
                self.stats["committed"] += len(pick)
        return x.cpu()
