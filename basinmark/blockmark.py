"""BasinMark V2: order is the embedding channel, behaviour is the verification channel.

V1 embedded by *substituting* tokens, and that is what priced it out: the median position
admits one token inside any usable fluency budget, so buying signal meant paying several
nats for it (see `legacy/` and the analysis section of the README).

V2 never touches a token. The model proposes what it would have proposed; the watermark
only decides **which position commits first**. A position whose model-preferred token
already answers the keyed re-denoising challenge is committed; one that does not is
deferred, and with more context the model may itself propose a compatible token there.
That is the property making dgMARK nearly free, applied to a behavioural observable
rather than a hash of the token identity.

Decoding follows the reference LLaDA schedule -- blocks in order, diffusion within a
block. A block's challenge table is built once with that block, and everything after it,
masked; nothing committed inside the block can then enter the table's conditioning, and
the detector rebuilds the identical table from the finished text.

The statistic is a count, not an average:

    m_i = 1[ eps_i * g_i(y_i) > 0 ],   T = sum_i m_i  ~  Binomial(n, 1/2) exactly

with one keyed orientation bit per position. Averaging a heavy-tailed contrast, as V1
did, leaves sign control with no leverage; a count gives every steered position exactly
one unit.
"""
import hashlib, hmac
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import binom
from .model import MASK_ID
from .prng import stream
from .challenges import orientation_bits, tie_bits, score, block_challenges


class BlockMark:
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
