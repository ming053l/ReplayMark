"""ReTrace V3: model-response-guided resampling.

The commit-order story is retired, by measurement. `exp/07` separated the two mechanisms
that the waiting scheduler had conflated:

    same context, fresh draw   P(compatible within R) = 0.475, 0.625, 0.738, 0.812, 0.875
    context only, no new noise  became compatible      = 0.062, 0.050, 0.075, 0.087, 0.113

Resampling supplies roughly eight times the capacity that committing other positions does,
so the carrier machinery built around deferral -- long waits, slack accounting, CCTC holes,
frontier-anchored commits -- was solving the wrong problem. What remains is much smaller:

    draw v ~ p_theta(. | x)  until  w_i * eps_i * g_i(v) > 0,  at most R times

and every retry is free, because the logits do not change: one forward already gives the
whole row. The step budget stops mattering; R does.

Two properties follow immediately. With per-position acceptance mass
`q_i = P_{v ~ p_i}[v in A_i]` the success probability is `1 - (1 - q_i)^R`, so R trades
watermark strength against nothing but arithmetic; and the accepted token is distributed as
`p_i(v | v in A_i)`.

That second point is a real change to the sampling distribution and is not claimed
otherwise. Every emitted token is still drawn from the model's own conditional and its
support is unchanged, but the distribution is conditioned on the acceptance predicate. The
novelty is not that the predicate is free of distortion -- it is that the predicate is the
model's own reconstruction response rather than a hash of the token identity.

The empirical curve saturates at 0.875 rather than approaching 1, which is the signature of
heterogeneous `q_i`: at some positions almost all mass sits on one side of the contrast and
no number of retries helps. Carrier selection therefore uses

    S_i = 2 * min(q_i^+, q_i^-)

which is orientation-symmetric -- it never looks at eps_i or at the message -- so the
conditional null is untouched, and is computed from the block-masked conditional, which the
verifier reconstructs exactly.
"""
import hashlib, hmac
import numpy as np, torch
import torch.nn.functional as F
from scipy.stats import binom
from .model import MASK_ID
from .prng import stream
from .challenges import orientation_bits, tie_bits, score, block_challenges, roles


class ResampleMark:
    def __init__(self, model, key: bytes, block_len=32, n_patterns=4, ctx_frac=0.20,
                 n_payload_bits=7, sync_frac=0.5, challenge="contrast", carrier_frac=0.5,
                 retries=4, temperature=0.8, nonce=None):
        self.M = model
        self.key = key if nonce is None else hmac.new(
            key, str(nonce).encode(), hashlib.sha256).digest()
        self.block_len, self.n_patterns, self.ctx_frac = block_len, n_patterns, ctx_frac
        self.n_payload_bits, self.sync_frac = n_payload_bits, sync_frac
        self.challenge, self.carrier_frac = challenge, carrier_frac
        self.retries, self.temperature = retries, temperature

    # ------------------------------------------------------------------ table
    @torch.no_grad()
    def _table(self, x, lo, gen_end):
        """Challenge contrast g and the acceptance masses q+/q-, all from the
        block-masked conditional so the verifier reproduces them exactly."""
        B = np.arange(lo, lo + self.block_len)
        pats, pairs = block_challenges(self.key, lo, self.block_len, self.n_patterns,
                                       self.ctx_frac, mode=self.challenge)
        base = x.clone()
        base[0, lo:gen_end] = MASK_ID
        batch = [base.clone() for _ in pats]
        for m, d in zip(batch, pats):
            m[0, torch.tensor(d)] = MASK_ID
        batch.append(base)
        lp = self.M.logprobs_rows(torch.cat(batch, 0), torch.tensor(B), chunk=2,
                                  dtype=torch.float64)
        u, v = pairs[0]
        g = lp[v] - lp[u]                                   # [|B|, V]
        # the sampling distribution the generator will actually draw from, at temperature
        p = torch.softmax(lp[-1] / self.temperature, dim=-1)
        qp = (p * (g > 0)).sum(1).numpy()
        qm = (p * (g < 0)).sum(1).numpy()
        S = 2.0 * np.minimum(qp, qm)                        # orientation-symmetric
        return B, g, S, qp, qm

    def _carrier(self, S):
        """Keep the positions with the most two-sided acceptance mass."""
        n = max(1, int(round(self.carrier_frac * len(S))))
        keep = np.zeros(len(S), dtype=bool)
        keep[np.argsort(-S)[:n]] = True
        return keep

    # -------------------------------------------------------------- detection
    @torch.no_grad()
    def detect(self, ids, prompt_len, gen_len, message=0):
        span = np.arange(prompt_len, prompt_len + gen_len)
        eps = orientation_bits(self.key, span)
        tie = tie_bits(self.key, span)
        role = roles(self.key, span, self.n_payload_bits, self.sync_frac)
        gen_end = prompt_len + gen_len
        hits = np.zeros(self.n_payload_bits, dtype=np.int64)
        tot = np.zeros(self.n_payload_bits, dtype=np.int64)
        hs = ns = 0
        for b in range(max(1, gen_len // self.block_len)):
            lo = prompt_len + b * self.block_len
            B, g, S, _, _ = self._table(ids, lo, gen_end)
            car = self._carrier(S)
            y = ids[0, torch.tensor(B)]
            gv = g.gather(1, y[:, None]).squeeze(1).numpy()
            e = np.array([eps[int(i)] for i in B], dtype=np.float64)
            tb = np.array([tie[int(i)] for i in B], dtype=np.int64)
            m, _ = score(gv, e, tb)
            for k, i in enumerate(B):
                if not car[k]:
                    continue
                j = role[int(i)]
                if j < 0:
                    ns += 1; hs += int(m[k])
                else:
                    tot[j] += 1; hits[j] += int(m[k])
        bits = np.array([int(hits[j] > tot[j] / 2) for j in range(self.n_payload_bits)])
        target = np.array([(message >> t) & 1 for t in range(self.n_payload_bits)])
        na = int(ns + tot.sum())
        ha = hs + sum(int(hits[j]) if ((message >> j) & 1) else int(tot[j] - hits[j])
                      for j in range(self.n_payload_bits))
        return dict(n_sync=ns, hits_sync=hs, rate_sync=hs / max(ns, 1),
                    z=float((hs - ns / 2) / np.sqrt(max(ns, 1) / 4)),
                    p_value=float(binom.sf(hs - 1, ns, 0.5)) if ns else 1.0,
                    n=na, rate_aligned=ha / max(na, 1),
                    p_aligned=float(binom.sf(ha - 1, na, 0.5)) if na else 1.0,
                    bits=bits.tolist(), bit_acc=float((bits == target).mean()),
                    n_seqs=int(max(1, gen_len // self.block_len) * (self.n_patterns + 1)))

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=256, steps=128, message=0, seed=0):
        gen = torch.Generator(device=self.M.device).manual_seed(seed)
        Pn = prompt_ids.shape[1]
        span = np.arange(Pn, Pn + gen_len)
        eps = orientation_bits(self.key, span)
        role = roles(self.key, span, self.n_payload_bits, self.sync_frac)
        want = {int(i): (1 if role[int(i)] < 0 else
                         (1 if ((message >> role[int(i)]) & 1) else -1)) for i in span}
        n_blocks = max(1, gen_len // self.block_len)
        steps_pb = max(1, steps // n_blocks)
        gen_end = Pn + gen_len
        x = torch.full((1, Pn + gen_len), MASK_ID, dtype=torch.long, device=self.M.device)
        x[:, :Pn] = prompt_ids.to(self.M.device)
        self.stats = dict(committed=0, carrier=0, accepted=0, draws=0, exhausted=0)

        for b in range(n_blocks):
            lo = Pn + b * self.block_len
            B, g, S, _, _ = self._table(x.cpu(), lo, gen_end)
            car = self._carrier(S)
            gmap = {int(q): g[k] for k, q in enumerate(B)}
            cmask = {int(q): bool(car[k]) for k, q in enumerate(B)}
            Bt = torch.tensor(B, device=x.device)
            for t in range(steps_pb):
                live = Bt[x[0, Bt] == MASK_ID]
                if live.numel() == 0:
                    break
                k = int(np.ceil(live.numel() / (steps_pb - t)))
                logits = self.M.model(x).logits[0]
                probs = torch.softmax(logits[live].double() / self.temperature, dim=-1)
                conf = probs.max(-1).values
                del logits
                order = torch.argsort(conf, descending=True)[:k].tolist()
                liv = live.tolist()
                for n in order:
                    i = liv[n]
                    row = probs[n]
                    tok = int(torch.multinomial(row, 1, generator=gen))
                    self.stats["draws"] += 1
                    if cmask[i]:
                        self.stats["carrier"] += 1
                        gi = gmap[i]
                        tgt = eps[i] * want[i]
                        for _ in range(self.retries - 1):
                            gv = float(gi[tok])
                            if gv != 0.0 and tgt * gv > 0:
                                break
                            # a retry costs nothing: the logits row is unchanged
                            tok = int(torch.multinomial(row, 1, generator=gen))
                            self.stats["draws"] += 1
                        gv = float(gi[tok])
                        if gv != 0.0 and tgt * gv > 0:
                            self.stats["accepted"] += 1
                        else:
                            self.stats["exhausted"] += 1
                    x[0, i] = tok
                    self.stats["committed"] += 1
        return x.cpu()
