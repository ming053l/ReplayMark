"""Archived V2: order steering as the embedding channel.

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
block. A block's Reproducible Response Bank is built once with that block, and everything after it,
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
from .challenges import (orientation_bits, tie_bits, score, block_challenges,
                         steerable_carrier, roles)


class BlockMark:
    def __init__(self, model, key: bytes, block_len=32, n_patterns=4, ctx_frac=0.20,
                 tau_conf=0.5, holes=4, n_payload_bits=7, sync_frac=0.5,
                 challenge="contrast", gap_nats=1.0, nonce=None):
        self.M = model
        self.key = key if nonce is None else hmac.new(
            key, str(nonce).encode(), hashlib.sha256).digest()
        self.block_len, self.n_patterns, self.ctx_frac = block_len, n_patterns, ctx_frac
        self.tau_conf, self.holes = tau_conf, holes
        # n_payload_bits counts PAYLOAD bits only; the presence pool is separate, so the
        # two are never the same number and cannot silently disagree
        self.n_payload_bits, self.sync_frac = n_payload_bits, sync_frac
        self.challenge, self.gap_nats = challenge, gap_nats

    # ---------- Reproducible Response Bank for one block ----------
    @torch.no_grad()
    def _build_response_bank(self, x, lo, gen_end):
        """g_i(.) for every position of the block, as a [|B|, V] tensor.

        The block AND everything after it are masked, which is the state the generator is
        in while the block is decoded -- so the detector, recomputing this on finished
        text, gets the identical response bank. Omitting the tail cost the previous version its
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
        batch.append(base)                            # unablated: the clean conditional
        lp = self.M.logprobs_rows(torch.cat(batch, 0), torch.tensor(B), chunk=2,
                                  dtype=torch.float64)
        u, v = pairs[0]
        carrier = steerable_carrier(lp[-1], self.gap_nats)
        return B, (lp[v] - lp[u]), carrier            # [|B|, V] float64, [|B|] bool

    # Backward-compatible alias for earlier diagnostic scripts.
    def _table(self, x, lo, gen_end):
        return self._build_response_bank(x, lo, gen_end)

    def _eps(self, span):
        return orientation_bits(self.key, span)

    # ---------- detection ----------
    @torch.no_grad()
    def detect(self, ids, prompt_len, gen_len, message=0):
        span = np.arange(prompt_len, prompt_len + gen_len)
        eps = self._eps(span)
        tie = tie_bits(self.key, span)
        gen_end = prompt_len + gen_len
        role = roles(self.key, span, self.n_payload_bits, self.sync_frac)
        hits = np.zeros(self.n_payload_bits, dtype=np.int64)
        tot = np.zeros(self.n_payload_bits, dtype=np.int64)
        hs = ns = n_tie = 0
        for b in range(max(1, gen_len // self.block_len)):
            lo = prompt_len + b * self.block_len
            B, g, car = self._build_response_bank(ids, lo, gen_end)
            y = ids[0, torch.tensor(B)]
            gv = g.gather(1, y[:, None]).squeeze(1).numpy()
            e = np.array([eps[int(i)] for i in B], dtype=np.float64)
            tb = np.array([tie[int(i)] for i in B], dtype=np.int64)
            m, nz = score(gv, e, tb)
            n_tie += nz
            for k, i in enumerate(B):
                if not car[k]:            # never steerable: counting it only adds noise
                    continue
                j = role[int(i)]
                if j < 0:
                    ns += 1; hs += int(m[k])
                else:
                    tot[j] += 1; hits[j] += int(m[k])
        # ---- presence: the sync group, whose target is always +1 ----
        # Presence must not be read off the payload groups. With a balanced message the
        # payload groups cancel: half are driven towards m=1 and half towards m=0, so a
        # PERFECT watermark still sums to n/2 and the aggregate z is ~0 by construction.
        # That, not a dead channel, is what the earlier aggregate reported. Testing
        # presence on a group whose target is fixed also avoids decoding a message from
        # the data and then testing significance against the message just decoded.
        bits = np.array([int(hits[j] > tot[j] / 2) for j in range(self.n_payload_bits)])
        target = np.array([(message >> t) & 1 for t in range(self.n_payload_bits)])

        # ---- verification of a CLAIMED message: align each group to its claimed bit ----
        na = int(ns + tot.sum())
        ha = hs + sum(int(hits[j]) if ((message >> j) & 1) else int(tot[j] - hits[j])
                      for j in range(self.n_payload_bits))
        return dict(n_sync=ns, hits_sync=hs, rate_sync=hs / max(ns, 1),
                    z=float((hs - ns / 2) / np.sqrt(max(ns, 1) / 4)),
                    p_value=float(binom.sf(hs - 1, ns, 0.5)) if ns else 1.0,
                    n=na, hits_aligned=ha, rate_aligned=ha / max(na, 1),
                    z_aligned=float((ha - na / 2) / np.sqrt(max(na, 1) / 4)),
                    p_aligned=float(binom.sf(ha - 1, na, 0.5)) if na else 1.0,
                    tie_frac=n_tie / max(na, 1),
                    bits=bits.tolist(),
                    bit_acc=float((bits == target).mean()),
                    n_payload=int(tot.sum()),
                    # one model invocation per ablation plus one unablated pass, per block
                    n_seqs=int(max(1, gen_len // self.block_len) * (self.n_patterns + 1)))

    # ---------- generation ----------
    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=256, steps=128, temperature=0.8, message=0,
                 seed=0):
        gen = torch.Generator(device=self.M.device).manual_seed(seed)
        Pn = prompt_ids.shape[1]
        span = np.arange(Pn, Pn + gen_len)
        eps = self._eps(span)
        tie = tie_bits(self.key, span)
        role = roles(self.key, span, self.n_payload_bits, self.sync_frac)
        want = {int(i): (1 if role[int(i)] < 0 else
                         (1 if ((message >> role[int(i)]) & 1) else -1)) for i in span}
        n_blocks = max(1, gen_len // self.block_len)
        steps_pb = max(1, steps // n_blocks)
        gen_end = Pn + gen_len

        x = torch.full((1, Pn + gen_len), MASK_ID, dtype=torch.long, device=self.M.device)
        x[:, :Pn] = prompt_ids.to(self.M.device)
        self.stats = dict(committed=0, wm=0, fallback=0, waited=0,
                          carrier_wm=0, carrier_fb=0, noncarrier=0)
        self.provenance = {}

        for b in range(n_blocks):
            lo = Pn + b * self.block_len
            B, g, car = self._build_response_bank(x.cpu(), lo, gen_end)
            gmap = {int(p): g[k] for k, p in enumerate(B)}
            Bt = torch.tensor(B, device=x.device)
            for t in range(steps_pb):
                live = Bt[x[0, Bt] == MASK_ID]
                if live.numel() == 0:
                    break
                # A step budget larger than the block only helps if a step is allowed
                # to commit NOTHING. exp/06 measured that a deferred position's proposal
                # changes 0.20 of the time after 1-2 steps but 0.57 after 11+, so the
                # channel converts waiting into fresh draws -- and the old schedule never
                # waited, because ceil(live / steps_left) is at least one. Commit only when
                # something is compatible, and force progress only once the remaining steps
                # are needed to finish the block.
                slack = (steps_pb - t) - int(live.numel())
                k = max(1, int(np.ceil(live.numel() / max(1, steps_pb - t))))
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
                cpos = {int(q): bool(car[kk]) for kk, q in enumerate(B)}
                cmask = [cpos[i] for i in liv]
                # a tie is not steerable: its score is fixed by the keyed coin
                ok = [(eps[i] * want[i] * float(gmap[i][v]) > 0)
                      if float(gmap[i][v]) != 0.0 else False
                      for i, v in zip(liv, cand)]

                elig = [n for n, c in enumerate(cf) if c >= self.tau_conf]
                eset = set(liv[n] for n in elig)
                safe = [n for n in elig
                        if sum(1 for jj in liv if jj < liv[n] and jj not in eset)
                        <= self.holes]
                # steer ONLY on carriers: the detector counts nothing else
                pick = [n for n in safe if cmask[n] and ok[n]][:k]
                self.stats["wm"] += len(pick)
                self.stats["carrier_wm"] += len(pick)
                if not pick and slack > 0:
                    self.stats["waited"] += 1
                    continue                      # spend a spare step waiting instead
                if len(pick) < k:
                    self.stats["_nd"] = len(pick)
                    rest = sorted((n for n in range(len(liv)) if n not in pick),
                                  key=lambda n: -cf[n])[:k - len(pick)]
                    self.stats["fallback"] += len(rest)
                    pick += rest
                else:
                    self.stats["_nd"] = len(pick)
                for idx, n in enumerate(pick):
                    x[0, liv[n]] = cand[n]
                    if cmask[n]:
                        driven = idx < self.stats["_nd"]
                        if not driven:
                            self.stats["carrier_fb"] += 1
                        # record provenance per position so a later accounting can use a
                        # single denominator; mixing sync-only rates with all-carrier
                        # counts is what invalidated the previous estimate
                        self.provenance[int(liv[n])] = "wm" if driven else "fb"
                    else:
                        self.stats["noncarrier"] += 1
                self.stats["committed"] += len(pick)
        return x.cpu()
