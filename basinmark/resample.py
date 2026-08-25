"""ReplayMark: probe-and-replay model-response watermarking.

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


class ReplayMark:
    def __init__(self, model, key: bytes, block_len=32, n_patterns=8, ctx_frac=0.20,
                 n_payload_bits=7, sync_frac=0.5, challenge="contrast", s_min=0.5,
                 retries=4, temperature=0.8, fallback="fresh", max_carriers=None, p_floor=0.0,
                 nonce=None, ctx_window=None):
        self.M = model
        self.key = key if nonce is None else hmac.new(
            key, str(nonce).encode(), hashlib.sha256).digest()
        self.block_len, self.n_patterns, self.ctx_frac = block_len, n_patterns, ctx_frac
        self.n_payload_bits, self.sync_frac = n_payload_bits, sync_frac
        # an ABSOLUTE threshold, not a rank. Taking the top half by S swept in the ~50 %
        # of positions whose acceptance mass is one-sided, and those score a certain miss
        # once retries are exhausted, which drags the rate below its own null.
        self.challenge, self.s_min = challenge, s_min
        self.retries, self.temperature = retries, temperature
        # On exhaustion, emitting the last REJECTED draw makes the position a certain miss,
        # so the statistic would need acceptance > 0.5 merely to beat its own null. One more
        # unconditional draw removes that: the position then hits with its natural
        # probability q_i, giving 1 - (1-q)^(R+1) rather than 1 - (1-q)^R, at no extra
        # forward pass. "last" keeps the old behaviour for the ablation.
        self.fallback = fallback
        self.max_carriers = max_carriers   # per block; None = every S > s_min position
        # Acceptance may additionally require p(v) >= p_floor * max_v p(v): a key-free,
        # embedder-only cap on the per-token NLL cost (at most -log p_floor beyond the
        # model's own argmax). The detector needs no knowledge of it -- it reads only the
        # emitted token -- so the null and the carrier rule are untouched. This lets R
        # grow without the quality budget growing with it.
        self.p_floor = p_floor
        # Windowed conditioning for edit robustness: when set, the RRB's conditional sees
        # only the prompt plus the last ctx_window generated tokens before the block; all
        # other generated context is masked in every arm, and ablations are drawn from the
        # visible region. An edit then perturbs only blocks whose window covers it, so
        # damage is proportional to edit density instead of propagating to every later
        # block. Positional, key-independent, so the verifier reproduces it exactly.
        self.ctx_window = ctx_window
        # The mask token is a model property (LLaDA 126336, Dream 151666); models that
        # declare .mask_id override the LLaDA default so the pipeline stays model-agnostic.
        self.mask_id = getattr(model, "mask_id", MASK_ID)
        self._p_len = None                 # set by generate()/detect()

    # ------------------------------------------------ Reproducible Response Bank (RRB)
    @torch.no_grad()
    def _build_response_bank(self, x, lo, gen_end):
        """Build the per-block Reproducible Response Bank.

        The RRB stores the selected response contrast g and its two-sided mass statistics,
        all from the block-masked conditional so the Response-Replay Verifier reconstructs
        them exactly.
        """
        B = np.arange(lo, lo + self.block_len)
        region = None
        if self.ctx_window is not None and self._p_len is not None:
            w_lo = max(self._p_len, lo - self.ctx_window)
            region = np.concatenate([np.arange(0, self._p_len), np.arange(w_lo, lo)])
        pats, pairs = block_challenges(self.key, lo, self.block_len, self.n_patterns,
                                       self.ctx_frac, mode=self.challenge, region=region)
        base = x.clone()
        base[0, lo:gen_end] = self.mask_id
        if region is not None and w_lo > self._p_len:
            base[0, self._p_len:w_lo] = self.mask_id
        batch = [base.clone() for _ in pats]
        for m, d in zip(batch, pats):
            m[0, torch.tensor(d)] = self.mask_id
        batch.append(base)
        lp = self.M.logprobs_rows(torch.cat(batch, 0), torch.tensor(B), chunk=2,
                                  dtype=torch.float64)
        # the sampling distribution the generator will actually draw from, at temperature
        p = torch.softmax(lp[-1] / self.temperature, dim=-1)
        # Choose the challenge pair PER POSITION, by two-sided acceptance mass. A fixed
        # near/far pair leaves mean S at 0.283; the best available pair raises it to 0.454
        # (exp/09). The choice is orientation-symmetric -- S is invariant under swapping the
        # pair -- and is computed from the block-masked conditional, so it never sees the
        # key's direction or the message and the verifier reproduces it exactly.
        L = len(pats)
        bestS = np.full(len(B), -1.0)
        bestUV = np.zeros((len(B), 2), dtype=np.int64)
        for u in range(L):
            for v in range(u + 1, L):
                gg = lp[v] - lp[u]
                qp = (p * (gg > 0)).sum(1).numpy()
                qm = (p * (gg < 0)).sum(1).numpy()
                ss = 2.0 * np.minimum(qp, qm)
                better = ss > bestS
                bestS[better] = ss[better]
                bestUV[better] = (u, v)
        g = torch.stack([lp[int(bestUV[k, 1]), k] - lp[int(bestUV[k, 0]), k]
                         for k in range(len(B))])
        # the chosen pair's masses on each side, under the block-masked conditional. S is
        # 2*min of these and is only a worst-orientation lower bound; the mass that
        # actually predicts acceptance is whichever side the key ends up asking for.
        qp = (p * (g > 0)).sum(1).numpy()
        qm = (p * (g < 0)).sum(1).numpy()
        return B, g, bestS, qp, qm

    # Backward-compatible alias for experiment scripts written before the RRB terminology.
    def _table(self, x, lo, gen_end):
        return self._build_response_bank(x, lo, gen_end)

    def _carrier(self, S):
        """Positions whose acceptance mass is two-sided enough to be worth counting.

        With S_i = 2 min(q+, q-), a single draw is accepted with probability at least
        S_i / 2 whichever side the key asks for, so R retries succeed with probability
        1 - (1 - S_i/2)^R. Below the threshold that stays under 0.5, and a position that
        exhausts its retries emits a rejected draw -- a certain miss, not a coin flip -- so
        including it costs signal rather than merely adding noise.
        """
        keep = S > self.s_min
        if self.max_carriers is not None and keep.sum() > self.max_carriers:
            # cap by S so the ppl cost scales with the carrier budget, not the block; the
            # rule is still computed from the block-masked conditional, so the verifier
            # reproduces it
            idx = np.argsort(-S)
            keep = np.zeros_like(keep)
            keep[idx[:self.max_carriers]] = True
        return keep

    # -------------------------------------------------------------- detection
    @torch.no_grad()
    def detect(self, ids, prompt_len, gen_len, message=0):
        self._p_len = prompt_len
        span = np.arange(prompt_len, prompt_len + gen_len)
        eps = orientation_bits(self.key, span)
        tie = tie_bits(self.key, span)
        role = roles(self.key, span, self.n_payload_bits, self.sync_frac)
        gen_end = prompt_len + gen_len
        hits = np.zeros(self.n_payload_bits, dtype=np.int64)
        tot = np.zeros(self.n_payload_bits, dtype=np.int64)
        hs = ns = 0
        blocks = []                        # per-block presence (hits, n) for local tests
        for b in range(max(1, gen_len // self.block_len)):
            lo = prompt_len + b * self.block_len
            B, g, S, _, _ = self._build_response_bank(ids, lo, gen_end)
            car = self._carrier(S)
            y = ids[0, torch.tensor(B)]
            gv = g.gather(1, y[:, None]).squeeze(1).numpy()
            e = np.array([eps[int(i)] for i in B], dtype=np.float64)
            tb = np.array([tie[int(i)] for i in B], dtype=np.int64)
            m, _ = score(gv, e, tb)
            bh = bn = 0
            for k, i in enumerate(B):
                if not car[k]:
                    continue
                j = role[int(i)]
                if j < 0:
                    ns += 1; hs += int(m[k]); bn += 1; bh += int(m[k])
                else:
                    tot[j] += 1; hits[j] += int(m[k])
            blocks.append((int(bh), int(bn)))
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
                    blocks=blocks,
                    n_seqs=int(max(1, gen_len // self.block_len) * (self.n_patterns + 1)))

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=256, steps=128, message=0, seed=0):
        gen = torch.Generator(device=self.M.device).manual_seed(seed)
        Pn = prompt_ids.shape[1]
        self._p_len = Pn
        span = np.arange(Pn, Pn + gen_len)
        eps = orientation_bits(self.key, span)
        role = roles(self.key, span, self.n_payload_bits, self.sync_frac)
        want = {int(i): (1 if role[int(i)] < 0 else
                         (1 if ((message >> role[int(i)]) & 1) else -1)) for i in span}
        n_blocks = max(1, gen_len // self.block_len)
        steps_pb = max(1, steps // n_blocks)
        gen_end = Pn + gen_len
        x = torch.full((1, Pn + gen_len), self.mask_id, dtype=torch.long,
                       device=self.M.device)
        x[:, :Pn] = prompt_ids.to(self.M.device)
        self.stats = dict(committed=0, carrier=0, accepted=0, draws=0, exhausted=0,
                          # S comes from the block-masked conditional, but sampling happens
                          # under the current decoder state; log the acceptance actually
                          # observed per draw so the two can be calibrated rather than
                          # assumed equal
                          draw_hits=0, draw_tot=0, s_sum=0.0,
                          qbase_sum=0.0, qgen_sum=0.0)
        self.qpairs = []                    # per-carrier (q_base_target, q_gen_target)

        for b in range(n_blocks):
            lo = Pn + b * self.block_len
            B, g, S, qp, qm = self._build_response_bank(x.cpu(), lo, gen_end)
            car = self._carrier(S)
            gmap = {int(q): g[k] for k, q in enumerate(B)}
            cmask = {int(q): bool(car[k]) for k, q in enumerate(B)}
            smap = {int(q): float(S[k]) for k, q in enumerate(B)}
            qpm = {int(q): (float(qp[k]), float(qm[k])) for k, q in enumerate(B)}
            Bt = torch.tensor(B, device=x.device)
            for t in range(steps_pb):
                live = Bt[x[0, Bt] == self.mask_id]
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
                        self.stats["s_sum"] += smap[i]
                        gi = gmap[i]
                        tgt = eps[i] * want[i]
                        # what the block-masked conditional predicts for THIS position's
                        # target side, and what the live decoder state actually offers
                        qb = qpm[i][0] if tgt > 0 else qpm[i][1]
                        qg = float((row * ((tgt * gi.to(row.device)) > 0)).sum())
                        self.stats["qbase_sum"] += qb
                        self.stats["qgen_sum"] += qg
                        self.qpairs.append((qb, qg))
                        # R is the number of GUIDED proposals, counting the first draw.
                        # Redrawing after the final check would waste a sample, since that
                        # draw is never examined; the single unconditional fallback below
                        # is what R+1 in the hit probability refers to.
                        hit = False
                        pmax = float(row.max())
                        for r in range(self.retries):
                            gv = float(gi[tok])
                            self.stats["draw_tot"] += 1
                            if (gv != 0.0 and tgt * gv > 0
                                    and float(row[tok]) >= self.p_floor * pmax):
                                self.stats["draw_hits"] += 1
                                hit = True
                                break
                            if r < self.retries - 1:
                                # a retry costs nothing: the logits row is unchanged
                                tok = int(torch.multinomial(row, 1, generator=gen))
                                self.stats["draws"] += 1
                        if hit:
                            self.stats["accepted"] += 1
                        else:
                            self.stats["exhausted"] += 1
                            if self.fallback == "fresh":
                                tok = int(torch.multinomial(row, 1, generator=gen))
                                self.stats["draws"] += 1
                    x[0, i] = tok
                    self.stats["committed"] += 1
        return x.cpu()


# Historical experiment scripts import ResampleMark.  Keep the alias so result
# reproduction does not depend on rewriting those archived entry points.
ResampleMark = ReplayMark
