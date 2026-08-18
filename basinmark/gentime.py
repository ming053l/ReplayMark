"""Generation-time BasinMark.

The post-hoc embedder is dominated on the quality-detection plane: dgMARK buys 92 % TPR
at 1 % FPR for x1.10 perplexity, BasinMark has no usable TPR at x1.35. The lesson from
dgMARK is that it never substitutes a token -- it steers which position is unmasked next.

Note what moving to generation time does and does not buy. The per-token price is
identical either way: choosing v instead of the model's preferred v* costs
log p(v*) - log p(v) whether the token was previously committed or not. What is saved is
**joint coherence**. The post-hoc embedder masks the whole carrier (30 % of the span) and
refills it in two steps, i.e. it fills half the carrier from independent marginals; the
generator commits progressively, so each decided token conditions the next. That is the
mechanism this file tests, and it is falsifiable: if perplexity does not improve at
matched tau/lambda, the coherence hypothesis is wrong and the flatness of g is the whole
story.

Structure forced by the guidance table: the arms condition on span \\ (Q u D), so every
NON-pool token must be final before the table is computed. Hence two phases -- fill the
non-pool positions, compute the table once, then fill the pool under guidance. The table
stays exactly valid because the arms mask all of Q and nothing outside Q is ever written.
"""
import numpy as np, torch
import torch.nn.functional as F
from .model import MASK_ID
from .select import LeverageMark
from .prng import payload_bits


class GenMark(LeverageMark):
    @torch.no_grad()
    def _denoise_subset(self, x, subset, steps, temperature, gen, guide=None):
        """LLaDA low-confidence remasking restricted to `subset` (absolute positions).

        With `guide`, carrier positions pick argmax over the admissible set of
        log p(v) + lam * s_j * g_i(v) instead of the model's own choice; every other
        position follows the reference sampler untouched.
        """
        sub = torch.tensor(np.sort(subset), device=x.device)
        for t in range(steps):
            live = sub[x[0, sub] == MASK_ID]
            if live.numel() == 0:
                break
            k = int(np.ceil(live.numel() / (steps - t)))
            logits = self.M.model(x).logits[0]
            if temperature > 0:
                u = torch.rand(logits.shape, device=logits.device, dtype=torch.float64,
                               generator=gen)
                x0 = (logits.double() / temperature - torch.log(-torch.log(u))).argmax(-1)
            else:
                x0 = logits.argmax(-1)
            conf = F.softmax(logits.double(), -1).gather(-1, x0[:, None]).squeeze(1)
            if guide is not None:
                lp = F.log_softmax(logits[live].float(), -1)
                top = lp.max(1, keepdim=True).values
                gm = torch.stack([guide.get(int(i), torch.zeros(lp.shape[1]))
                                  for i in live.tolist()]).to(lp.device)
                obj = torch.where(lp >= (top - self.tau), lp + self.lam * gm,
                                  torch.full_like(lp, torch.finfo(lp.dtype).min))
                x0 = x0.clone()
                x0[live] = obj.argmax(1)
                # rank by the position's intrinsic certainty, never by the chosen token,
                # or every guided pick is deferred to where the admissible set is a
                # singleton (the bug that starved the post-hoc embedder)
                conf = conf.clone()
                conf[live] = top.squeeze(1).double()
            del logits
            c = torch.full_like(conf, -np.inf)
            c[live] = conf[live]
            sel = torch.topk(c, k=min(k, live.numel())).indices
            x[0, sel] = x0[sel]
        return x

    @torch.no_grad()
    def generate(self, prompt_ids, gen_len=256, steps=256, temperature=0.8, message=0,
                 seed=0):
        gen = torch.Generator(device=self.M.device).manual_seed(seed)
        Pn = prompt_ids.shape[1]
        span = np.arange(Pn, Pn + gen_len)
        x = torch.full((1, Pn + gen_len), MASK_ID, dtype=torch.long, device=self.M.device)
        x[:, :Pn] = prompt_ids.to(self.M.device)

        Q, _, _, _ = self._pat(span)
        nonpool = np.setdiff1d(span, Q)
        # phase 1: everything the arms will condition on
        s1 = max(1, int(round(steps * len(nonpool) / gen_len)))
        x = self._denoise_subset(x, nonpool, s1, temperature, gen)

        # phase 2: the guidance table, computed once and exact from here on
        _, S_list, arms, pos_of, pairs, _ = self._prepare(x.cpu(), span)
        signs = payload_bits(self.key, self.n_probes, message)
        guide = {}
        for j, S in enumerate(S_list):
            for i in S:
                r = pos_of[int(i)]
                guide[int(i)] = signs[j] * torch.stack(
                    [arms[v][r] - arms[u][r] for u, v in pairs[j]]).mean(0)

        # phase 3: fill the pool, guided at carrier positions
        s3 = max(1, steps - s1)
        x = self._denoise_subset(x, Q, s3, temperature, gen, guide=guide)
        return x.cpu()
