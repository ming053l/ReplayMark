"""KGW baseline on LLaDA, in the same harness as everything else.

The eth-sri repo's C4 config is an *infilling* task with <|mask|> placeholders, not the
prompt-continuation setting dgMARK and BasinMark are run in, so driving it would swap one
mismatch for another. The dgMARK paper defines its own KGW baseline instead: "results are
derived from dLLMs configured to generate tokens sequentially from left to right", because
KGW's green list is seeded by the preceding token and that prefix does not exist under
order-agnostic decoding. That definition is what is implemented here, inside our own
generation loop, so model, prompts, length, sampler, perplexity evaluator and reporting
are identical to the BasinMark runs.

Green list: h = HMAC(key, previous token id) seeds a permutation; the first gamma*|V|
tokens are green and receive +delta on their logits before sampling.
Detection: |s|_G green tokens out of T gives z = (|s|_G - gamma*T) / sqrt(T*gamma*(1-gamma)).
"""
import hashlib, hmac
import numpy as np, torch
import torch.nn.functional as F
from .model import MASK_ID


_CACHE = {}


def green_mask(key: bytes, prev_token: int, vocab: int, gamma: float, device):
    ck = (int(prev_token), vocab, gamma)
    if ck not in _CACHE:
        seed = int.from_bytes(hmac.new(key, str(int(prev_token)).encode(),
                                       hashlib.sha256).digest()[:8], "big")
        g = torch.Generator(device="cpu").manual_seed(seed % (2 ** 63))
        perm = torch.randperm(vocab, generator=g)
        m = torch.zeros(vocab, dtype=torch.bool)
        m[perm[:int(gamma * vocab)]] = True
        if len(_CACHE) > 20000:
            _CACHE.clear()
        _CACHE[ck] = m
    return _CACHE[ck].to(device)


@torch.no_grad()
def kgw_generate(M, prompt_ids, gen_len=256, gamma=0.5, delta=2.0, key=b"kgw",
                 temperature=0.8, seed=0):
    """Left-to-right decoding with a dLLM: exactly one position is unmasked per step."""
    gen = torch.Generator(device=M.device).manual_seed(seed)
    P = prompt_ids.shape[1]
    x = torch.full((1, P + gen_len), getattr(M, "mask_id", MASK_ID), dtype=torch.long,
                   device=M.device)
    x[:, :P] = prompt_ids.to(M.device)
    for t in range(gen_len):
        i = P + t
        logits = M.model(x).logits[0, i].float()
        if delta > 0:
            logits = logits + delta * green_mask(key, int(x[0, i - 1]), logits.shape[0],
                                                 gamma, logits.device).float()
        if temperature > 0:
            p = F.softmax(logits / temperature, -1)
            x[0, i] = torch.multinomial(p, 1, generator=gen)
        else:
            x[0, i] = logits.argmax()
    return x.cpu()


def kgw_detect(M, ids, span, gamma=0.5, key=b"kgw", dedup=True, vocab=126464):
    """Score each DISTINCT (previous, current) bigram once.

    Without dedup the null breaks. The green list is seeded by the previous token, so a
    repeated bigram is scored repeatedly with a perfectly correlated indicator: one
    repeated bigram that happens to be green drags the whole count up. LLaDA is not
    trained for strict left-to-right decoding and produces repetitive text under it, and
    the un-deduplicated detector measured z = +3.32 with TPR 0.37 on the delta = 0
    NON-watermarked control -- a 37 % false-positive rate. Deduplication is the standard
    fix (Kirchenbauer et al. report it for the same reason).
    """
    seen, hits, tot = set(), 0, 0
    for i in span:
        prev, cur = int(ids[0, i - 1]), int(ids[0, i])
        if dedup:
            if (prev, cur) in seen:
                continue
            seen.add((prev, cur))
        hits += int(green_mask(key, prev, vocab, gamma, "cpu")[cur])
        tot += 1
    z = (hits - gamma * tot) / np.sqrt(max(tot, 1) * gamma * (1 - gamma))
    return dict(green=hits, n=tot, z=float(z), rate=hits / max(tot, 1),
                dup_frac=1.0 - tot / len(span))
