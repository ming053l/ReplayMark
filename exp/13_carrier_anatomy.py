"""Carrier anatomy: is q_gen unstable at single positions, and what does top-p leave?

Means hide the failure mode. q_base ~ 0.50 and q_gen ~ 0.50 with MAE 0.35 is exactly what
{0.03, 0.97, 0.06, 0.94, ...} looks like: fine on average, but whenever the key asks for
the 0.03 side, acceptance means rejecting 97% of the model's mass -- which is where the
perplexity damage would come from. So: quantiles and low-tail fractions of q_gen_target on
selected carriers, plus, offline, how much two-sided mass survives top-k/top-p truncation
of the LIVE row (if none does, an end-to-end top-p run is pointless). Also: why are some
generations empty -- count special/EOS tokens in the raw ids.
"""
import sys
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch
from basinmark.model import BasinModel, MASK_ID
from basinmark.data import c4_prompts
from basinmark.resample import ResampleMark
from basinmark.challenges import orientation_bits, roles

KEY, MESSAGE, GEN, BLK, NS, R = b"retrace-key-A", 0xA5, 256, 32, 6, 2
M = BasinModel()
qg_all, trunc = [], {k: [] for k in ("p80", "p90", "p95", "k10")}
empty_report = []

for i, p in enumerate(c4_prompts(M.tok, NS)):
    pl = p.shape[1]
    w = ResampleMark(M, KEY, block_len=BLK, s_min=0.5, retries=R, nonce=f"doc-{i}")
    span = np.arange(pl, pl + GEN)
    eps = orientation_bits(w.key, span)
    role = roles(w.key, span, w.n_payload_bits, w.sync_frac)
    want = {int(t): (1 if role[int(t)] < 0 else
                     (1 if ((MESSAGE >> role[int(t)]) & 1) else -1)) for t in span}
    gen = torch.Generator(device=M.device).manual_seed(3000 + i)
    x = torch.full((1, pl + GEN), MASK_ID, dtype=torch.long, device=M.device)
    x[:, :pl] = p.to(M.device)
    for b in range(GEN // BLK):
        lo = pl + b * BLK
        B, g, S, qp, qm = w._table(x.cpu(), lo, pl + GEN)
        car = w._carrier(S)
        gmap = {int(q): g[k] for k, q in enumerate(B)}
        cm = {int(q): bool(car[k]) for k, q in enumerate(B)}
        Bt = torch.tensor(B, device=x.device)
        for t in range(BLK):
            live = Bt[x[0, Bt] == MASK_ID]
            if live.numel() == 0:
                break
            logits = M.model(x).logits[0]
            probs = torch.softmax(logits[live].double() / 0.8, dim=-1)
            conf = probs.max(-1).values
            del logits
            n = int(torch.argmax(conf))
            ipos = int(live[n])
            row = probs[n]
            if cm[ipos]:
                gi = gmap[ipos].to(row.device)
                tgt = eps[ipos] * want[ipos]
                mask_t = (tgt * gi) > 0
                qg = float((row * mask_t).sum())
                qg_all.append(qg)
                # what survives truncation of the LIVE row, renormalised
                sr, si = torch.sort(row, descending=True)
                cs = torch.cumsum(sr, 0)
                for tag, keep in (("p80", cs <= 0.80), ("p90", cs <= 0.90),
                                  ("p95", cs <= 0.95),
                                  ("k10", torch.arange(len(sr), device=row.device) < 10)):
                    keep = keep.clone(); keep[0] = True
                    sub = torch.zeros_like(row)
                    sub[si[keep]] = row[si[keep]]
                    sub = sub / sub.sum()
                    qt = float((sub * mask_t).sum())
                    trunc[tag].append(min(qt, 1 - qt) * 2)   # two-sided mass after trunc
            # commit the model's own draw (no watermark) -- anatomy, not embedding
            x[0, ipos] = int(torch.multinomial(row, 1, generator=gen))
    ids = x[0, pl:pl + GEN].cpu()
    n_special = int((ids >= 126000).sum())
    txt = M.tok.decode(ids, skip_special_tokens=True)
    empty_report.append((i, n_special, len(txt)))
    print(f"[{i}] carriers so far {len(qg_all)}  special-tokens {n_special}  "
          f"text-chars {len(txt)}", flush=True)

q = np.array(qg_all)
print(f"\n== q_gen_target on selected carriers (n={len(q)}) ==")
print("quantiles P10/25/50/75/90:",
      " ".join(f"{np.quantile(q, x):.3f}" for x in (.1, .25, .5, .75, .9)))
for th in (0.05, 0.1, 0.2):
    print(f"P(q_gen < {th}) = {np.mean(q < th):.3f}")
print(f"mean {q.mean():.3f}  MAD-from-0.5 {np.mean(np.abs(q - 0.5)):.3f}")
print("\n== two-sided mass surviving LIVE-row truncation (median / P(mass<0.1)) ==")
for tag, v in trunc.items():
    v = np.array(v)
    print(f"{tag:>4}: median {np.median(v):.3f}   P(<0.1) {np.mean(v < 0.1):.3f}")
print("\n== empty-generation check (doc, special-token count, chars) ==")
for row_ in empty_report:
    print("  ", row_)
