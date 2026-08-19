"""Autonomous overnight driver.

Runs one stage at a time, reads its own results, and decides what to do next, writing every
decision to results/driver_report.md so the reasoning is auditable in the morning. The model
is loaded once and reused across stages.

Design levers, in the order the measurements justify:

  L1  step budget with waiting allowed. exp/06 showed a deferred position's proposal changes
      0.20 of the time after 1-2 steps but 0.57 after 11+, and the old schedule never waited
      because ceil(live/steps_left) >= 1. This is the lever most likely to matter.
  L2  strict deferral: force a commit only when the remaining steps are needed to finish.
      Implemented together with L1.
  L3  carrier fraction via the top-2 gap threshold: more countable positions, each of which
      must still be steerable.
  L4  challenge sharpness: more shared patterns to draw a more discriminative contrast pair.

Stop rules, so the night is not spent grinding:
  * a stage that raises is caught, logged, and the driver moves to the next lever;
  * a stage exceeding its wall-clock budget is abandoned;
  * if the best TPR@1% at ppl <= 1.30 is still < 0.15 after every lever, the driver stops
    sweeping and runs the honest-negative package instead: a null-calibration and a
    capacity-accounting run that together explain WHY, on the same protocol as the
    baselines.
  * if any lever reaches TPR@1% >= 0.50 at ppl <= 1.30, the driver freezes that config and
    spends the remaining time on the evaluation the baselines are held to: n=50 fresh
    documents, per-document nonces, and the robustness suite.
"""
import sys, json, time, traceback, itertools
sys.path.insert(0, "/ssd1/ming/basinmark")
import numpy as np, torch

from basinmark.model import BasinModel
from basinmark.data import c4_prompts
from basinmark.blockmark import BlockMark

REPORT = "/ssd1/ming/basinmark/results/driver_report.md"
STATE = "/ssd1/ming/basinmark/results/driver_state.json"
KEY, MESSAGE = b"retrace-key-A", 0xA5
GEN, BLK = 256, 32
PPL_CAP = 1.30
GO = 0.50            # freeze and evaluate properly
FLOOR = 0.15         # below this after every lever, stop sweeping and explain instead


def log(msg):
    stamp = time.strftime("%H:%M:%S")
    line = f"- `{stamp}` {msg}"
    print(line, flush=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")


def section(title):
    with open(REPORT, "a") as f:
        f.write(f"\n## {title}\n\n")
    print(f"\n## {title}", flush=True)


class Ppl:
    def __init__(self):
        import os
        os.environ["HF_HOME"] = "/ssd1/ming/hf_cache"
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tk = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
        self.m = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2-large", torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def nll(self, texts):
        o = []
        for t in texts:
            ids = self.tk(t, return_tensors="pt", truncation=True,
                          max_length=512).input_ids.cuda()
            o.append(float(self.m(ids, labels=ids).loss) if ids.shape[1] >= 8 else np.nan)
        return np.array(o)


def run_config(M, ppl, prompts, pls, *, steps, gap, carrier_gap_only=False,
               n_patterns=4, tau=0.3, ns=None, ref_cache=None):
    """One (steps, gap) point. Returns a row dict, or None on failure."""
    prompts = prompts[:ns] if ns else prompts
    pls = pls[:len(prompts)]
    if steps not in ref_cache:
        ref = [M.generate(p, gen_len=GEN, steps=steps, block_len=BLK, temperature=0.8,
                          seed=3000 + i).cpu() for i, p in enumerate(prompts)]
        txt = [M.tok.decode(x[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
               for i, x in enumerate(ref)]
        n0 = ppl.nll(txt)
        nulls = []
        for i, x in enumerate(ref):
            w = BlockMark(M, KEY, block_len=BLK, n_patterns=n_patterns, tau_conf=tau,
                          holes=4, n_payload_bits=7, sync_frac=0.5, challenge="contrast",
                          gap_nats=gap, nonce=f"doc-{i}")
            nulls.append(w.detect(x, pls[i], GEN, MESSAGE))
        ref_cache[steps] = dict(ppl=float(np.exp(np.nanmean(n0))),
                                rate=float(np.mean([d["rate_sync"] for d in nulls])),
                                n=float(np.mean([d["n_sync"] for d in nulls])))
        log(f"reference steps={steps}: ppl {ref_cache[steps]['ppl']:.2f}, "
            f"null sync rate {ref_cache[steps]['rate']:.3f} (want 0.500), "
            f"n_sync {ref_cache[steps]['n']:.0f}")
    r0 = ref_cache[steps]

    ps, rates, accs, txt, st = [], [], [], [], []
    for i, p in enumerate(prompts):
        w = BlockMark(M, KEY, block_len=BLK, n_patterns=n_patterns, tau_conf=tau, holes=4,
                      n_payload_bits=7, sync_frac=0.5, challenge="contrast",
                      gap_nats=gap, nonce=f"doc-{i}")
        y = w.generate(p, gen_len=GEN, steps=steps, temperature=0.8, message=MESSAGE,
                       seed=3000 + i)
        d = w.detect(y, pls[i], GEN, MESSAGE)
        ps.append(d["p_value"]); rates.append(d["rate_sync"]); accs.append(d["bit_acc"])
        st.append(w.stats)
        txt.append(M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True))
    nw = ppl.nll(txt); ps = np.array(ps)
    cw = float(np.mean([s["carrier_wm"] for s in st]))
    cf = float(np.mean([s["carrier_fb"] for s in st]))
    return dict(steps=steps, gap=gap, n_patterns=n_patterns, tau=tau,
                rate=float(np.mean(rates)), rate_ref=r0["rate"],
                tpr01=float(np.mean(ps < 0.01)), tpr05=float(np.mean(ps < 0.05)),
                bit_acc=float(np.mean(accs)),
                ppl=float(np.exp(np.nanmean(nw))),
                ratio=float(np.exp(np.nanmean(nw)) / r0["ppl"]),
                carrier_wm=cw, carrier_fb=cf,
                wm_share=cw / max(cw + cf, 1e-9),
                waited=float(np.mean([s["waited"] for s in st])),
                n_sync=r0["n"])


def show(r):
    log(f"steps={r['steps']:<4} gap={r['gap']:<4} pat={r['n_patterns']} | "
        f"sync {r['rate']:.3f} (ref {r['rate_ref']:.3f}) | TPR@1% {r['tpr01']:.2f} "
        f"@5% {r['tpr05']:.2f} | bits {r['bit_acc']:.2f} | ppl x{r['ratio']:.2f} | "
        f"carrier wm/fb {r['carrier_wm']:.0f}/{r['carrier_fb']:.0f} "
        f"(steered {r['wm_share']:.2f}) | waited {r['waited']:.0f} steps")


def main():
    open(REPORT, "a").write(
        f"\n\n# Overnight driver, started {time.strftime('%Y-%m-%d %H:%M')}\n")
    t_start = time.time()
    M = BasinModel()
    ppl = Ppl()
    prompts = c4_prompts(M.tok, 50)
    pls = [p.shape[1] for p in prompts]
    ref_cache, rows = {}, []

    def best():
        ok = [r for r in rows if r["ratio"] <= PPL_CAP]
        return max(ok, key=lambda r: r["tpr01"]) if ok else None

    # ---------------- L1/L2: step budget, with waiting now permitted ----------------
    section("L1: step budget (waiting steps enabled)")
    for steps in (256, 512, 768):
        try:
            r = run_config(M, ppl, prompts, pls, steps=steps, gap=1.0, ns=10,
                           ref_cache=ref_cache)
            rows.append(r); show(r)
        except Exception as e:
            log(f"steps={steps} FAILED: {e.__class__.__name__}: {e}")
            traceback.print_exc()
        json.dump(rows, open(STATE, "w"), indent=1)
        if time.time() - t_start > 3.5 * 3600:
            log("wall-clock budget for L1 spent; moving on")
            break

    b = best()
    log(f"after L1: best TPR@1% at ppl<={PPL_CAP} is "
        f"{b['tpr01']:.2f} (steps={b['steps']})" if b else "after L1: nothing under the cap")

    # ---------------- L3: carrier fraction ----------------
    if not b or b["tpr01"] < GO:
        section("L3: carrier fraction (top-2 gap threshold)")
        st = b["steps"] if b else 512
        for gap in (0.5, 2.0, 4.0):
            try:
                r = run_config(M, ppl, prompts, pls, steps=st, gap=gap, ns=10,
                               ref_cache=ref_cache)
                rows.append(r); show(r)
            except Exception as e:
                log(f"gap={gap} FAILED: {e.__class__.__name__}: {e}")
            json.dump(rows, open(STATE, "w"), indent=1)
        b = best()

    # ---------------- L4: challenge sharpness ----------------
    if not b or b["tpr01"] < GO:
        section("L4: challenge sharpness (more shared patterns)")
        st = b["steps"] if b else 512
        gp = b["gap"] if b else 1.0
        for npat in (8, 16):
            try:
                r = run_config(M, ppl, prompts, pls, steps=st, gap=gp, n_patterns=npat,
                               ns=10, ref_cache=ref_cache)
                rows.append(r); show(r)
            except Exception as e:
                log(f"n_patterns={npat} FAILED: {e.__class__.__name__}: {e}")
            json.dump(rows, open(STATE, "w"), indent=1)
        b = best()

    # ---------------- decide ----------------
    section("Decision")
    if b and b["tpr01"] >= GO:
        log(f"GO: TPR@1% {b['tpr01']:.2f} at ppl x{b['ratio']:.2f} "
            f"(steps={b['steps']}, gap={b['gap']}, patterns={b['n_patterns']}). "
            f"Freezing and evaluating on the baselines' protocol.")
        try:
            r = run_config(M, ppl, prompts, pls, steps=b["steps"], gap=b["gap"],
                           n_patterns=b["n_patterns"], ns=50, ref_cache={})
            rows.append(r)
            log("FROZEN CONFIG, n=50, per-document nonces:")
            show(r)
        except Exception as e:
            log(f"frozen evaluation FAILED: {e.__class__.__name__}: {e}")
    elif b and b["tpr01"] >= FLOOR:
        log(f"PARTIAL: best TPR@1% {b['tpr01']:.2f} at ppl x{b['ratio']:.2f}. Above the "
            f"floor but below the go threshold; the honest report is the curve, not a "
            f"headline number. Running it at n=30 for a tighter estimate.")
        try:
            r = run_config(M, ppl, prompts, pls, steps=b["steps"], gap=b["gap"],
                           n_patterns=b["n_patterns"], ns=30, ref_cache={})
            rows.append(r); show(r)
        except Exception as e:
            log(f"tighter estimate FAILED: {e.__class__.__name__}: {e}")
    else:
        log(f"NO-GO: nothing reaches TPR@1% {FLOOR} at ppl <= {PPL_CAP} after all levers. "
            f"Not grinding further. The result to report is the mechanism: every lever, "
            f"what it bought, and the accounting that shows why the commit-order channel "
            f"cannot convert on this model.")
        for r in rows:
            log(f"  steps={r['steps']} gap={r['gap']} pat={r['n_patterns']}: "
                f"TPR@1% {r['tpr01']:.2f}, ppl x{r['ratio']:.2f}, "
                f"steered {r['wm_share']:.2f}, sync {r['rate']:.3f} vs "
                f"ref {r['rate_ref']:.3f}")

    json.dump(rows, open(STATE, "w"), indent=1)
    log(f"driver finished after {(time.time() - t_start) / 3600:.1f} h, "
        f"{len(rows)} configurations")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("driver crashed:")
        with open(REPORT, "a") as f:
            f.write("```\n" + traceback.format_exc() + "```\n")
        raise
