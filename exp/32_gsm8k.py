"""Downstream task preservation: GSM8K accuracy under the watermark (dgMARK Table-2 axis).

dgMARK's project page argues quality with downstream tasks (MMLU/GSM8K/HumanEval), not
perplexity alone. The same axis matters more for ReTrace: its measured perplexity is at or
below the matched control, and a task metric checks that the sub-control perplexity is not
purchased with degenerate (easy, repetitive) text that happens to score well under GPT-2.

Protocol: LLaDA-8B-Instruct chat template, zero-shot with an answer-format instruction,
50 GSM8K test problems, 256 generated tokens, steps = gen_length; arms are the matched
control (s_min=2 -> zero carriers, same scheduler) and the frozen R16/kappa=0.05 config.
Accuracy = exact match on the final number. Detection is run on both arms: the watermark
arm should detect (short docs -> fewer carriers, so TPR here is not the headline number),
the control arm must sit at the null."""
import sys, json, re, time, subprocess, os, urllib.request
sys.path.insert(0, "/ssd2/ming/basinmark")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.model import BasinModel
from basinmark.resample import ResampleMark

KEY, GEN, BLK, NS = b"retrace-key-A", 256, 32, 50
DATA = "/ssd2/ming/basinmark/data/gsm8k_test.jsonl"
URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
       "grade_school_math/data/test.jsonl")
if not os.path.exists(DATA):
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    urllib.request.urlretrieve(URL, DATA)
rows = [json.loads(l) for l in open(DATA)][:NS]

NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
def final_number(s):
    m = NUM.findall(s.replace(",", ""))
    return m[-1].lstrip("$") if m else None

def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")

M = BasinModel()
INSTR = ("\nPlease reason step by step, and end your reply with the final numeric "
         "answer after '####'.")
prompts, pls = [], []
for r in rows:
    txt = M.tok.apply_chat_template([{"role": "user", "content": r["question"] + INSTR}],
                                    add_generation_prompt=True, tokenize=False)
    ids = M.tok(txt, return_tensors="pt").input_ids
    prompts.append(ids)
    pls.append(ids.shape[1])

BASE = dict(block_len=BLK, sync_frac=1.0, n_payload_bits=1)
ARMS = [("control", dict(s_min=2.0, retries=1)),
        ("R16k05", dict(s_min=0.5, retries=16, p_floor=0.05))]
out = {}
for name, kw in ARMS:
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        w = ResampleMark(M, KEY, nonce=f"gsm-{i}", **BASE, **kw)
        y = w.generate(p, gen_len=GEN, steps=GEN, message=0, seed=32000 + i)
        d = ResampleMark(M, KEY, nonce=f"gsm-{i}", **BASE, s_min=0.5,
                         retries=1).detect(y, pls[i], GEN, 0)
        txt = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
        pred = final_number(txt)
        rec.append(dict(ids=y[0].tolist(), p=d["p_value"], rate=d["rate_sync"],
                        pred=pred, ok=int(pred == gold(rows[i]["answer"]))))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{NS}  acc so far "
                  f"{np.mean([r['ok'] for r in rec]):.2f}  "
                  f"{(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    out[name] = rec
    ps = np.array([r["p"] for r in rec])
    print(f"{name:<8} | acc {np.mean([r['ok'] for r in rec]):.3f} "
          f"({sum(r['ok'] for r in rec)}/{NS}) | sync "
          f"{np.mean([r['rate'] for r in rec]):.3f} | TPR@5% {np.mean(ps<.05):.2f} "
          f"@1% {np.mean(ps<.01):.2f} @0.1% {np.mean(ps<.001):.2f}", flush=True)
# paired comparison: same problems, so report flips both ways, not just the delta
a = np.array([r["ok"] for r in out["control"]])
b = np.array([r["ok"] for r in out["R16k05"]])
print(f"paired: control {a.mean():.3f} vs wm {b.mean():.3f} | "
      f"wm-only-wrong {int(((a == 1) & (b == 0)).sum())} "
      f"wm-only-right {int(((a == 0) & (b == 1)).sum())}", flush=True)
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, gen=GEN, n=NS, pls=pls,
               ids={k: [r["ids"] for r in v] for k, v in out.items()},
               stats={k: [{kk: vv for kk, vv in r.items() if kk != "ids"} for r in v]
                      for k, v in out.items()}),
          open("/ssd2/ming/basinmark/results/32_gsm8k.json", "w"))
print(f"saved with git {sha[:8]}")
