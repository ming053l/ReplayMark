"""Table 2 MMLU column, multinomial half: all methods, one model per invocation.

200 questions drawn with a fixed shuffle from the full MMLU test split (cais/mmlu),
32 generated tokens (one block), answer = first standalone A-D letter in the decode.
Arms mirror exp/42's per-method-control design plus the Shibboleth pair:
  control  : matched multinomial sampler (ResampleMark s_min=2 -> zero carriers)
  shib     : Shibboleth R=16, kappa=0.05
  kgw_d0 / kgw_d1 : left-to-right, delta 0 / 1
  dg_orig / dg_wm / dg_beam3 : dgMARK repo argmax / multinomial top-3 / 3-beam
Accuracy only; 32-token outputs are too short for meaningful detection columns.
Usage: 40_mmlu.py {llada|dream}
"""
import sys, json, re, time, subprocess, os, glob
sys.path.insert(0, "/ssd2/ming/basinmark")
sys.path.insert(0, "/ssd2/ming/basinmark/baselines/dgmark-watermarking/src")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch, pandas as pd
from basinmark.kgw import kgw_generate
from basinmark.resample import ResampleMark
from generation import WatermarkGenerator

WHICH = sys.argv[1] if len(sys.argv) > 1 else "llada"
if WHICH == "dream":
    from basinmark.dream_model import DreamModel
    M = DreamModel()
else:
    from basinmark.model import BasinModel
    M = BasinModel()

GEN, NQ, KEY = 32, 200, b"retrace-key-A"
PQ = glob.glob("/ssd2/ming/hf_cache/hub/datasets--cais--mmlu/snapshots/*/all/test-*.parquet")[0]
df = pd.read_parquet(PQ)
rng = np.random.default_rng(40)
idx = rng.permutation(len(df))[:NQ]
rows = df.iloc[idx].reset_index(drop=True)
LET = "ABCD"

prompts, golds, pls = [], [], []
for _, r in rows.iterrows():
    q = (r["question"] + "\n" +
         "\n".join(f"{LET[j]}. {c}" for j, c in enumerate(r["choices"])) +
         "\nAnswer with only the letter of the correct option.")
    txt = M.tok.apply_chat_template([{"role": "user", "content": q}],
                                    add_generation_prompt=True, tokenize=False)
    ids = M.tok(txt, return_tensors="pt").input_ids
    prompts.append(ids)
    pls.append(ids.shape[1])
    golds.append(LET[int(r["answer"])])

ANS = re.compile(r"\b([ABCD])\b")
def pick(txt):
    m = ANS.search(txt)
    return m.group(1) if m else None

DG = WatermarkGenerator(M.model, M.tok, "cuda", mask_id=M.mask_id, private_key=None)
BASE = dict(block_len=32, sync_frac=1.0, n_payload_bits=1)

ARMS = [
    ("control", lambda i, p: ResampleMark(M, KEY, nonce=f"mm-{i}", **BASE, s_min=2.0,
                                          retries=1).generate(p, gen_len=GEN, steps=GEN,
                                                              message=0, seed=40000 + i)),
    ("shib", lambda i, p: ResampleMark(M, KEY, nonce=f"mm-{i}", **BASE, s_min=0.5,
                                       retries=16, p_floor=0.05).generate(
                                           p, gen_len=GEN, steps=GEN, message=0,
                                           seed=40000 + i)),
    ("kgw_d0", lambda i, p: kgw_generate(M, p, gen_len=GEN, delta=0.0, key=KEY,
                                         temperature=0.8, seed=40000 + i)),
    ("kgw_d1", lambda i, p: kgw_generate(M, p, gen_len=GEN, delta=1.0, key=KEY,
                                         temperature=0.8, seed=40000 + i)),
    ("dg_orig", lambda i, p: DG.generate_original(p.cuda(), steps=GEN, gen_length=GEN,
                                                  block_length=32).cpu()),
    ("dg_wm", lambda i, p: DG.generate_watermark_multinomial(
        p.cuda(), steps=GEN, gen_length=GEN, block_length=32, top_k=3).cpu()),
    ("dg_beam3", lambda i, p: DG.generate_beam_search(
        p.cuda(), steps=GEN, gen_length=GEN, block_length=32, beam_size=3,
        sampling_strategy="multinomial", top_k=3).cpu()),
]

out = {}
for name, fn in ARMS:
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        y = fn(i, p)
        txt = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
        pred = pick(txt)
        rec.append(dict(pred=pred, ok=int(pred == golds[i])))
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i+1}/{NQ}  acc {np.mean([r['ok'] for r in rec]):.3f}  "
                  f"{(time.time()-t0)/(i+1):.1f}s/q", flush=True)
    out[name] = rec
    print(f"{name:<9}| acc {np.mean([r['ok'] for r in rec]):.3f} "
          f"({sum(r['ok'] for r in rec)}/{NQ})", flush=True)

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, model=WHICH, gen=GEN, n=NQ, idx=idx.tolist(),
               stats={k: v for k, v in out.items()}),
          open(f"/ssd2/ming/basinmark/results/40_mmlu_{WHICH}.json", "w"))
print(f"saved with git {sha[:8]}")
