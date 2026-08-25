"""Table 2 HumanEval column, multinomial half: all methods, one model per invocation.

All 164 HumanEval problems, 256 generated tokens, pass@1 by executing the official test
against the extracted completion in a subprocess (10 s timeout). Same arms as exp/40.
Extraction: last ```python block containing "def <entry_point>", else first block, else
the raw decode appended to the original prompt. Usage: 41_humaneval.py {llada|dream}
"""
import sys, json, re, time, subprocess, os, tempfile
sys.path.insert(0, "/ssd2/ming/basinmark")
sys.path.insert(0, "/ssd2/ming/basinmark/baselines/dgmark-watermarking/src")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.kgw import kgw_generate
from basinmark.resample import ReplayMark
from generation import WatermarkGenerator

WHICH = sys.argv[1] if len(sys.argv) > 1 else "llada"
if WHICH == "dream":
    from basinmark.dream_model import DreamModel
    M = DreamModel()
else:
    from basinmark.model import BasinModel
    M = BasinModel()

GEN, KEY = 256, b"retrace-key-A"
PY = "/home/ming0531/miniconda3/envs/mmada/bin/python"
tasks = [json.loads(l) for l in open("/ssd2/ming/basinmark/data/HumanEval.jsonl")]

prompts, pls = [], []
for t in tasks:
    q = ("Complete the following Python function. Output the complete function "
         "definition (including any imports it needs) inside a single ```python code "
         "block.\n\n```python\n" + t["prompt"] + "```")
    txt = M.tok.apply_chat_template([{"role": "user", "content": q}],
                                    add_generation_prompt=True, tokenize=False)
    ids = M.tok(txt, return_tensors="pt").input_ids
    prompts.append(ids)
    pls.append(ids.shape[1])

BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
def extract(txt, task):
    blocks = BLOCK.findall(txt)
    named = [b for b in blocks if f"def {task['entry_point']}" in b]
    if named:
        return named[-1]
    if blocks:
        return task["prompt"] + "\n" + blocks[0]
    return task["prompt"] + txt

def passes(code, task):
    prog = code + "\n\n" + task["test"] + f"\ncheck({task['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([PY, path], capture_output=True, timeout=10,
                           cwd=tempfile.gettempdir())
        return int(r.returncode == 0)
    except subprocess.TimeoutExpired:
        return 0
    finally:
        os.unlink(path)

DG = WatermarkGenerator(M.model, M.tok, "cuda", mask_id=M.mask_id, private_key=None)
BASE = dict(block_len=32, sync_frac=1.0, n_payload_bits=1)

ARMS = [
    ("control", lambda i, p: ReplayMark(M, KEY, nonce=f"he-{i}", **BASE, s_min=2.0,
                                          retries=1).generate(p, gen_len=GEN, steps=GEN,
                                                              message=0, seed=41000 + i)),
    ("shib", lambda i, p: ReplayMark(M, KEY, nonce=f"he-{i}", **BASE, s_min=0.5,
                                       retries=16, p_floor=0.05).generate(
                                           p, gen_len=GEN, steps=GEN, message=0,
                                           seed=41000 + i)),
    ("kgw_d0", lambda i, p: kgw_generate(M, p, gen_len=GEN, delta=0.0, key=KEY,
                                         temperature=0.8, seed=41000 + i)),
    ("kgw_d1", lambda i, p: kgw_generate(M, p, gen_len=GEN, delta=1.0, key=KEY,
                                         temperature=0.8, seed=41000 + i)),
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
        ok = passes(extract(txt, tasks[i]), tasks[i])
        rec.append(dict(ok=ok))
        if (i + 1) % 25 == 0:
            print(f"  [{name}] {i+1}/164  pass@1 {np.mean([r['ok'] for r in rec]):.3f}  "
                  f"{(time.time()-t0)/(i+1):.0f}s/task", flush=True)
    out[name] = rec
    print(f"{name:<9}| pass@1 {np.mean([r['ok'] for r in rec]):.3f} "
          f"({sum(r['ok'] for r in rec)}/164)", flush=True)

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, model=WHICH, gen=GEN, n=164,
               stats={k: v for k, v in out.items()}),
          open(f"/ssd2/ming/basinmark/results/41_humaneval_{WHICH}.json", "w"))
print(f"saved with git {sha[:8]}")
