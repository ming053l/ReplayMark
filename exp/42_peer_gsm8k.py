"""Peer methods on downstream task quality + detectability: GSM8K under KGW and dgMARK.

Fills the quality table's KGW / dgMARK / dgMARK+3-beam rows (multinomial GSM8K column) and
checks each arm's detectability on task outputs, mirroring exp/32's protocol: 50 GSM8K test
problems, 256 generated tokens, chat template, exact match on the final number.

Each method keeps its own decoding regime and its own control, exactly as in Table 1:
  kgw_d0   : left-to-right, delta=0  (KGW's unwatermarked control)
  kgw_d1   : left-to-right, delta=1
  dg_orig  : dgMARK repo generate_original (argmax confidence decode, no watermark)
  dg_wm    : dgMARK generate_watermark_multinomial, top_k=3
  dg_beam3 : dgMARK generate_beam_search, beam 3, multinomial top_k=3
Detection: KGW deduplicated bigram z; dgMARK parity match ratio -> z=(r-.5)sqrt(n)/.5 on the
EOT-trimmed span. Usage: 42_peer_gsm8k.py {llada|dream}
"""
import sys, json, re, time, subprocess, os
sys.path.insert(0, "/ssd2/ming/basinmark")
sys.path.insert(0, "/ssd2/ming/basinmark/baselines/dgmark-watermarking/src")
os.environ["HF_HOME"] = "/ssd2/ming/hf_cache"
import numpy as np, torch
from basinmark.kgw import kgw_generate, kgw_detect
from generation import WatermarkGenerator
from utils import check_watermark_compliance

WHICH = sys.argv[1] if len(sys.argv) > 1 else "llada"
if WHICH == "dream":
    from basinmark.dream_model import DreamModel
    M = DreamModel()
    VOCAB, EOT = 152064, 151643
else:
    from basinmark.model import BasinModel
    M = BasinModel()
    VOCAB, EOT = 126464, 126081

GEN, NS = 256, 50
DATA = "/ssd2/ming/basinmark/data/gsm8k_test.jsonl"
rows = [json.loads(l) for l in open(DATA)][:NS]

NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
def final_number(s):
    m = NUM.findall(s.replace(",", ""))
    return m[-1].lstrip("$") if m else None

def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")

INSTR = ("\nPlease reason step by step, and end your reply with the final numeric "
         "answer after '####'.")
prompts, pls = [], []
for r in rows:
    txt = M.tok.apply_chat_template([{"role": "user", "content": r["question"] + INSTR}],
                                    add_generation_prompt=True, tokenize=False)
    ids = M.tok(txt, return_tensors="pt").input_ids
    prompts.append(ids)
    pls.append(ids.shape[1])

# dgMARK generator drives the same underlying model (already logits-aligned for Dream).
DG = WatermarkGenerator(M.model, M.tok, "cuda", mask_id=M.mask_id, private_key=None)

def dg_detect(y, p_len):
    ids = y[0, p_len:p_len + GEN].tolist()
    try:
        ids = ids[:ids.index(EOT) + 1]
    except ValueError:
        pass
    hits = sum(check_watermark_compliance(p_len + j + 1, t, None)
               for j, t in enumerate(ids))
    n = len(ids)
    return (hits - n / 2) / np.sqrt(max(n, 1) / 4)

def run_arm(name, fn, detect):
    rec, t0 = [], time.time()
    for i, p in enumerate(prompts):
        y = fn(i, p)
        txt = M.tok.decode(y[0, pls[i]:pls[i] + GEN], skip_special_tokens=True)
        pred = final_number(txt)
        rec.append(dict(ids=y[0].tolist(), z=float(detect(y, pls[i])), pred=pred,
                        ok=int(pred == gold(rows[i]["answer"]))))
        if (i + 1) % 10 == 0:
            print(f"  [{name}] {i+1}/{NS}  acc {np.mean([r['ok'] for r in rec]):.2f}  "
                  f"{(time.time()-t0)/(i+1):.0f}s/doc", flush=True)
    zs = np.array([r["z"] for r in rec])
    print(f"{name:<9}| acc {np.mean([r['ok'] for r in rec]):.3f} "
          f"({sum(r['ok'] for r in rec)}/{NS}) | z {zs.mean():+.2f} | "
          f"TPR@5% {np.mean(zs>1.645):.2f} @1% {np.mean(zs>2.326):.2f} "
          f"@0.1% {np.mean(zs>3.090):.2f}", flush=True)
    return rec

KEY = b"kgw-key"
kdet = lambda y, pl: kgw_detect(M, y, np.arange(pl, pl + GEN), key=KEY, dedup=True,
                                vocab=VOCAB)["z"]
ARMS = [
    ("kgw_d0", lambda i, p: kgw_generate(M, p, gen_len=GEN, delta=0.0, key=KEY,
                                         temperature=0.8, seed=42000 + i), kdet),
    ("kgw_d1", lambda i, p: kgw_generate(M, p, gen_len=GEN, delta=1.0, key=KEY,
                                         temperature=0.8, seed=42000 + i), kdet),
    ("dg_orig", lambda i, p: DG.generate_original(p.cuda(), steps=GEN, gen_length=GEN,
                                                  block_length=32).cpu(),
     lambda y, pl: dg_detect(y, pl)),
    ("dg_wm", lambda i, p: DG.generate_watermark_multinomial(
        p.cuda(), steps=GEN, gen_length=GEN, block_length=32, top_k=3).cpu(),
     lambda y, pl: dg_detect(y, pl)),
    ("dg_beam3", lambda i, p: DG.generate_beam_search(
        p.cuda(), steps=GEN, gen_length=GEN, block_length=32, beam_size=3,
        sampling_strategy="multinomial", top_k=3).cpu(),
     lambda y, pl: dg_detect(y, pl)),
]
out = {}
for name, fn, det in ARMS:
    out[name] = run_arm(name, fn, det)

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     cwd="/ssd2/ming/basinmark").stdout.strip()
json.dump(dict(sha=sha, model=WHICH, gen=GEN, n=NS, pls=pls,
               stats={k: [{kk: vv for kk, vv in r.items() if kk != "ids"} for r in v]
                      for k, v in out.items()},
               ids={k: [r["ids"] for r in v] for k, v in out.items()}),
          open(f"/ssd2/ming/basinmark/results/42_peer_gsm8k_{WHICH}.json", "w"))
print(f"saved with git {sha[:8]}")
