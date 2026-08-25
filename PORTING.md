# Porting basinmark to another server

## What's in the bundle
- `basinmark/` - the method package (model wrappers, ReplayMark, KGW, challenges, data).
- `exp/` — every experiment script (index below).
- `baselines/dgmark-watermarking/` — dgMARK reproduction, LOCALLY PATCHED:
  `scripts/generate.py` gained `--mask_id / --shift_logits / --eot_id` (Dream support).
  Do not re-clone upstream over it.
- `data/` — C4 validation slice, GSM8K test, HumanEval.
- `results/` — all measured results; several scripts READ these as inputs
  (23_floor.json, 29_clean.json) so keep them.
- `DESIGN.md`, `RESUME.md`, `DETECTOR_SURVEY.md` — context.

## 1. Configure paths
The core package accepts model paths directly and reads the following optional variables:
```bash
export HF_HOME=$CACHE
export REPLAYMARK_LLADA_MODEL=GSAI-ML/LLaDA-8B-Instruct
export REPLAYMARK_DREAM_MODEL=Dream-org/Dream-v0-Instruct-7B
export REPLAYMARK_C4=$NEW/data/c4-validation.json.gz
```
The numbered experiment files preserve their original machine paths as run provenance. For a fresh
run, invoke the Python file from the repository root and redirect its output locally, or adapt the
corresponding launcher to your interpreter.

## 2. Environment
python 3.10, torch 2.7.1+cu126, transformers 4.46.2 (Dream's remote code was written
against this; newer transformers may break `trust_remote_code` model loading),
scipy 1.15.3, pandas + pyarrow (MMLU parquet), matplotlib (viz only).

## 3. Models (download into $CACHE)
```bash
HF_HOME=$CACHE hf download GSAI-ML/LLaDA-8B-Instruct     # snapshot 08b83a6f... expected
HF_HOME=$CACHE hf download Dream-org/Dream-v0-Instruct-7B
HF_HOME=$CACHE hf download openai-community/gpt2-large    # PPL scorer
HF_HOME=$CACHE hf download cais/mmlu --repo-type dataset --include "all/test-*"
```
The wrappers use Hugging Face IDs by default and also accept explicit local snapshot paths.
GPU note: the wrappers re-enable mem-efficient SDPA (a TITAN RTX / sm75 workaround);
harmless on newer GPUs. fp16 needs ~17GB for LLaDA-8B at 512+ tokens.

## 4. Sanity checks before burning GPU-hours
```bash
python - <<'EOF'   # Dream wrapper smoke test (CPU, ~1 min)
import sys, torch; sys.path.insert(0, ".")
from basinmark.dream_model import DreamModel, DREAM_MASK_ID
M = DreamModel(dtype=torch.float32, device="cpu")
ids = M.tok("The capital of France is", return_tensors="pt").input_ids
x = torch.cat([ids, torch.full((1,4), DREAM_MASK_ID)], 1)
print(M.tok.decode(M.model(x).logits[0, ids.shape[1]].argmax()))  # expect " Paris"
EOF
```
exp/36 also runs a 64-token generate+detect preflight and asserts before its main arms.

## 5. Experiment index (the reusable ones)
- 23/28/29: ReplayMark detectability arms (LLaDA, 512/1024 tok)
- 30–31: dgMARK@512 arms + eval; 35/37: KGW@512 (LLaDA/Dream)
- 32/39: GSM8K control-vs-watermark (LLaDA/Dream)
- 33/33b/44: block-local detection & editing attacks — DETECTION-ONLY on saved
  outputs in results/, cheap, good first jobs on a new box
- 34: ctx_window=128 arm; 36: Dream detectability; 38: dgMARK@512 Dream
- 40/41/42: MMLU / HumanEval / peer-GSM8K harnesses (both models via argv llada|dream)
- 43: carrier-suitability visualization (CPU-capable); 45: detection wall-clock;
  46: peer re-denoising attack
- Launchers `exp/run*.sh` chain stages by grepping "=== ... ===" markers in logs/ and
  waiting for a free GPU — start them all with
  `setsid nohup ./exp/<runner>.sh > logs/<name>.log 2>&1 < /dev/null &`.
  On a multi-GPU box set CUDA_VISIBLE_DEVICES per chain; the "GPU free" gates assume
  one visible device.

## 6. Keys and protocol invariants
Key `b"retrace-key-A"` with per-doc nonces; KGW key `b"kgw-key"`. 512-token length policy
for all comparisons; per-method controls under each method's own decoding regime; every
table cell in the paper maps to a results/ file — keep that discipline.
