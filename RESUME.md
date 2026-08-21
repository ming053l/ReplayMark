# RESUME — state at shutdown (2026-08-21 ~16:06 CST)

Server is being powered off by the network admin. This file records exactly what was
running and how to restart the chain. Everything else (paper, code, results to date) is
committed and pushed to GitHub; the Overleaf project is synced to the same sources.

## What was running

**exp/29_clean.py** (clean graduation run, 1024 tokens, skip=1400, src_min=700, n=16,
per-doc nonces `g2-{i}`, seed 13000+i):

- **control arm: FINISHED.** Log-recorded result (JSON was NOT written — 29 saves only
  at the very end, so these numbers survive only in `logs/29_clean.log` and here):
  `control | sync 0.481 | TPR@5% 0.06 @1% 0.00 @0.1% 0.00`  (null clean; the
  src_min=700 prompt fix works — no repeat of the degenerate-prompt 0.466 pool).
- **R16k05 arm (s_min=0.5, retries=16, p_floor=0.05): ~10–12/16 done at 486 s/doc,
  WILL BE LOST at shutdown.** Nothing is persisted per-doc.

**Verdict for restart: rerun exp/29_clean.py from scratch** (~4.5 h). The control arm's
token ids live only in process memory, and the quality pairing needs them, so partial
reuse is not possible. The control numbers above are a cross-check that the rerun's
control arm should reproduce (same seeds/nonces → identical prompts and generations).

## Queue behind it (never started, nothing lost)

1. **exp/run30.sh** → `exp/30_dgmark512.sh` (dgMARK original/k=1/3-beam @512 tok, n=30,
   same C4 file → `results/baselines/dg512_*.csv`) then `exp/31_dg512_eval.py`
   (same-axes evaluator → prints TPR@5/1/0.1% + ppl ratio). Gate: GPU free.
   Writes `=== DG512 DONE ===` marker into `logs/run30.log`.
2. **exp/run32.sh** → `exp/32_gsm8k.py` (GSM8K downstream task, n=50, 256 tok,
   control vs R16k05, accuracy + detection; dataset already at
   `data/gsm8k_test.jsonl`). Gate: waits for the DONE marker in `logs/run30.log`,
   then GPU free.

## Restart procedure (copy-paste)

```bash
cd /ssd1/ming/basinmark
# 1) graduation run first (holds GPU ~4.5 h)
setsid nohup ./exp/run29.sh > logs/run29.log 2>&1 < /dev/null &
# 2) dgMARK@512 chain — waits for GPU to free, so safe to launch immediately
setsid nohup ./exp/run30.sh > logs/run30.log 2>&1 < /dev/null &
# 3) GSM8K — waits for run30's DONE marker, so safe to launch immediately
setsid nohup ./exp/run32.sh > logs/run32.log 2>&1 < /dev/null &
```

CAUTION: `run30.sh`'s gate is only "GPU free", so do NOT start it without 29 already
running (or it will jump the queue). Launch order above is correct: start 29 first,
confirm `nvidia-smi` shows the python process, then launch 30 and 32.

Progress check: `tr '\r' '\n' < logs/29_clean.log | grep -vE 'Loading|it/s|s/it' | tail`

## What the chain is for

- 29 → the 1024-token confirmatory row (target: TPR@1% ≥ 0.85 at sub-1.0× ratio).
- 30/31 → the same-budget, length-matched dgMARK comparison the paper's Table 1
  caption currently apologizes for lacking (512 vs 256 mismatch).
- 32 → the downstream-task (GSM8K) quality axis for the Discussion/Experiments.

## Paper state

- NeurIPS 2026 template (GPT revision, reviewed: all quoted numbers verified against
  the measurement record). Compiles 10 pages, 0 errors.
- GitHub: pushed through commit "Adopt GPT revision: NeurIPS 2026 template...".
- Overleaf project 6a8805111903ef804b4e2eae: synced (branch `main`) with full sources
  (main.tex, neurips_2026.sty, main.bib, sections/, tables/, figures/*.pdf).

## After results land

Update `tables/baseline_table.tex` (29's 1024 row + dg512 rows at matched length),
re-check the caption's length-mismatch note, rerun robustness at the frozen R16k05
config on saved outputs, and push GitHub + Overleaf again.
