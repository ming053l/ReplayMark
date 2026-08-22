# RESUME — state as of 2026-08-22 (post-reboot, chain relaunched)

Repo and hf_cache now live under **/ssd2/ming** (moved from /ssd1 after the admin reboot).
Chain scripts for exp/30–34 and basinmark/{data,model}.py were rewritten to /ssd2; other
exp/ scripts still hardcode /ssd1 and will fail if rerun as-is.

## Chain state

- **exp/29 (1024-tok graduation): DONE, saved** (`results/29_clean.json`, git 8acfc12b).
  Verdict unchanged: detection passes, quality flags red at 1024; the honest 1024 headline
  stays R8/kappa=0.1. Do NOT quote 29's ratio as a quality win.
- **exp/30 (dgMARK @512, n=30 x 3 arms): RELAUNCHED 2026-08-22 ~13:30** (`logs/run30.log`,
  arm log `logs/30_dg512.log`). The pre-shutdown partial CSV (12/30 rows) is being
  overwritten. ~90 s/sample → roughly 45 min/arm plus model load.
- **exp/31 (dg512 eval): runs inside run30.sh** after the arms; prints ppl ratio + TPR rows.
- **exp/32 (GSM8K, 50 problems, 256 tok, control vs watermark): queued** behind
  `=== DG512 DONE ===` in logs/run30.log. Writes `results/32_gsm8k.json`.
- **exp/33 (block-local exact detection) + exp/34 (ctx_window=128): queued** behind
  `=== GSM8K DONE ===` in logs/run32.log (both inside run33.sh).
- **exp/35 (KGW @512, deltas 0/1, n=30): queued** behind `=== ROBUST DONE ===` in
  logs/run33.log (run35.sh). Writes `results/kgw512.json`.
- **exp/36–39 (Dream-7B-Instruct block, user request 2026-08-22): queued** behind
  `=== KGW512 DONE ===` in logs/run35.log (run36.sh):
  36 = Shibboleth detectability (control/R1/R8k10/R16k05 @512; preflight asserts the port
  before burning hours) → `results/36_dream.json`;
  37 = KGW@512 on Dream (vocab 152064) → `results/kgw512_dream.json`;
  38 = dgMARK@512 on Dream (3 arms; repo patched with --mask_id/--shift_logits/--eot_id)
  → `results/baselines/dgdream_*.csv` + eval in logs/38_dgdream_eval.log;
  39 = Dream GSM8K (control vs R16k05) → `results/39_dream_gsm8k.json`.
  Dream port: `basinmark/dream_model.py` (mask 151666, SHIFTED logits — Dream reads
  position i from raw logits i-1); ResampleMark/kgw/BasinModel now use `model.mask_id`.
  CPU smoke test passed (shift verified, "capital of France → Paris" at first mask).

## LENGTH POLICY (user, 2026-08-22)

All method comparisons are locked to **512 tokens** from now on. The 256-token KGW/dgMARK
rows in the baseline table are interim: when exp/31 (dgMARK@512) and exp/35 (KGW@512) land,
REPLACE the 256-tok rows (do not keep both), drop the dual-length caveats from the caption
and the setup paragraph, and re-bold columns. Any future baseline must be run at 512.

Progress: `tr '\r' '\n' < logs/<name>.log | grep -vE 'Loading|it/s|s/it' | tail`

## Paper state — canonical is the NEW Overleaf project

- **Overleaf project 6a8932298ef42d96245d3b28** (GPT-revised draft) is canonical; the old
  project in earlier RESUME notes is superseded. Method renamed **ReTrace → Shibboleth**;
  title "Shibboleth: Probe-and-Replay Model-Response Watermarking for Diffusion Language
  Models", NeurIPS 2026 workshop (Foundations of Language Model Security) template.
- **Fabricated numbers were removed 2026-08-22**: the GPT draft had invented a full
  Dream-7B-Instruct results block and an MMLU/GSM8K/HumanEval quality grid never run
  locally. Deleted with do-not-restore comments; quality_table.tex is now a GSM8K stub
  matching exp/32's actual design. Every remaining number was checked against results/.
- Local `paper/` mirrors the corrected Overleaf tree (compiles: 10 pages).
  `sections/discussion.tex` is orphaned (GPT draft merged it into the conclusion) but kept
  for its practicality-asymmetry text.
- After 30/31 land: fill the two commented dg512 rows in `tables/baseline_table.tex`.
  After 32: fill `tables/quality_table.tex` and un-comment its \input + the downstream
  paragraph in experiments.tex. After 33/34: robustness-hardening paragraph with measured
  numbers; robustness_table.tex is still held back (mostly \tbd).

## Improvement roadmap (user-approved direction)

From the vNext plan: (1) UBRG — KL-budgeted response guidance replacing fixed R;
(2) probe-count sweep L=2/4/8 (L=4 cuts detection to 16*5=80 evals); (3) batched
verification; (4) multi-view response routing; (5) content anchors + local replay (exp/33/34
are the first steps). Robustness table design to follow the detector-design survey.
