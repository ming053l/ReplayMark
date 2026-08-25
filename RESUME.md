# RESUME — 2026-08-25: ALL local experiment chains COMPLETE

Everything measured is in the paper (Overleaf 93c3649 / GitHub 6eb0122; main text 8 pages).
Landed since the last note: Table 2 multinomial half complete BOTH models (Shibboleth ties
its own control in all six quality cells; dgMARK loses 0.15-0.28 on every Dream benchmark);
exp/44 sparse survival (pooled 0.81@1% at 50% in-place destruction; pooled=bonf 0.50 at 75%
— crossover only at extremes); exp/45 wall-clock (Shibboleth 26.8 s/doc vs KGW 0.17 s vs
dgMARK <1 ms); exp/46 peer re-denoise HONEST NEGATIVE (10% re-denoise: KGW 0.90->0.85,
dgMARK 0.85->0.80 — the attack is hard for prefix-synced replay specifically); exp/48
carrier stats (mean |P| 192-271, spread 30-500 from EOS padding; per-doc Wilson CIs in
Appendix). Carrier-map figure promoted into the method section by the user.

STILL OPEN: greedy-Shibboleth mode + Table 2 greedy cells, empirical-FPR curve, paraphrase
attack — assigned to server B (branch serverB; its exp/47 L-sweep numbers are in the paper
but results/47_*.json has NOT been pushed yet — reconcile when it lands).

# RESUME — 2026-08-23 evening: ALL SCHEDULED RUNS LANDED except exp/33b (in flight)

Everything measured is in the paper (Overleaf 6eca03d / GitHub 5518bf3). Landed today:
exp/33 (block-local: pooled wins under random edits), dg512 beam+eval, exp/35 KGW@512,
exp/32 GSM8K LLaDA, exp/34 (window free on clean, no gain under random rd10), exp/36 Dream
Shibboleth (R16k05 .90/.90), exp/37 Dream KGW (1.84x .57/.37), exp/38 Dream dgMARK (k=1
1.75x .37/.27; beam 0.71x .30/.30), exp/42 peer GSM8K both models (dgMARK -0.28 acc on
Dream; KGW barely embeds on tasks), exp/39 Dream Shibboleth GSM8K (0.680 vs 0.660 tie).
Headline: cross-model inversion — on Dream, Shibboleth is the strongest detector AND the
only no-task-cost watermark, under shared (LLaDA-tuned) settings for every method.

- exp/33b LANDED 2026-08-24: contiguous 10%/30% re-denoise costs NOTHING (0.88 = clean at
  both operating points; forward-only propagation spares all upstream carriers). Pooled
  beats Bonferroni in every measured regime; copy-paste sparse survival is the only regime
  left untested. PEER CHAIN FULLY DONE (=== PEER CHAIN DONE === in logs/runpeer.log).
- NEXT BUILD: exp/40 MMLU + exp/41 HumanEval harnesses (user approved; several GPU-days)
  to fill the remaining quality-table columns. GSM8K column is COMPLETE both models.


# (superseded) RESUME — 2026-08-22 evening: runcp.sh RUNNING (CP-ordered chain)

After the reboot the chain was relaunched as ONE sequential script, `exp/runcp.sh`
(`logs/runcp.log`), cheapest-per-cell first — the old five-runner gate structure is
superseded:
1. exp/33 block-local detector (detection-only, robustness rows)  → results/33_robust.json
2. dg512 beam arm only (original+watermark already banked) + exp/31 eval
3. exp/35 KGW@512 → results/kgw512.json  (completes the LLaDA 512 lock)
4. exp/32 GSM8K LLaDA → results/32_gsm8k.json
5. exp/34 ctx_window=128 arm → results/34_window.json
6. Dream block: exp/36 → 37 → 38(gen+eval) → 39
If interrupted again: check which "=== NN exit ===" markers are in logs/runcp.log, comment
out the finished stages in exp/runcp.sh, and relaunch it the same way.

## (superseded) Relaunch plan written at the 15:30 shutdown

Verified at shutdown (15:15): dg512 **original COMPLETE (30 rows)**, **watermark COMPLETE
(30 rows)**, beam arm interrupted partway (partial CSV exists — delete
`results/baselines/dg512_beam3*` and rerun it). Edit `exp/30_dgmark512.sh`: delete the whole
`for M in original watermark` loop (both done), keep ONLY the beam command. Then:
```bash
cd /ssd2/ming/basinmark
setsid nohup ./exp/run30.sh > logs/run30.log 2>&1 < /dev/null &
setsid nohup ./exp/run32.sh > logs/run32.log 2>&1 < /dev/null &
setsid nohup ./exp/run33.sh > logs/run33.log 2>&1 < /dev/null &
setsid nohup ./exp/run35.sh > logs/run35.log 2>&1 < /dev/null &
setsid nohup ./exp/run36.sh > logs/run36.log 2>&1 < /dev/null &
```
(Each gates on the previous stage's DONE marker + free GPU; safe to start all five.)
NOTE: exp/31's eval needs ALL THREE dg512 CSVs; run30.sh runs it automatically at the end.

## Next build task (user-approved 2026-08-22): exp/40 MMLU + exp/41 HumanEval harness

Quality table is now the full dgMARK-style grid (all \tbd). Needs harnesses for MMLU and
HumanEval plus task-prompt drivers for KGW/dgMARK arms and a greedy-reference Shibboleth mode
(argmax reference decoder, carriers still draw R proposals from the live conditional).
Several GPU-days; queue after run36. GSM8K cells come free from exp/32 (LLaDA) / exp/39 (Dream).

# Earlier state (2026-08-22 morning, post-reboot relaunch)

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
