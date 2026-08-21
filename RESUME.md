# RESUME — state at shutdown (updated 2026-08-21 16:15 CST)

Server may be powered off by the network admin. Chain state and relaunch procedure below.
Paper (NeurIPS template, GPT revision) is pushed to GitHub and synced to Overleaf.

## Chain state

- **exp/29 (1024-tok graduation): DONE, saved** (`results/29_clean.json`, git 8acfc12b).
  Verdict: detection passes, quality flags red —
  `R16k05 | sync 0.640 | TPR@5/1/0.1% = 0.88/0.88/0.88` but
  `ratio 0.428 with repetition 0.530 vs control 0.342` (valid n=9). The sub-1.0 ratio is
  driven by repetitive text at this length; the honest 1024 headline stays R8/kappa=0.1
  (repetition parity). Do NOT quote 29's ratio as a quality win.
- **exp/30 (dgMARK @512, n=30 x 3 arms): RUNNING at shutdown risk.** dgMARK writes CSVs
  at the end of each arm; an interrupted arm must be rerun. Then exp/31 evaluates.
- **exp/32 (GSM8K): queued** behind run30's `=== DG512 DONE ===` marker in logs/run30.log.
- **exp/33/34 (robustness improvements): queued** behind run32's `=== GSM8K DONE ===`
  marker in logs/run32.log. 33 = block-local exact detection (pooled vs Bonferroni-min vs
  Stouffer) under 5%/10% same-model re-denoise, on 29's saved outputs — no new
  generation. 34 = ctx_window=128 windowed-conditioning arm @512 (gen+detect), clean and
  attacked TPR plus paired quality, vs the full-context arm on shared prompts/seeds.

## Relaunch after reboot (in this order)

```bash
cd /ssd1/ming/basinmark
# 29 is done — do not rerun. Start whatever the chain had not finished:
setsid nohup ./exp/run30.sh > logs/run30.log 2>&1 < /dev/null &   # if 30/31 incomplete
setsid nohup ./exp/run32.sh > logs/run32.log 2>&1 < /dev/null &   # waits for 30 marker
setsid nohup ./exp/run33.sh > logs/run33.log 2>&1 < /dev/null &   # waits for 32 marker
```

CAUTION: run30's gate is only "GPU free". If 30 already finished (check
`ls results/baselines/dg512_*`), skip it and hand-write the DONE marker instead:
`echo "=== DG512 DONE ===" >> logs/run30.log` so 32 unblocks.

Progress: `tr '\r' '\n' < logs/<name>.log | grep -vE 'Loading|it/s|s/it' | tail`

## Why 33/34 exist (the practicality plan)

Practicality is asymmetric (embedding cheap, verification expensive+fragile). Two of the
three weaknesses have structural fixes now queued:
1. **Edit fragility from pooling** (33): the pooled count dilutes intact blocks with
   damaged ones; per-block exact tests (Bonferroni-min) should recover detection when
   edits are local. Runs on saved outputs.
2. **Edit fragility from propagation** (34): full-prefix conditioning lets one edit
   perturb every later block's bank; ctx_window=W bounds damage to ~ceil(W/32)+1 blocks.
   Costs: shorter context may weaken the contrast — that trade is what 34 measures.
3. **Detection cost**: unaddressed by these runs; L=4 ablation (5 seqs/block, ~45%
   cheaper detection) is the next candidate if 33/34 land well.

## Paper state

NeurIPS 2026 template, 10 pages, compiles clean. GitHub: through "Adopt GPT revision"
plus this commit. Overleaf 6a8805111903ef804b4e2eae synced (branch main). After
30/31/32/33/34 land: add dg512 same-budget rows, GSM8K downstream row, and (if positive)
the robustness-hardening paragraph with measured numbers.
