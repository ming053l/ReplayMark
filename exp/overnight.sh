#!/usr/bin/env bash
# Overnight run. One serial queue; every stage writes its own log and JSON so a crash in
# one stage does not lose the others. Protocol is fixed by paper/tables/baseline_table.tex:
# LLaDA-8B-Instruct fp16, C4 prompts truncated to 300 chars, 256 generated tokens,
# GPT-2-large perplexity against a same-regime control, TPR at Hoeffding thresholds.
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/22_kgw" >/dev/null; do sleep 30; done
echo "=== [1/3] 23_blocklocal (CCTC gates) ==="
$P exp/23_blocklocal.py > logs/23_blocklocal.log 2>&1; echo "=== 23 exit $? ==="
echo "=== [2/3] 24_blocklocal_sweep (step budget = retry budget) ==="
$P exp/24_blocklocal_sweep.py > logs/24_sweep.log 2>&1; echo "=== 24 exit $? ==="
echo "=== [3/3] 25_final (frozen config, n=50, attacks) ==="
if [ -f exp/25_final.py ]; then
  $P exp/25_final.py > logs/25_final.log 2>&1; echo "=== 25 exit $? ==="
else
  echo "25_final.py not written yet -- depends on which config survives stage 2"
fi
echo "=== OVERNIGHT DONE ==="
