#!/usr/bin/env bash
# CP-ordered relaunch (2026-08-22 evening): cheapest-per-cell first.
# 33 (detector-only, fills robustness rows) -> dg512 beam + eval (completes dgMARK@512)
# -> 35 KGW@512 (completes LLaDA 512 lock) -> 32 GSM8K (first quality cells)
# -> 34 ctx-window arm -> Dream block 36/37/38/39.
cd /ssd2/ming/basinmark
P=/home/ming0531/miniconda3/envs/mmada/bin/python
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/33_robust_detector.py > logs/33_robust.log 2>&1; echo "=== 33 exit $? ==="
bash exp/30_dgmark512.sh > logs/30_dg512_beam.log 2>&1; echo "=== 30beam exit $? ==="
$P exp/31_dg512_eval.py > logs/31_dg512eval.log 2>&1; echo "=== 31 exit $? ==="
echo "=== DG512 DONE ==="
$P exp/35_kgw512.py > logs/35_kgw512.log 2>&1; echo "=== 35 exit $? ==="
echo "=== KGW512 DONE ==="
$P exp/32_gsm8k.py > logs/32_gsm8k.log 2>&1; echo "=== 32 exit $? ==="
echo "=== GSM8K DONE ==="
$P exp/34_window.py > logs/34_window.log 2>&1; echo "=== 34 exit $? ==="
echo "=== ROBUST DONE ==="
$P exp/36_dream_detect.py > logs/36_dream.log 2>&1; echo "=== 36 exit $? ==="
$P exp/37_dream_kgw512.py > logs/37_dream_kgw.log 2>&1; echo "=== 37 exit $? ==="
bash exp/38_dgmark_dream.sh > logs/38_dgdream.log 2>&1; echo "=== 38 gen exit $? ==="
$P exp/38_dgdream_eval.py > logs/38_dgdream_eval.log 2>&1; echo "=== 38 eval exit $? ==="
$P exp/39_dream_gsm8k.py > logs/39_dream_gsm8k.log 2>&1; echo "=== 39 exit $? ==="
echo "=== DREAM DONE ==="
