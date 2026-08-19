#!/usr/bin/env bash
# Overnight run, serial. Gate on the GPU being free rather than on pgrep: a pgrep pattern
# also matches the shell that wrote this file, since the pattern appears in its command
# line, and the previous version deadlocked on exactly that.
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
echo "=== [1/3] 23_blocklocal (CCTC gates) ==="
$P exp/23_blocklocal.py > logs/23_blocklocal.log 2>&1; echo "=== 23 exit $? ==="
echo "=== [2/3] 24_sweep (step budget = retry budget) ==="
$P exp/24_blocklocal_sweep.py > logs/24_sweep.log 2>&1; echo "=== 24 exit $? ==="
echo "=== [3/3] 25_final ==="
if [ -f exp/25_final.py ]; then
  $P exp/25_final.py > logs/25_final.log 2>&1; echo "=== 25 exit $? ==="
else
  echo "25_final.py depends on which config survives stage 2; not written yet"
fi
echo "=== OVERNIGHT DONE ==="
