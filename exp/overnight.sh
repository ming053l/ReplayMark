#!/usr/bin/env bash
# Overnight. Gate on the GPU being free -- a pgrep pattern also matches the shell that
# wrote this file, which deadlocked an earlier version.
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
echo "=== [1/2] 27_countmark (bounded-indicator statistic) ==="
$P exp/01_blockmark.py > logs/27_countmark.log 2>&1; echo "=== 27 exit $? ==="
echo "=== [2/2] 24_sweep (old averaging statistic, for the record) ==="
$P exp/24_blocklocal_sweep.py > logs/24_sweep.log 2>&1; echo "=== 24 exit $? ==="
echo "=== OVERNIGHT DONE ==="
