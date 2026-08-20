#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
$P exp/22_tpr_boost.py > logs/22_boost.log 2>&1; echo "=== 22 exit $? ==="
$P exp/23_floor.py > logs/23_floor.log 2>&1; echo "=== 23 exit $? ==="
$P exp/24_len1024.py > logs/24_len.log 2>&1; echo "=== 24 exit $? ==="
echo "=== NIGHT2 DONE ==="
$P exp/22_tpr_boost.py > logs/22_boost.log 2>&1; echo "=== 22b exit $? ==="
echo "=== NIGHT2 FULLY DONE ==="
$P exp/25_combo.py > logs/25_combo.log 2>&1; echo "=== 25 exit $? ==="
echo "=== NIGHT2 v3 DONE ==="
