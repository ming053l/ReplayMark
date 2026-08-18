#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/1[47]_" >/dev/null || pgrep -f "exp/22_kgw" >/dev/null; do sleep 30; done
echo "=== 23_blocklocal ==="
$P exp/23_blocklocal.py > logs/23_blocklocal.log 2>&1
echo "=== 23 exit $? ==="
