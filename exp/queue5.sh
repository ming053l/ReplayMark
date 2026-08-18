#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/08" >/dev/null || pgrep -f "exp/09" >/dev/null || pgrep -f "exp/10" >/dev/null; do sleep 20; done
echo "=== starting 12_null ==="
$P exp/12_null.py > logs/null.log 2>&1
echo "=== queue5 complete ==="
