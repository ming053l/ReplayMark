#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "07[_]tune_carrier" >/dev/null; do sleep 20; done
echo "=== starting 08_shared ==="
$P exp/08_shared.py > logs/shared.log 2>&1
echo "=== queue2 complete ==="
