#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/0[89]" >/dev/null || pgrep -f "exp/1[02]" >/dev/null; do sleep 20; done
echo "=== starting 11_tradeoff ==="
$P exp/11_tradeoff.py > logs/tradeoff.log 2>&1
echo "=== queue6 complete ==="
