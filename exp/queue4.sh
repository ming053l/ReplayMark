#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/08" >/dev/null || pgrep -f "exp/09" >/dev/null; do sleep 20; done
echo "=== starting 10_attacks_shared ==="
$P exp/10_attacks_shared.py > logs/attacks_shared.log 2>&1
echo "=== queue4 complete ==="
