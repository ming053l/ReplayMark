#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "05[_]carrier" >/dev/null; do sleep 20; done
echo "=== starting 06_e2e_carrier ==="
$P exp/06_e2e_carrier.py > logs/e2e_carrier.log 2>&1
echo "=== starting 04_attacks ==="
$P exp/04_attacks.py > logs/attacks.log 2>&1
echo "=== queue complete ==="
