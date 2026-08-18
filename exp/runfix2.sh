#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/14_fixedkey" >/dev/null; do sleep 20; done
echo "=== 15_tau ==="; $P exp/15_tau.py > logs/15_tau.log 2>&1; echo "=== 15_tau exit $? ==="
echo "=== DONE. 13_attacks_v2 held until the operating point has acceptable quality. ==="
