#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/1[45]_" >/dev/null; do sleep 20; done
echo "=== 16_pareto ==="; $P exp/16_pareto.py > logs/16_pareto.log 2>&1
echo "=== 16_pareto exit $? ==="
