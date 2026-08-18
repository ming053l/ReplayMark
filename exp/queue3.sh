#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/08" >/dev/null || pgrep -f "exp/07" >/dev/null; do sleep 20; done
echo "=== starting 09_blocks ==="
$P exp/09_blocks.py > logs/blocks.log 2>&1
echo "=== queue3 complete ==="
