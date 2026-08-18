#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/1[56]_" >/dev/null; do sleep 20; done
echo "=== 14_fixedkey_null ==="
$P exp/14_fixedkey_null.py > logs/14_fixedkey_null.log 2>&1
echo "=== 14 exit $? ==="
