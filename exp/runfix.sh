#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/11_trade" >/dev/null; do sleep 20; done
for s in 14_fixedkey_null 13_attacks_v2; do
  echo "=== $s ==="; $P "exp/$s.py" > "logs/${s}.log" 2>&1; echo "=== $s exit $? ==="
done
echo "=== FIXES DONE ==="
