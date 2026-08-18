#!/usr/bin/env bash
# Single serial runner: one 8B model at a time on the one 24 GB card.
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "exp/09_blocks" >/dev/null; do sleep 20; done
for job in 09_blocks:blocks 10_attacks_shared:attacks_shared 12_null:null 11_tradeoff:tradeoff; do
  s=${job%%:*}; l=${job##*:}
  [ -s "logs/$l.log" ] && grep -q "=====" "logs/$l.log" && { echo "skip $s"; continue; }
  echo "=== $s ==="
  $P "exp/$s.py" > "logs/$l.log" 2>&1
  echo "=== $s exit $? ==="
done
echo "=== ALL DONE ==="
