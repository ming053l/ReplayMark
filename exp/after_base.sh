#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "runbaselines" >/dev/null; do sleep 30; done
echo "=== 18_dgmark_eval ==="
$P exp/18_dgmark_eval.py > logs/18_dgmark_eval.log 2>&1; echo "=== 18 exit $? ==="
for s in 17_capacity 16_pareto 14_fixedkey_null; do
  echo "=== $s ==="; $P "exp/$s.py" > "logs/${s}.log" 2>&1; echo "=== $s exit $? ==="
done
echo "=== ALL DONE ==="
