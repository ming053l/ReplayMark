#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "runbaselines" >/dev/null || pgrep -f "dgmark_beam" >/dev/null; do sleep 30; done
# 21 is cheap and decides whether content-selected challenges are worth building;
# 20's lam=0 control decides whether global two-phase generation is viable at all.
for s in 18_dgmark_eval 21_challenge 20_gentime 16_pareto 17_capacity 14_fixedkey_null; do
  echo "=== $s ==="; $P "exp/$s.py" > "logs/${s}.log" 2>&1; echo "=== $s exit $? ==="
done
echo "=== ALL DONE ==="
