#!/usr/bin/env bash
# One serial runner. Everything now shares basinmark/data.py's prompt construction
# (C4 documents truncated to 300 chars, dgMARK's published protocol), the same LLaDA
# checkpoint, 256-token generations and GPT-2-large for perplexity.
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while pgrep -f "runbaselines" >/dev/null || pgrep -f "dgmark_beam" >/dev/null; do sleep 30; done
for s in 18_dgmark_eval 22_kgw 21_challenge 20_gentime 16_pareto 17_capacity 14_fixedkey_null; do
  echo "=== $s ==="; $P "exp/$s.py" > "logs/${s}.log" 2>&1; echo "=== $s exit $? ==="
done
echo "=== ALL DONE ==="
