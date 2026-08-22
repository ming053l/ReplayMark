#!/usr/bin/env bash
cd /ssd2/ming/basinmark
# gate: run33.sh (exp 33+34) must be done, then GPU free
while ! grep -q "=== ROBUST DONE ===" logs/run33.log 2>/dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/35_kgw512.py > logs/35_kgw512.log 2>&1
echo "=== 35 exit $? ==="; echo "=== KGW512 DONE ==="
