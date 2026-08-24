#!/usr/bin/env bash
cd /ssd2/ming/basinmark
P=/home/ming0531/miniconda3/envs/mmada/bin/python
while ! grep -q "=== EXTRAS DONE ===" logs/runx.log 2>/dev/null; do sleep 300; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/48_carrier_stats.py > logs/48_carriers.log 2>&1; echo "=== 48 exit $? ==="
echo "=== CARRIERS DONE ==="
