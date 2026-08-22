#!/usr/bin/env bash
cd /ssd2/ming/basinmark
while ! grep -q "=== DREAM DONE ===" logs/runcp.log 2>/dev/null; do sleep 120; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/33b_localized.py > logs/33b_localized.log 2>&1
echo "=== 33b exit $? ==="
