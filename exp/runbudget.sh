#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/10_budget.py > logs/10_budget.log 2>&1
echo "=== budget exit $? ==="
