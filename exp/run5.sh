#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== 05_driven_subset ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/05_driven_subset.py > logs/05_driven.log 2>&1
echo "=== exit $? ==="
