#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== driver start ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/driver.py > logs/driver.log 2>&1
echo "=== driver exit $? ==="
