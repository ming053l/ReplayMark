#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== 01_blockmark with steerable carrier ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/01_blockmark.py > logs/01_blockmark.log 2>&1
echo "=== exit $? ==="
