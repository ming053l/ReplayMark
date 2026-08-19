#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== 28_where ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/02_blockmark_where.py > logs/28_where.log 2>&1
echo "=== 28 exit $? ==="
