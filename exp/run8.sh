#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== 08_resample_mvp ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/08_resample_mvp.py > logs/08_resample_mvp.log 2>&1
echo "=== exit $? ==="
