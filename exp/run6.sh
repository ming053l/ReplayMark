#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== 06_deferral_effect ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/06_deferral_effect.py > logs/06_deferral.log 2>&1
echo "=== exit $? ==="
