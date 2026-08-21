#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
bash exp/30_dgmark512.sh > logs/30_dg512.log 2>&1; echo "=== 30 exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/31_dg512_eval.py > logs/31_dg512eval.log 2>&1
echo "=== 31 exit $? ==="; echo "=== DG512 DONE ==="
