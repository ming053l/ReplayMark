#!/usr/bin/env bash
P=/home/ming0531/miniconda3/envs/mmada/bin/python
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
echo "=== 27 rerun with float64 log_softmax ==="
$P exp/27_countmark.py > logs/27b_f64.log 2>&1; echo "=== 27b exit $? ==="
echo "=== F64 RUN DONE ==="
