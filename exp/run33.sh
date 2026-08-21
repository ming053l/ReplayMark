#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while ! grep -q "=== GSM8K DONE ===" logs/run32.log 2>/dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/33_robust_detector.py > logs/33_robust.log 2>&1
echo "=== 33 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/34_window.py > logs/34_window.log 2>&1
echo "=== 34 exit $? ==="; echo "=== ROBUST DONE ==="
