#!/usr/bin/env bash
cd /ssd1/ming/basinmark
# gate: run30.sh must have finished (marker in its nohup log), then GPU free
while ! grep -q "=== DG512 DONE ===" logs/run30.log 2>/dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/32_gsm8k.py > logs/32_gsm8k.log 2>&1
echo "=== 32 exit $? ==="; echo "=== GSM8K DONE ==="
