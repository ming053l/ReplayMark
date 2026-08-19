#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
echo "=== 03_table_agreement ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/03_table_agreement.py > logs/03_table_agreement.log 2>&1
echo "=== exit $? ==="
