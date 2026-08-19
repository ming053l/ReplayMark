#!/usr/bin/env bash
cd /ssd1/ming/basinmark
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 20; done
/home/ming0531/miniconda3/envs/mmada/bin/python exp/14_presence_only.py > logs/14_presence.log 2>&1
echo "=== 14 exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/15_length.py > logs/15_length.log 2>&1
echo "=== 15 exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/16_decisive.py > logs/16_decisive.log 2>&1
echo "=== 16 exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/17_llr.py > logs/17_llr.log 2>&1
echo "=== 17 exit $? ==="
