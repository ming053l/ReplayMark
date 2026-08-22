#!/usr/bin/env bash
# Dream-7B-Instruct block: detectability (36), KGW@512 (37), dgMARK@512 (38), GSM8K (39).
cd /ssd2/ming/basinmark
P=/home/ming0531/miniconda3/envs/mmada/bin/python
while ! grep -q "=== KGW512 DONE ===" logs/run35.log 2>/dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/36_dream_detect.py > logs/36_dream.log 2>&1; echo "=== 36 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/37_dream_kgw512.py > logs/37_dream_kgw.log 2>&1; echo "=== 37 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
bash exp/38_dgmark_dream.sh > logs/38_dgdream.log 2>&1; echo "=== 38 gen exit $? ==="
$P exp/38_dgdream_eval.py > logs/38_dgdream_eval.log 2>&1; echo "=== 38 eval exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/39_dream_gsm8k.py > logs/39_dream_gsm8k.log 2>&1; echo "=== 39 exit $? ==="
echo "=== DREAM DONE ==="
