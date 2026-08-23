#!/usr/bin/env bash
# Table 2 multinomial half: MMLU both models first (cheaper column), then HumanEval.
cd /ssd2/ming/basinmark
P=/home/ming0531/miniconda3/envs/mmada/bin/python
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/40_mmlu.py llada > logs/40_mmlu_llada.log 2>&1; echo "=== 40L exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/40_mmlu.py dream > logs/40_mmlu_dream.log 2>&1; echo "=== 40D exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/41_humaneval.py llada > logs/41_he_llada.log 2>&1; echo "=== 41L exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/41_humaneval.py dream > logs/41_he_dream.log 2>&1; echo "=== 41D exit $? ==="
echo "=== TABLE2 DONE ==="
