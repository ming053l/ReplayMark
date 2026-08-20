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
/home/ming0531/miniconda3/envs/mmada/bin/python exp/18_detector_ablation.py > logs/18_ablation.log 2>&1
echo "=== 18 exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/19_freeze.py > logs/19_freeze.log 2>&1
echo "=== 19 exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/20_robust.py > logs/20_robust.log 2>&1
echo "=== 20 exit $? ==="
echo "=== OVERNIGHT CHAIN DONE ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/17_llr.py > logs/17_llr.log 2>&1
echo "=== 17b exit $? ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/18_detector_ablation.py > logs/18_ablation.log 2>&1
echo "=== 18b exit $? ==="
echo "=== CHAIN FULLY DONE ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/21_R_matched.py > logs/21_R.log 2>&1
echo "=== 21 exit $? ==="
echo "=== NIGHT COMPLETE ==="
/home/ming0531/miniconda3/envs/mmada/bin/python exp/22_tpr_boost.py > logs/22_boost.log 2>&1
echo "=== 22 exit $? ==="
echo "=== NIGHT COMPLETE v2 ==="
