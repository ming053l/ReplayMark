#!/usr/bin/env bash
# Post-Table-2 extras: 44 sparse survival (detection-only), 45 wall-clock, 46 peer re-denoise.
cd /ssd2/ming/basinmark
P=/home/ming0531/miniconda3/envs/mmada/bin/python
while ! grep -q "=== TABLE2 DONE ===" logs/runq.log 2>/dev/null; do sleep 300; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/44_sparse_survival.py > logs/44_sparse.log 2>&1; echo "=== 44 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/45_timing.py > logs/45_timing.log 2>&1; echo "=== 45 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/46_peer_redenoise.py > logs/46_peer.log 2>&1; echo "=== 46 exit $? ==="
echo "=== EXTRAS DONE ==="
