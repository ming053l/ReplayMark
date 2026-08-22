#!/usr/bin/env bash
# Peer-first reorder (user, 2026-08-23): peers on quality + detectability before the rest.
# Waits for the in-flight exp/36 (Dream Shibboleth detectability) to finish, then:
#   42 llada  : KGW + dgMARK on GSM8K, LLaDA   (peer quality)
#   37        : KGW@512 on Dream               (peer detectability)
#   38        : dgMARK@512 on Dream + eval     (peer detectability)
#   42 dream  : KGW + dgMARK on GSM8K, Dream   (peer quality)
#   39        : Shibboleth GSM8K on Dream
#   33b       : localized-edit attack (saved outputs)
cd /ssd2/ming/basinmark
P=/home/ming0531/miniconda3/envs/mmada/bin/python
while pgrep -f 36_dream_detect > /dev/null; do sleep 60; done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/42_peer_gsm8k.py llada > logs/42_peer_llada.log 2>&1; echo "=== 42L exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/37_dream_kgw512.py > logs/37_dream_kgw.log 2>&1; echo "=== 37 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
bash exp/38_dgmark_dream.sh > logs/38_dgdream.log 2>&1; echo "=== 38 gen exit $? ==="
$P exp/38_dgdream_eval.py > logs/38_dgdream_eval.log 2>&1; echo "=== 38 eval exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/42_peer_gsm8k.py dream > logs/42_peer_dream.log 2>&1; echo "=== 42D exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/39_dream_gsm8k.py > logs/39_dream_gsm8k.log 2>&1; echo "=== 39 exit $? ==="
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 30; done
$P exp/33b_localized.py > logs/33b_localized.log 2>&1; echo "=== 33b exit $? ==="
echo "=== PEER CHAIN DONE ==="
