#!/usr/bin/env bash
# The paper's flagship configuration is dgMARK + 3-beam one-step lookahead (Table 2).
LLADA=/ssd1/ming/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
C4=/ssd1/ming/basinmark/data/c4-validation.json.gz
MMADA=/home/ming0531/miniconda3/envs/mmada/bin/python
OUT=/ssd1/ming/basinmark/results/baselines
while pgrep -f "runbaselines" >/dev/null; do sleep 30; done
cd /ssd1/ming/basinmark/baselines/dgmark-watermarking
echo "=== dgMARK 3-beam (50 samples, 256 tok) ==="
$MMADA scripts/generate.py --method beam --beam_size 3 --num_samples 50 --gen_length 256 \
  --block_length 32 --sampling_strategy multinomial --top_k 3 \
  --model_name "$LLADA" --cache_dir /ssd1/ming/hf_cache \
  --dataset_path "$C4" --output_prefix "$OUT/dgmark_beam3" || echo "BEAM FAILED"
echo "=== dgMARK beam DONE ==="
