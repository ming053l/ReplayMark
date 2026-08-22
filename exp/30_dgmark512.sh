#!/usr/bin/env bash
# dgMARK at 512 tokens, n=30 -- the length-matched row the table is missing.
LLADA=/ssd2/ming/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
C4=/ssd2/ming/basinmark/data/c4-validation.json.gz
P=/home/ming0531/miniconda3/envs/mmada/bin/python
OUT=/ssd2/ming/basinmark/results/baselines
cd /ssd2/ming/basinmark/baselines/dgmark-watermarking
for M in original watermark; do
  $P scripts/generate.py --method $M --num_samples 30 --gen_length 512 \
    --block_length 32 --sampling_strategy multinomial --top_k 3 \
    --model_name "$LLADA" --cache_dir /ssd2/ming/hf_cache \
    --dataset_path "$C4" --output_prefix "$OUT/dg512_$M" || echo "dg512 $M FAILED"
done
$P scripts/generate.py --method beam --beam_size 3 --num_samples 30 --gen_length 512 \
  --block_length 32 --sampling_strategy multinomial --top_k 3 \
  --model_name "$LLADA" --cache_dir /ssd2/ming/hf_cache \
  --dataset_path "$C4" --output_prefix "$OUT/dg512_beam3" || echo "dg512 beam FAILED"
