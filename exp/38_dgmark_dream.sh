#!/usr/bin/env bash
# dgMARK at 512 tokens on Dream-7B-Instruct, n=30 x 3 arms -- the Dream block's dgMARK rows.
DREAM=$(ls -d /ssd2/ming/hf_cache/hub/models--Dream-org--Dream-v0-Instruct-7B/snapshots/*/)
C4=/ssd2/ming/basinmark/data/c4-validation.json.gz
P=/home/ming0531/miniconda3/envs/mmada/bin/python
OUT=/ssd2/ming/basinmark/results/baselines
cd /ssd2/ming/basinmark/baselines/dgmark-watermarking
for M in original watermark; do
  $P scripts/generate.py --method $M --num_samples 30 --gen_length 512 \
    --block_length 32 --sampling_strategy multinomial --top_k 3 \
    --model_name "$DREAM" --cache_dir /ssd2/ming/hf_cache \
    --mask_id 151666 --shift_logits --eot_id 151643 \
    --dataset_path "$C4" --output_prefix "$OUT/dgdream_$M" || echo "dgdream $M FAILED"
done
$P scripts/generate.py --method beam --beam_size 3 --num_samples 30 --gen_length 512 \
  --block_length 32 --sampling_strategy multinomial --top_k 3 \
  --model_name "$DREAM" --cache_dir /ssd2/ming/hf_cache \
  --mask_id 151666 --shift_logits --eot_id 151643 \
  --dataset_path "$C4" --output_prefix "$OUT/dgdream_beam3" || echo "dgdream beam FAILED"
