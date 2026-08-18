#!/usr/bin/env bash
# Baseline runs. One 24 GB card, 8B fp16 = 16 GB, so these must run SERIALLY.
set -e
LLADA=/ssd1/ming/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
C4=/ssd1/ming/basinmark/data/c4-validation.json.gz
MMADA=/home/ming0531/miniconda3/envs/mmada/bin/python
OUT=/ssd1/ming/basinmark/results/baselines
mkdir -p "$OUT"

# ---- dgMARK (ICML 2026) -- decoding-order watermark -------------------------
# Patched for sm75: bfloat16 -> float16, mem-efficient SDPA re-enabled.
# Run on LLaDA-8B-Instruct (not its default LLaDA-1.5) so it shares a model with
# BasinMark; use --model_name GSAI-ML/LLaDA-1.5 to reproduce the paper's own numbers.
cd /ssd1/ming/basinmark/baselines/dgmark-watermarking
for METHOD in original watermark; do
  $MMADA scripts/generate.py \
    --method $METHOD --num_samples 50 --gen_length 192 --block_length 32 \
    --sampling_strategy multinomial --top_k 3 \
    --model_name "$LLADA" --cache_dir /ssd1/ming/hf_cache \
    --dataset_path "$C4" --output_prefix "$OUT/dgmark_$METHOD"
done
$MMADA scripts/detect.py \
  --watermarked "$OUT/dgmark_watermark.csv" --original "$OUT/dgmark_original.csv" \
  --plot "$OUT/dgmark_detection.png"

# ---- eth-sri (ICLR 2026) -- red/green in expectation, + KGW/KTH/Unigram/AAR/OA
# Separate env: py3.12 / torch 2.8 / transformers 4.56 (conda env `dlmwm`).
# Their LLaDA configs already target GSAI-ML/LLaDA-8B-Instruct.
# NOTE: main configs are num_samples=200, gen_length=300, steps=300 -> ~1.5-2 h each
# on this card. Cut num_samples for iteration; run full only for the final table.
source /home/ming0531/miniconda3/etc/profile.d/conda.sh
conda activate dlmwm
export HF_HOME=/ssd1/ming/hf_cache
cd /ssd1/ming/basinmark/baselines/diffusion-lm-watermark
for CFG in no_watermark KGW ourWatermark; do
  python scripts/run_config.py --config "configs/main/Llada/${CFG}_llada8b_instruct.yaml"
done
