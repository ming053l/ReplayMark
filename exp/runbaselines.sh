#!/usr/bin/env bash
# Baselines on the SAME base model, prompts and generation length as BasinMark's main
# setting (LLaDA-8B-Instruct, C4 realnewslike, 256 tokens). Smoke test first.
LLADA=/ssd1/ming/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
C4=/ssd1/ming/basinmark/data/c4-validation.json.gz
MMADA=/home/ming0531/miniconda3/envs/mmada/bin/python
OUT=/ssd1/ming/basinmark/results/baselines
mkdir -p "$OUT"

echo "=== dgMARK smoke (2 samples) ==="
cd /ssd1/ming/basinmark/baselines/dgmark-watermarking
$MMADA scripts/generate.py --method watermark --num_samples 2 --gen_length 256 \
  --block_length 32 --sampling_strategy multinomial --top_k 3 \
  --model_name "$LLADA" --cache_dir /ssd1/ming/hf_cache \
  --dataset_path "$C4" --output_prefix "$OUT/smoke" || { echo "DGMARK SMOKE FAILED"; exit 1; }
echo "=== dgMARK smoke OK ==="

for METHOD in original watermark; do
  echo "=== dgMARK $METHOD (50 samples, 256 tok) ==="
  $MMADA scripts/generate.py --method $METHOD --num_samples 50 --gen_length 256 \
    --block_length 32 --sampling_strategy multinomial --top_k 3 \
    --model_name "$LLADA" --cache_dir /ssd1/ming/hf_cache \
    --dataset_path "$C4" --output_prefix "$OUT/dgmark_$METHOD" || echo "dgMARK $METHOD FAILED"
done
echo "=== dgMARK detect ==="
$MMADA scripts/detect.py --watermarked "$OUT/dgmark_watermark.csv" \
  --original "$OUT/dgmark_original.csv" --plot "$OUT/dgmark_detection.png" \
  || echo "dgMARK DETECT FAILED"
echo "=== dgMARK DONE ==="

echo "=== eth-sri smoke ==="
source /home/ming0531/miniconda3/etc/profile.d/conda.sh
conda activate dlmwm
export HF_HOME=/ssd1/ming/hf_cache
cd /ssd1/ming/basinmark/baselines/diffusion-lm-watermark
for c in no_watermark KGW ourWatermark; do
  sed -e 's/gen_length: [0-9]*/gen_length: 256/' -e 's/steps: [0-9]*/steps: 256/' \
      "configs/sm75/${c}.yaml" > "configs/sm75/${c}_256.yaml"
done
sed 's/num_samples: 50/num_samples: 2/' configs/sm75/KGW_256.yaml > configs/sm75/_smoke.yaml
python scripts/run_config.py --config configs/sm75/_smoke.yaml \
  || { echo "ETHSRI SMOKE FAILED"; exit 1; }
echo "=== eth-sri smoke OK ==="
for CFG in no_watermark KGW ourWatermark; do
  echo "=== eth-sri $CFG (50 samples, 256 tok) ==="
  python scripts/run_config.py --config "configs/sm75/${CFG}_256.yaml" \
    || echo "eth-sri $CFG FAILED"
done
echo "=== BASELINES DONE ==="
