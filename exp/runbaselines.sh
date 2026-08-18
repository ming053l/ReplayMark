#!/usr/bin/env bash
# Baselines, serial, after the BasinMark runs. Smoke test each with 2 samples first so
# an environment failure surfaces in a minute instead of after an hour.
LLADA=/ssd1/ming/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
C4=/ssd1/ming/basinmark/data/c4-validation.json.gz
MMADA=/home/ming0531/miniconda3/envs/mmada/bin/python
OUT=/ssd1/ming/basinmark/results/baselines
mkdir -p "$OUT"
cd /ssd1/ming/basinmark
while pgrep -f "exp/1[12]_" >/dev/null; do sleep 20; done

echo "=== dgMARK smoke test (2 samples) ==="
cd /ssd1/ming/basinmark/baselines/dgmark-watermarking
$MMADA scripts/generate.py --method watermark --num_samples 2 --gen_length 192 \
  --block_length 32 --sampling_strategy multinomial --top_k 3 \
  --model_name "$LLADA" --cache_dir /ssd1/ming/hf_cache \
  --dataset_path "$C4" --output_prefix "$OUT/smoke" || { echo "DGMARK SMOKE FAILED"; exit 1; }
echo "=== dgMARK smoke OK ==="

for METHOD in original watermark; do
  echo "=== dgMARK $METHOD (50 samples) ==="
  $MMADA scripts/generate.py --method $METHOD --num_samples 50 --gen_length 192 \
    --block_length 32 --sampling_strategy multinomial --top_k 3 \
    --model_name "$LLADA" --cache_dir /ssd1/ming/hf_cache \
    --dataset_path "$C4" --output_prefix "$OUT/dgmark_$METHOD"
done
$MMADA scripts/detect.py --watermarked "$OUT/dgmark_watermark.csv" \
  --original "$OUT/dgmark_original.csv" --plot "$OUT/dgmark_detection.png"
echo "=== dgMARK DONE ==="

echo "=== eth-sri smoke test ==="
source /home/ming0531/miniconda3/etc/profile.d/conda.sh
conda activate dlmwm
export HF_HOME=/ssd1/ming/hf_cache
cd /ssd1/ming/basinmark/baselines/diffusion-lm-watermark
sed 's/num_samples: 50/num_samples: 2/' configs/sm75/KGW.yaml > configs/sm75/_smoke.yaml
python scripts/run_config.py --config configs/sm75/_smoke.yaml \
  || { echo "ETHSRI SMOKE FAILED"; exit 1; }
echo "=== eth-sri smoke OK ==="
for CFG in no_watermark KGW ourWatermark; do
  echo "=== eth-sri $CFG ==="
  python scripts/run_config.py --config "configs/sm75/${CFG}.yaml"
done
echo "=== BASELINES DONE ==="
