#!/usr/bin/env bash
set -e

echo "Running computer vision workflow..."

python src/cct_ds_challenge/train_cv.py \
    --config configs/cv/default.yaml

echo "CV workflow complete."
