#!/usr/bin/env bash
set -e

echo "Running both workflows in sequence."

echo "Currently running computer vision workflow."
bash scripts/run_cv.sh

echo "Currently running service request workflow."
bash scripts/run_sr.sh

echo "All workflows completed."