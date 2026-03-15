#!/usr/bin/env bash
set -e

echo "Running all workflows in sequence."

bash scripts/download_data.sh

bash scripts/run_cv.sh

bash scripts/run_sr.sh

echo "All workflows completed."