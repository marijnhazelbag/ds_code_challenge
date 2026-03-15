#!/usr/bin/env bash
set -e

echo "Running both workflows in sequence."

echo "Downloading raw challenge data from the public City of Cape Town challenge S3 bucket."
bash scripts/download_data.sh

bash scripts/run_cv.sh

echo "Currently running service request workflow."
bash scripts/run_sr.sh

echo "All workflows completed."