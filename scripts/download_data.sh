#!/usr/bin/env bash
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo "Downloading raw challenge data from the public City of Cape Town challenge S3 bucket."
python -m cct_ds_challenge.download_data "$@"