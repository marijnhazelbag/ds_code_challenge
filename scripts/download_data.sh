#!/usr/bin/env bash
set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

python -m cct_ds_challenge.download_data "$@"