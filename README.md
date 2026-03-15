<img src="img/city_emblem.png" alt="City Logo"/>

# City of Cape Town Data Science Code Challenge

This repository contains my submission for the City of Cape Town Data Science Unit data science code challenge.

Original challenge repository:
[https://github.com/cityofcapetown/ds_code_challenge](https://github.com/cityofcapetown/ds_code_challenge)

## Tasks completed

### 1. Computer Vision Classification

Detect whether aerial images contain swimming pools.

A transfer learning approach is used to classify aerial imagery tiles as containing a swimming pool or not. The training pipeline includes dataset download, data loading, model training, and evaluation.

The workflow is implemented in:

src/cct_ds_challenge/train_cv.py

and executed via:

./scripts/run_cv.sh

### 2. Service Request Predictive Modelling

Predict the number of sewer blockage / overflow service requests per spatial hexagon.

The modelling workflow aggregates service request data at H3 level 8 hex resolution and builds explanatory features from service request patterns and building characteristics.

The modelling pipeline includes:

Baseline model

Poisson regression using service request derived features.

Improved model

Poisson regression including Google Open Buildings features such as building count, mean building area, and variability in building size.

Sensitivity analysis

A Negative Binomial model is used as a robustness check for potential overdispersion in the count data.

The modelling workflow is implemented in:

src/cct_ds_challenge/train_sr.py

and executed via:

./scripts/run_sr.sh

The model generates evaluation metrics, driver analysis tables, diagnostic plots, and an HTML report.

Outputs are written to:

reports/sr/

## Environment

Tested with Python 3.12.13.

Create environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Running the project

Run the full pipeline:

./scripts/run_all.sh

This script runs the following steps in sequence:

1. Download raw challenge data
2. Run the computer vision workflow
3. Run the service request modelling workflow

Workflows can also be executed individually:

./scripts/run_cv.sh
./scripts/run_sr.sh

## Raw challenge data

Raw challenge data is not committed to this repository. It is downloaded programmatically from the public City of Cape Town challenge S3 bucket.

The challenge README states that the provided AWS credentials do not grant special access beyond what is already anonymously available. This repository therefore downloads the data directly from the public bucket.

Run:

./scripts/download_data.sh

## External data

### Google Open Buildings

The service request modelling task uses Google Open Buildings as an additional explanatory dataset.

* non sewer (target) requests
* diversity of requests (excluding target)

A South Africa tile of the dataset was downloaded manually and used as input to the feature engineering pipeline. The feature engineering script assigns buildings to Cape Town H3 hexes using their coordinates and computes per-hex summary features such as:

* building count
* mean building area
* standard deviation of building area

This processing is implemented in:

src/cct_ds_challenge/build_sr_features.py

To simplify evaluation of the modelling workflow, the processed feature tables are committed to the repository. This avoids requiring reviewers to obtain and process the Google Open Buildings dataset themselves.

The committed processed outputs are:

data/processed/hex_building_features.csv
data/processed/sewer_hex_features.csv
data/processed/sr_type_hex_counts.csv

## Feature engineering

Service request modelling features were generated using:

src/cct_ds_challenge/build_sr_features.py

To simplify reproducibility during evaluation, the processed feature tables used for modelling are committed to the repository. This allows the modelling workflow to run immediately without recomputing features from large raw datasets.

## Repository structure

```
data
    raw
    processed
    external

notebooks
    exploratory analysis

src
    reusable code modules

reports
    generated modelling outputs

scripts
    workflow scripts

```

## Outputs

### Computer vision

Outputs from the computer vision workflow are written to:

reports/cv/

These include:

* training histories
* evaluation metrics
* class summaries
* the generated HTML report

Example files:

reports/cv/baseline_metrics.csv
reports/cv/improved_metrics.csv
reports/cv/cv_report.html

### Service request modelling

Outputs from the service request modelling workflow are written to:

reports/sr/

This includes the generated modelling report and supporting figures.

reports/sr/sr_report.html

## AI usage

Use of generative AI during development is documented in:

AI_log.md

AI tools were used primarily for drafting initial code scaffolding, suggesting refactoring approaches, and reviewing modelling strategies. All generated code and suggestions were reviewed and modified before inclusion in the final solution.
