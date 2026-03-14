
<img src="img/city_emblem.png" alt="City Logo"/>

# City of Cape Town Data Science Code Challenge

This repository contains my submission for the City of Cape Town Data Science Unit code challenge.

Original challenge repository:
https://github.com/cityofcapetown/ds_code_challenge

## Tasks completed

1. Computer Vision Classification  
Detect whether aerial images contain swimming pools.

2. Service Request Predictive Modelling  
Predict service request patterns using spatial hexagon aggregation and historical data.

## Environment

Tested with Python 3.12.13.

Create environment:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Running the project

Run both workflows:

./scripts/run_all.sh

Or individually:

./scripts/run_cv.sh
./scripts/run_sr.sh

## Raw challenge data
Raw challenge data is not committed to this repository. It is downloaded programmatically from the public City of Cape Town challenge S3 bucket.

The challenge README states that the provided AWS credentials do not grant special access beyond what is already anonymously available, so this repository intentionally uses public download only.

Run:
./scripts/download_data.sh

## Repository structure

data  
Raw and processed datasets

notebooks  
Exploratory analysis and modelling notebooks

src  
Reusable code modules

reports  
Final outputs including the generated report

tests  
Basic data validation tests

scripts  
Utility scripts for running workflows

## Outputs

The final report is generated as an executed notebook and exported to HTML.

reports/exports/challenge_report.html

## AI usage

Use of generative AI during development is documented in:

AI_log.md