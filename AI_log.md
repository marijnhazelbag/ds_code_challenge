## Entry 1

Date: 2026-03-14  
Tool: ChatGPT (GPT-5)  
Task: Repository structure and environment setup

Summary: Used AI to propose an initial project structure and development environment for the challenge repository.

Changes made by me:
- Simplified the suggested directory structure to keep the repository minimal at the start.
- Reduced the initial dependency list to only the packages needed for early exploration, with the intention to expand later.

## Entry 2

Date: 2026-03-14  
Tool: ChatGPT (GPT-5)  
Task: Implement reproducible dataset download pipeline (`download_data.py`)

Summary: Used AI to generate an initial implementation for automatically downloading the challenge datasets from the City of Cape Town S3 bucket into the project structure (`data/raw/service_requests` and `data/raw/swimming_pools`). The generated code included manifest parsing, logging, and logic to skip files that already exist.

Changes made by me:
- Corrected image download logic after discovering the manifest only contains filenames (e.g. `W07C_11_85.tif`). Updated URL construction to include the correct prefix `images/swimming-pool/<label>/`.
- Refactored the script to use the project's `paths.py` module rather than defining path constants inside the script.
- Removed reliance on the provided AWS credentials and implemented public HTTPS downloads instead, as the bucket contents are anonymously accessible and credentials may rotate.
- Verified that existing files are skipped on subsequent runs to support reproducible and incremental data downloads.

Outcome: The final implementation allows the repository to be cloned and executed without manual data setup, downloading the required datasets automatically while keeping large raw data out of version control.

## Entry 3

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Build initial computer vision baseline in a single training script

Summary: Used AI to help draft an initial end-to-end baseline for the swimming pool classification task in a single training script. This covered dataset loading, train/validation splitting, a first transfer-learning model setup, training loop, and basic evaluation outputs.

Changes made by me:
- Adjusted the script structure to fit the project layout and downloaded data locations used in this repository.
- Simplified parts of the generated code to keep the first version easy to run and inspect.
- Verified the script ran end to end in my environment and produced baseline outputs for the challenge.
- Used this first version as a working baseline before refactoring into a more maintainable structure.

Outcome: The initial single-file script provided a fast baseline for the swimming pool classification task and served as the starting point for later cleanup and modularisation.

## Entry 4

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Refactor CV baseline into modular utilities and configuration-driven pipeline

Summary: Used AI to help refactor the initial single-file CV baseline into a cleaner structure with reusable utility modules, central path handling, and YAML-based configuration. This included separating data loading, common helpers, modelling, and reporting concerns.

Changes made by me:
- Split the original single-file implementation into `cv_utils` modules to improve readability and maintainability.
- Added configuration files for CV experiments and consolidated path handling via `paths.py`.
- Renamed the refactored training entrypoint to `train_cv.py` and removed the older baseline script to avoid duplicate entrypoints.
- Fixed issues introduced during refactoring, including import and type-hint related problems, while avoiding circular dependencies.

Outcome: The CV workflow is now easier to run, debug, and extend, while still preserving the original baseline functionality developed for the challenge.

## Entry 5

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Design geospatial feature engineering pipeline for service request modelling

Summary: Used AI to help design the feature engineering pipeline for the service request modelling task. This included defining how to aggregate service request counts by hex, avoid target leakage, and incorporate Google Buildings data as explanatory variables.

Changes made by me:
- Chose to implement feature engineering as a **separate preprocessing script** rather than embedding it directly in the training script, keeping data preparation reproducible and modular.
- Defined the final set of derived features including `non_target_requests`, `diversity_ex_target`, and log-transformed predictors to stabilise variance.
- Decided to compute building features (count, mean area, area variability) per hex using Google Buildings data and store these as processed features.
- Ensured that feature construction avoids target leakage by subtracting sewer requests from total service requests when constructing predictors.

Outcome: The repository now contains a clear preprocessing step that transforms raw service request and building data into a modelling-ready dataset.

---

## Entry 6

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Implement feature engineering script (`build_sr_features.py`)

Summary: Used AI to generate the initial implementation of the feature engineering pipeline for the service request modelling task. The script aggregates service requests per hex, constructs modelling features, and writes processed outputs to the repository’s `data/processed` directory.

Changes made by me:
- Adjusted file paths to match the repository’s `paths.py` conventions.
- Simplified intermediate outputs and removed unnecessary artefacts so that only modelling-ready datasets are written to disk.
- Fixed dependency issues (e.g. parquet engine requirements) and verified the outputs locally.
- Validated the resulting parquet files by loading them in a notebook and inspecting summary statistics.

Outcome: The feature engineering pipeline can now be run end-to-end to generate modelling datasets in a reproducible way.

---

## Entry 7

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Develop baseline modelling workflow for sewer request prediction (`train_sr.py`)

Summary: Used AI to help draft the initial modelling workflow for predicting sewer blockage/overflow requests per H3 hex. The script trains a baseline Poisson regression model, evaluates performance on validation and test sets, and generates a reproducible HTML report including metrics, figures, and driver analysis.

Changes made by me:
- Adapted the generated script to work with the feature dataset produced by `build_sr_features.py`.
- Introduced filtering to remove structural-zero hexes with extremely low building counts, which correspond to areas unlikely to generate sewer incidents.
- Verified that the script executes end-to-end and produces evaluation outputs and plots in the repository’s reports directory.
- Ensured all outputs are saved automatically to support reproducibility.

Outcome: The repository now contains a complete modelling workflow that can be executed with a single command to generate results and documentation.

---

## Entry 8

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Improve model comparison and interpretability in sewer request modelling report

Summary: Used AI to revise the modelling approach and report structure so that the comparison between models reflects meaningful methodological improvements. The workflow was updated to compare a baseline Poisson model using service-request-derived features against an improved Poisson model that incorporates Google Buildings data.

Changes made by me:
- Redefined the modelling comparison so that the **improvement step reflects additional explanatory data** rather than simply switching model families.
- Retained the Negative Binomial model as a **sensitivity analysis** for overdispersed count data rather than presenting it as the primary improved model.
- Corrected the driver analysis tables so that positive and negative effects are separated based on coefficient sign.
- Added clear interpretations of incidence-rate ratios to explain the magnitude of identified drivers.
- Expanded the report narrative to clarify that the analysis is **explanatory rather than causal**, given the limited feature set available.

Outcome: The final modelling workflow and report now present a clearer and more defensible explanation of sewer request drivers, while maintaining reproducible evaluation and transparent modelling choices.

---

## Entry 9

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Strengthen modelling rationale and interpretability discussion in report

Summary: Used AI to refine the modelling rationale and supporting discussion in the generated HTML report. This included clarifying the reasoning behind model selection, train/validation/test splitting, and the use of interpretable statistical models rather than complex black-box approaches.

Changes made by me:
- Added discussion explaining why interpretable count models were preferred for this task.
- Included a reference to recent research highlighting the instability of attribution methods used to interpret complex neural networks.
- Clarified that the train/validation/test split is used to evaluate generalisation and ensure fair comparison between modelling approaches.
- Emphasised that the results should be interpreted as **associations rather than causal effects**, due to omitted variables such as infrastructure age, rainfall, and sewer network topology.

Outcome: The report now provides stronger methodological justification for the modelling choices and better communicates the limitations and interpretation of the results.

---

## Entry 10

Date: 2026-03-15  
Tool: ChatGPT (GPT-5)  
Task: Review and refinement of AI-assisted modelling outputs

Summary: Used AI iteratively to review modelling results, inspect generated reports, and identify areas where the analysis or interpretation could be improved before finalising the challenge submission.

Changes made by me:
- Critically reviewed the AI-generated modelling outputs and report content rather than accepting suggestions verbatim.
- Adjusted the narrative around model performance where validation improvements did not translate into test-set gains.
- Ensured that model comparisons and driver interpretations remained scientifically defensible.
- Integrated only those AI suggestions that improved clarity, reproducibility, or methodological soundness.

Outcome: The final solution reflects a combination of AI-assisted drafting and manual review, ensuring that modelling decisions and interpretations remain the responsibility of the author.
