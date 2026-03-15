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