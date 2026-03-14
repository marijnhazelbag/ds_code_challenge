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

