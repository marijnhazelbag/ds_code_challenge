from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_EXTERNAL = DATA_DIR / "external"
DATA_PROCESSED = DATA_DIR / "processed"

RAW_SERVICE_REQUESTS = DATA_RAW / "service_requests"
RAW_SWIMMING_POOLS = DATA_RAW / "swimming_pools"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_EXPORTS = REPORTS_DIR / "exports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"