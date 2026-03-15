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
REPORTS_CV = REPORTS_DIR / "cv"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"


def _resolve_from_root(root_dir: Path, path_value: str | Path) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (root_dir / path_obj).resolve()


def resolve_cv_paths(
    project_root: str | Path = PROJECT_ROOT,
    image_dir: str | Path = RAW_SWIMMING_POOLS,
    output_dir: str | Path = REPORTS_CV,
) -> dict[str, Path]:
    """Resolve and create the main input/output paths for the CV pipeline."""
    root_dir = Path(project_root).resolve()
    image_dir = _resolve_from_root(root_dir, image_dir)
    output_dir = _resolve_from_root(root_dir, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    return {
        "root": root_dir,
        "image_dir": image_dir,
        "output_dir": output_dir,
        "figures_dir": output_dir / "figures",
        "models_dir": output_dir / "models",
    }
