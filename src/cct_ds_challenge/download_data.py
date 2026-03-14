from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import requests

from cct_ds_challenge.paths import (
    DATA_RAW,
    PROJECT_ROOT,
    RAW_SERVICE_REQUESTS,
    RAW_SWIMMING_POOLS,
)

BASE_URL = "https://cct-ds-code-challenge-input-data.s3.af-south-1.amazonaws.com"

SERVICE_REQUEST_FILES = [
    "sr.csv.gz",
    "sr_hex.csv.gz",
    "sr_hex_truncated.csv",
    "city-hex-polygons-8.geojson",
    "city-hex-polygons-8-10.geojson",
]

MANIFEST_URLS = {
    "yes": f"{BASE_URL}/images/swimming-pool/yes/manifest",
    "no": f"{BASE_URL}/images/swimming-pool/no/manifest",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(
    url: str,
    destination: Path,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
    overwrite: bool = False,
) -> None:
    if destination.exists() and not overwrite:
        logging.debug("Skipping existing file: %s", destination)
        return

    ensure_dir(destination.parent)
    logging.info("Downloading %s -> %s", url, destination)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(destination, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    output_file.write(chunk)


def parse_manifest_line(line: str) -> str | None:
    """
    Handle several plausible manifest formats:
    1. Plain filename
    2. Relative key
    3. Full https URL
    4. s3:// URI
    5. JSON line with an image reference field
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("{") and line.endswith("}"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            pass
        else:
            for key in ("source-ref", "source_ref", "image", "image_url", "url", "key", "path"):
                if key in record and isinstance(record[key], str):
                    return record[key]

    return line


def normalize_manifest_entry(entry: str, label: str) -> str:
    entry = entry.strip()

    if entry.startswith("https://") or entry.startswith("http://"):
        return entry

    if entry.startswith("s3://"):
        without_scheme = entry[len("s3://") :]
        bucket, _, key = without_scheme.partition("/")
        return f"https://{bucket}.s3.af-south-1.amazonaws.com/{key}"

    if entry.startswith("/"):
        entry = entry[1:]

    if "/" not in entry:
        return f"{BASE_URL}/images/swimming-pool/{label}/{entry}"

    return f"{BASE_URL}/{entry}"


def infer_filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def download_service_request_files(
    destination_dir: Path = RAW_SERVICE_REQUESTS,
    overwrite: bool = False,
) -> None:
    ensure_dir(destination_dir)

    for filename in SERVICE_REQUEST_FILES:
        url = f"{BASE_URL}/{filename}"
        destination = destination_dir / filename
        download_file(url=url, destination=destination, overwrite=overwrite)


def download_manifest(
    manifest_url: str,
    destination: Path,
    overwrite: bool = False,
) -> list[str]:
    download_file(manifest_url, destination, overwrite=overwrite)
    lines = destination.read_text(encoding="utf-8").splitlines()

    entries: list[str] = []
    for line in lines:
        parsed = parse_manifest_line(line)
        if parsed:
            entries.append(parsed)

    return entries


def download_swimming_pool_images(
    destination_dir: Path = RAW_SWIMMING_POOLS,
    overwrite: bool = False,
    max_per_class: int | None = None,
) -> None:
    ensure_dir(destination_dir)

    for label, manifest_url in MANIFEST_URLS.items():
        label_dir = destination_dir / label
        ensure_dir(label_dir)

        manifest_destination = label_dir / "manifest"
        entries = download_manifest(
            manifest_url=manifest_url,
            destination=manifest_destination,
            overwrite=overwrite,
        )

        if max_per_class is not None:
            entries = entries[:max_per_class]

        logging.info("Found %s manifest entries for label '%s'", len(entries), label)

        for idx, entry in enumerate(entries, start=1):
            file_url = normalize_manifest_entry(entry, label=label)
            filename = infer_filename_from_url(file_url)
            destination = label_dir / filename

            try:
                download_file(file_url, destination, overwrite=overwrite)
            except requests.HTTPError as exc:
                logging.warning(
                    "Failed to download image %s/%s for label '%s': %s",
                    idx,
                    len(entries),
                    label,
                    exc,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download City of Cape Town DS challenge datasets."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root. Used only for validation/logging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Optional cap on number of swimming pool images downloaded per class.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip swimming pool image downloads.",
    )
    parser.add_argument(
        "--skip-service-requests",
        action="store_true",
        help="Skip service request and geojson downloads.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def validate_project_root(project_root: Path) -> None:
    if project_root.resolve() != PROJECT_ROOT.resolve():
        logging.warning(
            "Provided project root (%s) differs from package project root (%s). "
            "Downloads will still use paths defined in cct_ds_challenge.paths.",
            project_root,
            PROJECT_ROOT,
        )


def main() -> int:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    validate_project_root(args.project_root)

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Raw data directory: %s", DATA_RAW)
    logging.info("Service request directory: %s", RAW_SERVICE_REQUESTS)
    logging.info("Swimming pool directory: %s", RAW_SWIMMING_POOLS)

    try:
        if not args.skip_service_requests:
            download_service_request_files(
                destination_dir=RAW_SERVICE_REQUESTS,
                overwrite=args.overwrite,
            )

        if not args.skip_images:
            download_swimming_pool_images(
                destination_dir=RAW_SWIMMING_POOLS,
                overwrite=args.overwrite,
                max_per_class=args.max_images_per_class,
            )
    except requests.RequestException as exc:
        logging.exception("Download failed due to network/request error: %s", exc)
        return 1
    except Exception as exc:
        logging.exception("Download failed due to unexpected error: %s", exc)
        return 1

    logging.info("Download completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())