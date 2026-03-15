from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point


COMMON_LON_COLS = ["longitude", "lon", "lng", "x", "LONGITUDE", "LON", "LNG", "X"]
COMMON_LAT_COLS = ["latitude", "lat", "y", "LATITUDE", "LAT", "Y"]


def find_first_existing(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def load_city_boundary(hex_geojson_path: Path) -> gpd.GeoDataFrame:
    hex_gdf = gpd.read_file(hex_geojson_path)

    if hex_gdf.crs is None:
        hex_gdf = hex_gdf.set_crs("EPSG:4326")
    else:
        hex_gdf = hex_gdf.to_crs("EPSG:4326")

    return hex_gdf


def detect_geometry_mode(sample_chunk: pd.DataFrame) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
        mode: "wkt" or "point"
        geometry_col: geometry column if mode == "wkt"
        lon_col: longitude column if mode == "point"
        lat_col: latitude column if mode == "point"
    """
    if "geometry" in sample_chunk.columns:
        sample_vals = sample_chunk["geometry"].dropna().astype(str).head(5).tolist()
        if any(val.startswith(("POLYGON", "MULTIPOLYGON", "POINT")) for val in sample_vals):
            return "wkt", "geometry", None, None

    lon_col = find_first_existing(sample_chunk.columns, COMMON_LON_COLS)
    lat_col = find_first_existing(sample_chunk.columns, COMMON_LAT_COLS)
    if lon_col and lat_col:
        return "point", None, lon_col, lat_col

    raise ValueError(
        "Could not determine geometry representation in the buildings file. "
        "Expected either a WKT 'geometry' column or lon/lat columns."
    )


def chunk_to_geodf(
    chunk: pd.DataFrame,
    mode: str,
    geometry_col: Optional[str],
    lon_col: Optional[str],
    lat_col: Optional[str],
) -> gpd.GeoDataFrame:
    if mode == "wkt":
        geom = chunk[geometry_col].fillna("").map(lambda x: wkt.loads(x) if x else None)
        gdf = gpd.GeoDataFrame(chunk.copy(), geometry=geom, crs="EPSG:4326")
        gdf = gdf[gdf.geometry.notnull()].copy()
        return gdf

    if mode == "point":
        valid = chunk[lon_col].notna() & chunk[lat_col].notna()
        chunk = chunk.loc[valid].copy()
        geom = [Point(xy) for xy in zip(chunk[lon_col], chunk[lat_col])]
        gdf = gpd.GeoDataFrame(chunk, geometry=geom, crs="EPSG:4326")
        return gdf

    raise ValueError(f"Unsupported geometry mode: {mode}")


def filter_buildings_to_city(
    input_csv_gz: Path,
    hex_geojson_path: Path,
    output_csv_gz: Path,
    chunksize: int = 100_000,
    predicate: str = "intersects",
) -> None:
    """
    Filter a large Google Buildings CSV.GZ down to features intersecting the Cape Town hex boundary.

    predicate:
        - "intersects" is safer for building polygons touching the boundary
        - "within" is stricter
    """
    hex_gdf = load_city_boundary(hex_geojson_path)
    city_boundary = hex_gdf.union_all()

    output_csv_gz.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(input_csv_gz, compression="gzip", chunksize=chunksize)

    first_chunk = next(reader)
    mode, geometry_col, lon_col, lat_col = detect_geometry_mode(first_chunk)

    total_rows = 0
    kept_rows = 0
    wrote_header = False

    def process_and_write(chunk: pd.DataFrame, wrote_header_flag: bool) -> bool:
        nonlocal total_rows, kept_rows

        total_rows += len(chunk)

        gdf = chunk_to_geodf(chunk, mode, geometry_col, lon_col, lat_col)

        if gdf.empty:
            return wrote_header_flag

        if predicate == "within":
            keep_mask = gdf.geometry.within(city_boundary)
        else:
            keep_mask = gdf.geometry.intersects(city_boundary)

        filtered = gdf.loc[keep_mask].drop(columns="geometry", errors="ignore").copy()
        kept_rows += len(filtered)

        if filtered.empty:
            return wrote_header_flag

        filtered.to_csv(
            output_csv_gz,
            mode="a",
            header=not wrote_header_flag,
            index=False,
            compression="gzip",
        )
        return True

    wrote_header = process_and_write(first_chunk, wrote_header)

    for chunk in reader:
        wrote_header = process_and_write(chunk, wrote_header)

    print(f"Done. Total input rows: {total_rows:,}")
    print(f"Rows kept inside city boundary: {kept_rows:,}")
    print(f"Filtered file written to: {output_csv_gz}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Google Buildings CSV.GZ to buildings within the City of Cape Town hex boundary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/1dd_buildings.csv.gz"),
        help="Path to the original Google Buildings CSV.GZ file.",
    )
    parser.add_argument(
        "--hex-geojson",
        type=Path,
        default=Path("data/raw/service_requests/city-hex-polygons-8.geojson"),
        help="Path to the city H3 hex polygons GeoJSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/1dd_buildings_cape_town.csv.gz"),
        help="Path to the filtered output CSV.GZ.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Chunk size for streaming the input CSV.",
    )
    parser.add_argument(
        "--predicate",
        type=str,
        choices=["intersects", "within"],
        default="intersects",
        help="Spatial predicate to use for filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filter_buildings_to_city(
        input_csv_gz=args.input,
        hex_geojson_path=args.hex_geojson,
        output_csv_gz=args.output,
        chunksize=args.chunksize,
        predicate=args.predicate,
    )


if __name__ == "__main__":
    main()
