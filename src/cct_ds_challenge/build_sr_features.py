from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from paths import (
    SR_HEX,
    CITY_HEX_POLYGONS_8,
    GOOGLE_BUILDINGS_RAW,
    DATA_PROCESSED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build service-request hex-level features."
    )
    parser.add_argument(
        "--sr-path",
        type=Path,
        default=SR_HEX,
        help="Path to sr_hex.csv.gz",
    )
    parser.add_argument(
        "--hex-path",
        type=Path,
        default=CITY_HEX_POLYGONS_8,
        help="Path to city hex polygons",
    )
    parser.add_argument(
        "--buildings-path",
        type=Path,
        default=GOOGLE_BUILDINGS_RAW,
        help="Path to raw Google Buildings dataset",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="Sewer: Blocked/Overflow",
        help="Request type to model",
    )
    parser.add_argument(
        "--min-building-confidence",
        type=float,
        default=0.0,
        help="Optional minimum Google Buildings confidence threshold",
    )
    return parser.parse_args()


def detect_request_type_column(df: pd.DataFrame) -> str:
    candidates = ["code", "request_type", "type", "service_request_type"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find request type column. Expected one of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def detect_hex_id_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = ["index", "h3_index", "hex_id", "h3_level8_index"]
    for col in candidates:
        if col in gdf.columns:
            return col
    raise ValueError(
        f"Could not find hex id column in hex GeoJSON. Expected one of {candidates}. "
        f"Available columns: {list(gdf.columns)}"
    )


def load_service_requests(sr_path: Path) -> tuple[pd.DataFrame, str]:
    print(f"Loading service requests from: {sr_path}")
    df = pd.read_csv(sr_path, compression="gzip")

    request_type_col = detect_request_type_column(df)

    if "h3_level8_index" not in df.columns:
        raise ValueError(
            "Expected 'h3_level8_index' column in service requests data, but it was not found."
        )

    df = df.copy()
    df["h3_level8_index"] = df["h3_level8_index"].astype(str)
    df = df[df["h3_level8_index"] != "0"].copy()
    df[request_type_col] = df[request_type_col].astype(str)

    print(f"Loaded {len(df):,} valid geolocated service requests.")
    return df, request_type_col


def load_hexes(hex_path: Path) -> tuple[gpd.GeoDataFrame, str]:
    print(f"Loading hex polygons from: {hex_path}")
    hex_gdf = gpd.read_file(hex_path)

    if hex_gdf.crs is None:
        hex_gdf = hex_gdf.set_crs("EPSG:4326")
    else:
        hex_gdf = hex_gdf.to_crs("EPSG:4326")

    hex_id_col = detect_hex_id_column(hex_gdf)
    hex_gdf = hex_gdf.copy()
    hex_gdf[hex_id_col] = hex_gdf[hex_id_col].astype(str)

    print(f"Loaded {len(hex_gdf):,} hex polygons.")
    return hex_gdf, hex_id_col


def build_type_hex_counts(
    sr_df: pd.DataFrame,
    request_type_col: str,
) -> pd.DataFrame:
    print("Building long-format request counts table for task 2.1...")

    counts = (
        sr_df.groupby(["h3_level8_index", request_type_col], dropna=False)
        .size()
        .reset_index(name="request_count")
        .rename(columns={request_type_col: "request_type"})
        .sort_values(["h3_level8_index", "request_type"])
        .reset_index(drop=True)
    )

    print(f"Created task 2.1 table with {len(counts):,} rows.")
    return counts


def build_sr_derived_features(
    sr_df: pd.DataFrame,
    hex_gdf: gpd.GeoDataFrame,
    hex_id_col: str,
    request_type_col: str,
    target_type: str,
) -> pd.DataFrame:
    print(f"Building SR-derived features for target type: {target_type}")

    all_hexes = pd.DataFrame(
        {"h3_level8_index": hex_gdf[hex_id_col].astype(str).unique()}
    )

    total_requests = (
        sr_df.groupby("h3_level8_index")
        .size()
        .reset_index(name="total_requests")
    )

    request_diversity = (
        sr_df.groupby("h3_level8_index")[request_type_col]
        .nunique()
        .reset_index(name="request_diversity")
    )

    target_requests = (
        sr_df.loc[sr_df[request_type_col] == target_type]
        .groupby("h3_level8_index")
        .size()
        .reset_index(name="sewer_requests")
    )

    features = (
        all_hexes.merge(total_requests, on="h3_level8_index", how="left")
        .merge(request_diversity, on="h3_level8_index", how="left")
        .merge(target_requests, on="h3_level8_index", how="left")
    )

    for col in ["total_requests", "request_diversity", "sewer_requests"]:
        features[col] = features[col].fillna(0).astype(int)

    print(f"Built SR-derived feature table for {len(features):,} hexes.")
    return features


def read_buildings_csv(
    buildings_path: Path,
    min_confidence: float = 0.0,
) -> gpd.GeoDataFrame:
    print(f"Loading Google Buildings data from: {buildings_path}")

    df = pd.read_csv(
        buildings_path,
        compression="gzip",
        usecols=["latitude", "longitude", "area_in_meters", "confidence"],
    )

    required_cols = {"latitude", "longitude", "area_in_meters", "confidence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Google Buildings file is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["latitude", "longitude", "area_in_meters", "confidence"]).copy()

    if min_confidence > 0:
        df = df[df["confidence"] >= min_confidence].copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    print(f"Loaded {len(gdf):,} building points after filtering.")
    return gdf


def build_google_building_features(
    buildings_path: Path,
    hex_gdf: gpd.GeoDataFrame,
    hex_id_col: str,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    print("Building Google Buildings features per hex...")

    buildings_gdf = read_buildings_csv(
        buildings_path=buildings_path,
        min_confidence=min_confidence,
    )

    joined = gpd.sjoin(
        buildings_gdf,
        hex_gdf[[hex_id_col, "geometry"]],
        how="inner",
        predicate="within",
    )

    building_features = (
        joined.groupby(hex_id_col)
        .agg(
            building_count=("area_in_meters", "size"),
            mean_building_area=("area_in_meters", "mean"),
            sd_building_area=("area_in_meters", "std"),
        )
        .reset_index()
        .rename(columns={hex_id_col: "h3_level8_index"})
    )

    all_hexes = pd.DataFrame(
        {"h3_level8_index": hex_gdf[hex_id_col].astype(str).unique()}
    )

    building_features = all_hexes.merge(
        building_features, on="h3_level8_index", how="left"
    )

    building_features["building_count"] = (
        building_features["building_count"].fillna(0).astype(int)
    )
    building_features["mean_building_area"] = (
        building_features["mean_building_area"].fillna(0.0)
    )
    building_features["sd_building_area"] = (
        building_features["sd_building_area"].fillna(0.0)
    )

    print(f"Built Google Buildings feature table for {len(building_features):,} hexes.")
    return building_features


def assemble_modelling_table(
    sr_features: pd.DataFrame,
    building_features: pd.DataFrame,
) -> pd.DataFrame:
    print("Assembling final modelling table...")

    df = sr_features.merge(building_features, on="h3_level8_index", how="left")

    df["building_count"] = df["building_count"].fillna(0).astype(int)
    df["mean_building_area"] = df["mean_building_area"].fillna(0.0)
    df["sd_building_area"] = df["sd_building_area"].fillna(0.0)

    print(f"Final modelling table has {len(df):,} rows and {df.shape[1]} columns.")
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main() -> None:
    args = parse_args()

    output_dir = DATA_PROCESSED
    output_dir.mkdir(parents=True, exist_ok=True)

    sr_type_hex_counts_path = output_dir / "sr_type_hex_counts.csv"
    hex_building_features_path = output_dir / "hex_building_features.csv"
    sewer_hex_features_path = output_dir / "sewer_hex_features.csv"

    sr_df, request_type_col = load_service_requests(args.sr_path)
    hex_gdf, hex_id_col = load_hexes(args.hex_path)

    type_hex_counts = build_type_hex_counts(
        sr_df=sr_df,
        request_type_col=request_type_col,
    )

    sr_features = build_sr_derived_features(
        sr_df=sr_df,
        hex_gdf=hex_gdf,
        hex_id_col=hex_id_col,
        request_type_col=request_type_col,
        target_type=args.target_type,
    )

    building_features = build_google_building_features(
        buildings_path=args.buildings_path,
        hex_gdf=hex_gdf,
        hex_id_col=hex_id_col,
        min_confidence=args.min_building_confidence,
    )

    modelling_table = assemble_modelling_table(
        sr_features=sr_features,
        building_features=building_features,
    )

    save_csv(type_hex_counts, sr_type_hex_counts_path)
    save_csv(building_features, hex_building_features_path)
    save_csv(modelling_table, sewer_hex_features_path)

    print("\nDone.")
    print("Outputs:")
    print(f"  - {sr_type_hex_counts_path}")
    print(f"  - {hex_building_features_path}")
    print(f"  - {sewer_hex_features_path}")


if __name__ == "__main__":
    main()
