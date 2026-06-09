import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def _to_gdf(df):
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)],
        crs="EPSG:4326",
    )
    gdf = gdf.drop(["latitude", "longitude"], axis=1)
    return gdf


def _create_storm_id(row):
    if pd.notna(row["name"]) and row["name"]:
        return f"{row['name']}_{row['genesis_basin']}_{row['season']}".lower()
    return row["name"]


def _normalize_longitude(df, longitude_col="longitude"):
    """
    Convert longitude values >180° back to -180 to 180° range.
    """
    df_normalized = df.copy()
    mask = df_normalized[longitude_col] > 180
    df_normalized.loc[mask, longitude_col] = (
        df_normalized.loc[mask, longitude_col] - 360
    )
    return df_normalized


def _convert_season(row):
    """
    Follows convention to use the subsequent year if the cyclone is in the
    southern hemisphere and occurring after June. Relies on "south"/"South"
    being present in the input basin.
    """
    season = row["valid_time"].year
    basin = row["basin"]
    is_southern_hemisphere = (
        "south" in basin.lower() if isinstance(basin, str) else False
    )
    is_july_or_later = row["valid_time"].month >= 7
    if is_southern_hemisphere and is_july_or_later:
        season += 1
    return season


def check_crs(gdf, expected_crs="EPSG:4326"):
    """Check if GeoDataFrame has the expected CRS."""
    if gdf.crs is None:
        return False
    return str(gdf.crs) == expected_crs or gdf.crs.to_string() == expected_crs


def check_quadrant_list(series):
    """Check if each value is a list of exactly 4 elements."""

    def validate_item(x):
        return isinstance(x, list) and len(x) == 4

    return series.apply(validate_item).all()


def check_coordinate_bounds(gdf):
    """Check if all Point geometries have valid lat/lon coordinates."""

    def validate_point(geom):
        # if geom is None or pd.isna(geom):
        #     return False  # No null geometries allowed
        if hasattr(geom, "x") and hasattr(geom, "y"):
            return (-180 <= geom.x <= 180) and (-90 <= geom.y <= 90)
        return False

    return gdf.geometry.apply(validate_point).all()


def check_unique_when_storm_id_not_null(df):
    # Filter out rows where storm_id is null
    filtered_df = df.dropna(subset=["storm_id"])

    # Check if the combination is unique in the filtered data
    duplicates = filtered_df.duplicated(
        subset=["storm_id", "valid_time", "leadtime"]
    )
    return ~duplicates.any()
