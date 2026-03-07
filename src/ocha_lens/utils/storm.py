import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Point

QUADS = ["ne", "se", "sw", "nw"]


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


def interpolate_track(
    gdf: gpd.GeoDataFrame,
    time_col: str = "valid_time",
    freq: str = "30min",
    include_ends: bool = True,
) -> gpd.GeoDataFrame:
    """
    Resample a point GeoDataFrame track to a regular time grid.

    - Geometry must be POINT in EPSG:4326
    - Lat/lon interpolated with PCHIP
    - Other numeric columns interpolated linearly
    - Handles dateline crossings safely
    """

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame must be EPSG:4326")

    work = gdf.copy()

    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.sort_values(time_col).drop_duplicates(subset=[time_col])
    work = work[~work.geometry.is_empty]

    work["lat"] = work.geometry.y
    work["lon"] = work.geometry.x

    n = len(work)

    if n == 0:
        return gpd.GeoDataFrame(columns=[time_col, "geometry"], crs=4326)

    if n == 1:
        row = work.iloc[0]
        return gpd.GeoDataFrame(
            {time_col: [row[time_col]]},
            geometry=[Point(row["lon"], row["lat"])],
            crs=4326,
        )

    # --- target time grid
    tmin, tmax = work[time_col].min(), work[time_col].max()

    start = tmin.floor(freq) if include_ends else tmin.ceil(freq)
    end = tmax.ceil(freq) if include_ends else tmax.floor(freq)

    target = pd.date_range(start, end, freq=freq, tz="UTC")
    target = target[(target >= tmin) & (target <= tmax)]

    if target.empty:
        target = pd.DatetimeIndex([tmin, tmax])

    # --- time axis
    t0 = work[time_col].iloc[0]

    x = (work[time_col] - t0).dt.total_seconds().to_numpy()
    x_new = (pd.Series(target) - t0).dt.total_seconds().to_numpy()

    # --- interpolate lat
    y_lat = work["lat"].to_numpy(float)
    lat_interp = PchipInterpolator(x, y_lat)
    lat_new = lat_interp(x_new)

    # --- interpolate lon (dateline-safe)
    y_lon = antimeridian_safe_lon(work["lon"].to_numpy(float))
    lon_interp = PchipInterpolator(x, y_lon)
    lon_new = lon_interp(x_new)

    # convert back to standard range
    lon_new = ((lon_new + 180) % 360) - 180

    # --- other numeric columns
    other_cols = work.select_dtypes(include=[np.number]).columns.difference(
        ["lat", "lon"]
    )

    out = pd.DataFrame(index=target)
    out["lat"] = lat_new
    out["lon"] = lon_new

    for col in other_cols:
        y = work[col].to_numpy(float)
        out[col] = np.interp(x_new, x, y)

    out.index.name = time_col
    out = out.reset_index()

    geometry = gpd.points_from_xy(out["lon"], out["lat"])

    out = gpd.GeoDataFrame(
        out.drop(columns=["lat", "lon"]),
        geometry=geometry,
        crs=4326,
    )

    return out


def antimeridian_safe_lon(lon):
    """
    Convert longitude array to a continuous sequence suitable for plotting.

    Handles antimeridian crossings by:
    1. unwrapping longitude
    2. shifting to the optimal 360° window around the track center

    Parameters
    ----------
    lon : array-like
        Longitudes in degrees (any range).

    Returns
    -------
    numpy.ndarray
        Continuous longitudes suitable for plotting.
    """

    lon = np.asarray(lon)

    lon_unwrap = np.rad2deg(np.unwrap(np.deg2rad(lon)))

    center = lon_unwrap.mean()

    lon_shift = lon_unwrap - 360 * np.round((lon_unwrap - center) / 360)

    return lon_shift


def expand_quad_col(df, col, quads=None):
    if quads is None:
        quads = QUADS

    if f"{col}_{quads[0]}" in df:
        print(f"already done for {col}")
        return df

    def parse_quad(x):
        if pd.isna(x):
            return [np.nan] * len(quads)

        x = x.strip("{}")
        vals = x.split(",")

        return [np.nan if v in ("NaN", "", "nan") else float(v) for v in vals]

    df_expanded = df[col].apply(parse_quad).apply(pd.Series)

    df_expanded.columns = [f"{col}_{q}" for q in quads]

    return df.join(df_expanded)
