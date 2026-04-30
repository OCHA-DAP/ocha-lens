from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from shapely import Polygon
from shapely.geometry import MultiPolygon, Point, box

QUADS = ["ne", "se", "sw", "nw"]
NM_TO_M = 1.852 * 1000
BUFFER_SPEEDS = [34, 50, 64]

GEO_CRS_MERIDIAN = "+proj=longlat +datum=WGS84 +lon_wrap=0"
GEO_CRS_ANTIMERIDIAN = "+proj=longlat +datum=WGS84 +lon_wrap=180"
GEO_CRS_ANTIMERIDIAN_NEGATIVE = "+proj=longlat +datum=WGS84 +lon_wrap=-180"

BASIN_GEO_CRS = {
    "EP": GEO_CRS_ANTIMERIDIAN_NEGATIVE,
    "NA": GEO_CRS_MERIDIAN,
    "NI": GEO_CRS_MERIDIAN,
    "SI": GEO_CRS_MERIDIAN,
    "SA": GEO_CRS_MERIDIAN,
    "WP": GEO_CRS_ANTIMERIDIAN,
    "SP": GEO_CRS_ANTIMERIDIAN,
}

PROJ_CRS_MERIDIAN = "EPSG:3857"
PROJ_CRS_ANTIMERIDIAN = "EPSG:3832"

BASIN_PROJ_CRS = {
    "EP": PROJ_CRS_ANTIMERIDIAN,
    "NA": PROJ_CRS_MERIDIAN,
    "NI": PROJ_CRS_MERIDIAN,
    "SI": PROJ_CRS_MERIDIAN,
    "SA": PROJ_CRS_MERIDIAN,
    "WP": PROJ_CRS_ANTIMERIDIAN,
    "SP": PROJ_CRS_ANTIMERIDIAN,
}


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
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame must be EPSG:4326")

    work = gdf.copy()

    work[time_col] = pd.to_datetime(work[time_col], utc=True)
    work = work.sort_values(time_col).drop_duplicates(subset=[time_col])
    work = work[~work.geometry.is_empty]

    work["lat"] = work.geometry.y
    work["lon"] = work.geometry.x

    n = len(work)

    # --- ensure output schema matches input
    numeric_cols = work.select_dtypes(include=[np.number]).columns.difference(
        ["lat", "lon"]
    )

    if n == 0:
        return gpd.GeoDataFrame(columns=work.columns, crs=4326)

    # --- target time grid
    tmin, tmax = work[time_col].min(), work[time_col].max()

    start = tmin.floor(freq) if include_ends else tmin.ceil(freq)
    end = tmax.ceil(freq) if include_ends else tmax.floor(freq)

    target = pd.date_range(start, end, freq=freq, tz="UTC")
    target = target[(target >= tmin) & (target <= tmax)]

    if target.empty:
        target = pd.DatetimeIndex([tmin, tmax])

    t0 = work[time_col].iloc[0]

    x = (work[time_col] - t0).dt.total_seconds().to_numpy()
    x_new = (pd.Series(target) - t0).dt.total_seconds().to_numpy()

    out = pd.DataFrame(index=target)

    # --- lat/lon handling
    lat = work["lat"].to_numpy(float)
    lon = antimeridian_safe_lon(work["lon"].to_numpy(float))

    if n == 1:
        lat_new = np.repeat(lat[0], len(x_new))
        lon_new = np.repeat(lon[0], len(x_new))

    elif n == 2:
        lat_new = np.interp(x_new, x, lat)
        lon_new = np.interp(x_new, x, lon)

    else:
        lat_new = PchipInterpolator(x, lat)(x_new)
        lon_new = PchipInterpolator(x, lon)(x_new)

    lon_new = ((lon_new + 180) % 360) - 180

    out["lat"] = lat_new
    out["lon"] = lon_new

    # --- other numeric columns
    for col in numeric_cols:
        y = work[col].to_numpy(float)

        if n == 1:
            out[col] = np.repeat(y[0], len(x_new))
        else:
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

    # check if all new columns are already present
    new_cols = [f"{col}_{q}" for q in quads]
    if all(c in df.columns for c in new_cols):
        print("All expanded columns already exist. Skipping expansion.")
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


def _radius_from_quadrants(
    theta_deg: np.ndarray, ne: float, se: float, sw: float, nw: float
) -> np.ndarray:
    """
    Return radius for each angle by linearly interpolating between the
    four quadrant control points defined at bearings:
        45°  -> NE
        135° -> NW
        225° -> SW
        315° -> SE
    Bearing convention: 0° = East, 90° = North (mathematical).
    """
    # Control bearings (deg) and radii, with wrap-around point to close the loop
    bearings = np.array([45, 135, 225, 315, 405], dtype=float)
    radii = np.array([ne, nw, sw, se, ne], dtype=float)

    # Map all thetas into [0, 360) and also allow values up to 405 for interpolation
    t = (theta_deg % 360).astype(float)
    # For values in [0,45), make an equivalent in [360,405) to interpolate to NE nicely
    t_wrap = t.copy()
    t_wrap[t < 45] += 360

    # Interpolate and then map back (the interpolation function is periodic due to control duplication)
    r = np.interp(t_wrap, bearings, radii)
    return r


def make_quadrant_disk(
    center_xy: Tuple[float, float],
    ne: float,
    se: float,
    sw: float,
    nw: float,
    n_points: int = 360,
) -> Polygon:
    """
    Build a smooth polygon around (x, y) using quadrant radii. Units assumed meters.
    - center_xy: (x, y) in EPSG:3832
    - ne, se, sw, nw: radii for quadrants (meters)
    - n_points: angular resolution
    Bearing convention: 0°=East, 90°=North; polygon traced counter-clockwise.
    """
    x0, y0 = center_xy
    theta = np.linspace(0, 360, n_points, endpoint=False)  # degrees
    r = _radius_from_quadrants(theta, ne, se, sw, nw)

    # Convert polar -> Cartesian
    th = np.deg2rad(theta)
    xs = x0 + r * np.cos(th)
    ys = y0 + r * np.sin(th)

    # Ensure valid ring: close the polygon
    coords = np.column_stack([xs, ys])
    return Polygon(coords)


def build_merged_wind_buffer(
    gdf: gpd.GeoDataFrame,
    quad_cols: Tuple[str, str, str, str],
):
    """
    Build a merged wind buffer polygon from quadrant radii columns.
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataFrame with point geometries and quadrant radius columns
    quad_cols: Tuple[str, str, str, str]
        Names of the four quadrant radius columns in order:
        (ne_col, se_col, sw_col, nw_col)

    Returns
    -------
    gpd.GeoSeries or None
        Merged polygon of wind buffers, or None if all radius values are NaN

    """
    ne_col, se_col, sw_col, nw_col = quad_cols
    polys = []
    gdf[[ne_col, se_col, sw_col, nw_col]] = (
        gdf[[ne_col, se_col, sw_col, nw_col]].fillna(0) * NM_TO_M
    )
    for _, row in gdf.iterrows():
        if row[[ne_col, se_col, sw_col, nw_col]].isna().all():
            return None

        poly = make_quadrant_disk(
            (row.geometry.x, row.geometry.y),
            row[ne_col],
            row[se_col],
            row[sw_col],
            row[nw_col],
        )
        polys.append(poly)
    return gpd.GeoSeries(polys).union_all()


def calculate_wind_buffers_gdf(
    gdf: gpd.GeoDataFrame,
    quad_cols_format: str = "usa_quadrant_radius_{speed}_{quad}",
    valid_time_col: str = "valid_time",
):
    """
    Calculate wind buffer polygons for given wind speed quadrants.
    Note that this function interpolates the storm track to a regular
    30-minute interval before calculating the wind buffers.
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with storm track data including quadrant radius columns
    quad_cols_format: str = 'quadrant_radius_{speed}_{quad}'
        Format string for quadrant radius columns, with placeholders for
        speed and quad (e.g., 'quadrant_radius_{speed}_{quad}')
    lon_col: str = 'Longitude'
        Name of the longitude column in df
    lat_col: str = 'Latitude'
        Name of the latitude column in df
    valid_time_col: str = 'valid_time'
        Name of the valid time column in df

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with wind buffer polygons for each speed

    """
    basin = gdf["basin"].iloc[0] if "basin" in gdf.columns else None
    geo_crs = BASIN_GEO_CRS.get(basin, GEO_CRS_MERIDIAN)
    proj_crs = BASIN_PROJ_CRS.get(basin, PROJ_CRS_MERIDIAN)

    gdf_interp = interpolate_track(
        gdf,
        time_col=valid_time_col,
    )
    # Reproject via basin-appropriate lon_wrap CRS so antimeridian-crossing
    # tracks have continuous longitudes before projecting to Mercator.
    gdf_interp = gdf_interp.to_crs(geo_crs)
    gdf_interp = gdf_interp.to_crs(proj_crs)
    dicts = []
    geoms = []
    for speed in BUFFER_SPEEDS:
        speed_quad_cols = tuple(
            quad_cols_format.format(speed=speed, quad=x) for x in QUADS
        )
        geoms.append(build_merged_wind_buffer(gdf_interp, speed_quad_cols))
        dicts.append({"speed": speed})
    result = gpd.GeoDataFrame(dicts, geometry=geoms, crs=proj_crs).to_crs(
        "EPSG:4326"
    )
    world = box(-180, -90, 180, 90)
    result.geometry = result.geometry.intersection(world)
    return result


def _filled_geom(geom):
    """Return geometry with interior rings removed (fills donut holes)."""
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior.coords) for p in geom.geoms])
    return geom


def _best_atcf_for_polygon(geom, tracks_at_time: gpd.GeoDataFrame) -> str:
    """Return the atcf_id whose track points are most contained within geom.

    Falls back to nearest centroid if no track points land inside the filled
    polygon (e.g. coarse track resolution relative to a small polygon).
    """
    filled = _filled_geom(geom)
    counts = (
        tracks_at_time.assign(inside=tracks_at_time.geometry.within(filled))
        .groupby("atcf_id")["inside"]
        .sum()
    )
    if counts.max() > 0:
        return counts.idxmax()
    centroids = tracks_at_time.groupby("atcf_id").geometry.apply(
        lambda gs: gs.union_all().centroid
    )
    return centroids.apply(lambda c: geom.centroid.distance(c)).idxmin()


def match_wsp_to_tracks(
    gdf_wsp: gpd.GeoDataFrame,
    gdf_tracks: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Match WSP polygons to NHC track forecasts by issued_time.

    For issued_times with a single active storm the polygon is assigned
    directly.  For multiple active storms the MultiPolygon is exploded into
    individual components and each is matched to the track whose forecast
    points fall inside it (using hole-filled geometry to handle donut shapes).

    Parameters
    ----------
    gdf_wsp:
        Rows from storms.nhc_wsp_polygon.
        Required columns: id, issued_time, wind_threshold_kt, percentage, geometry.
    gdf_tracks:
        Rows from storms.nhc_tracks_geo.
        Required columns: atcf_id, issued_time, geometry (Point, EPSG:4326).

    Returns
    -------
    GeoDataFrame
        gdf_wsp with an added ``atcf_id`` column.  For multi-storm
        issued_times the MultiPolygon rows are exploded into individual polygon
        rows (one per storm).  Rows whose issued_time has no corresponding
        tracks are returned with ``atcf_id=None``.
    """
    gdf_wsp = gdf_wsp[gdf_wsp.geometry.notna()].copy()

    track_counts = gdf_tracks.groupby("issued_time")["atcf_id"].nunique()
    single_times = track_counts[track_counts == 1].index
    multi_times = track_counts[track_counts > 1].index

    # --- single-storm: direct merge ---
    single_atcf = gdf_tracks[
        gdf_tracks["issued_time"].isin(single_times)
    ].drop_duplicates("issued_time")[["issued_time", "atcf_id"]]
    gdf_single = gdf_wsp[gdf_wsp["issued_time"].isin(single_times)].merge(
        single_atcf, on="issued_time", how="left"
    )

    # --- multi-storm: explode then spatial match ---
    gdf_multi = gdf_wsp[gdf_wsp["issued_time"].isin(multi_times)].copy()
    if not gdf_multi.empty:
        gdf_multi_exp = gdf_multi.explode(index_parts=False).reset_index(
            drop=True
        )
        atcf_ids = []
        for _, row in gdf_multi_exp.iterrows():
            tracks_at_time = gdf_tracks[
                gdf_tracks["issued_time"] == row["issued_time"]
            ]
            atcf_ids.append(
                _best_atcf_for_polygon(row.geometry, tracks_at_time)
            )
        gdf_multi_exp["atcf_id"] = atcf_ids
    else:
        gdf_multi_exp = gdf_multi.copy()
        gdf_multi_exp["atcf_id"] = pd.Series(dtype="object")

    # --- no-track times: return with atcf_id=None ---
    tracked_times = set(single_times) | set(multi_times)
    gdf_no_track = gdf_wsp[~gdf_wsp["issued_time"].isin(tracked_times)].copy()
    gdf_no_track["atcf_id"] = None

    return gpd.GeoDataFrame(
        pd.concat(
            [gdf_single, gdf_multi_exp, gdf_no_track], ignore_index=True
        ),
        crs=gdf_wsp.crs,
    )
