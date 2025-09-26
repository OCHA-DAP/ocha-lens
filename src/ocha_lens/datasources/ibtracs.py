import logging
import os
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import List, Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pandera.pandas as pa
import xarray as xr

from ocha_lens.utils.storm import (
    _create_storm_id,
    _normalize_longitude,
    _to_gdf,
    check_coordinate_bounds,
    check_crs,
    check_quadrant_list,
)

logger = logging.getLogger(__name__)


STORM_SCHEMA = pa.DataFrameSchema(
    {
        "sid": pa.Column(str, nullable=False),
        "atcf_id": pa.Column(str, nullable=True),
        "number": pa.Column("int16", nullable=False),
        "season": pa.Column(
            "int64", pa.Check.between(1840, 2100), nullable=False
        ),
        "name": pa.Column(str, nullable=True),
        "genesis_basin": pa.Column(str, nullable=False),
        "provisional": pa.Column(bool, nullable=False),
        "storm_id": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
    unique=["sid", "storm_id"],
    report_duplicates="all",
)

TRACK_SCHEMA = pa.DataFrameSchema(
    {
        # TODO: Investigate condition where wind speed is -1
        "wind_speed": pa.Column(
            "Int64", pa.Check.between(-1, 300), nullable=True
        ),
        "pressure": pa.Column(
            "Int64", pa.Check.between(800, 1100), nullable=True
        ),
        "max_wind_radius": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "last_closed_isobar_radius": pa.Column(
            "Int64", pa.Check.ge(0), nullable=True
        ),
        "last_closed_isobar_pressure": pa.Column(
            "Int64", pa.Check.between(800, 1100), nullable=True
        ),
        "gust_speed": pa.Column(
            "Int64", pa.Check.between(0, 400), nullable=True
        ),
        "sid": pa.Column(str, nullable=False),
        "provider": pa.Column(str, nullable=False),
        "basin": pa.Column(str, nullable=False),
        "nature": pa.Column(str, nullable=True),
        "valid_time": pa.Column(pd.Timestamp, nullable=False),
        "quadrant_radius_34": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=False
        ),
        "quadrant_radius_50": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=False
        ),
        "quadrant_radius_64": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=True
        ),
        "point_id": pa.Column(str, nullable=False),
        "storm_id": pa.Column(str, nullable=True),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=False),
    },
    strict=True,
    coerce=True,
    unique=["sid", "storm_id", "valid_time"],
    report_duplicates="all",
    checks=[
        pa.Check(
            lambda gdf: check_crs(gdf, "EPSG:4326"),
            error="CRS must be EPSG:4326",
        ),
        pa.Check(
            lambda gdf: check_coordinate_bounds(gdf),
            error="All coordinates must be within valid lat/lon bounds",
        ),
    ],
)


def download_ibtracs(
    dataset: Literal[
        "ALL", "ACTIVE", "last3years", "EP", "NA", "NI", "SA", "SI", "SP", "WP"
    ] = "ALL",
    save_dir: Optional[str] = "storm",
) -> Path:
    """
    Download IBTrACS data to a specified or temporary directory.

    Parameters
    ----------
    dataset : {"ALL", "ACTIVE", "last3years", "EP", "NA", "NI", "SA", "SI", "SP", "WP"}, default "ALL"
        Which IBTrACS dataset to download:
        - "ALL": Complete historical record
        - "ACTIVE": Records for active storms only
        - "last3years": Records from the past three years
        - "EP": Eastern North Pacific basin
        - "NA": North Atlantic basin
        - "NI": North Indian basin
        - "SA": South Atlantic basin
        - "SI": South Indian basin
        - "SP": South Pacific basin
        - "WP": Western North Pacific basin
    save_dir : str, optional
        Directory to download to.

    Returns
    -------
    Path
        Path to the downloaded file
    """
    save_dir = Path(save_dir) / "raw"
    os.makedirs(save_dir, exist_ok=True)

    url = (
        "https://www.ncei.noaa.gov/data/"
        "international-best-track-archive-for-climate-stewardship-ibtracs/"
        f"v04r01/access/netcdf/IBTrACS.{dataset}.v04r01.nc"
    )

    filename = f"IBTrACS.{dataset}.v04r01.nc"
    download_path = save_dir / filename
    urllib.request.urlretrieve(url, download_path)

    return download_path


def load_ibtracs(
    file_path: Optional[str] = None,
    dataset: Literal[
        "ALL", "ACTIVE", "last3years", "EP", "NA", "NI", "SA", "SI", "SP", "WP"
    ] = "ALL",
) -> xr.Dataset:
    """
    Load IBTrACS data from NetCDF file or download to a temporary directory.

    Parameters
    ----------
    file_path : str, optional
        Path to the IBTrACS NetCDF file. If None, downloads the file to a temp directory.
    dataset : {"ALL", "ACTIVE", "last3years", "EP", "NA", "NI", "SA", "SI", "SP", "WP"}, default "ALL"
        Which IBTrACS dataset to download:
        - "ALL": Complete historical record
        - "ACTIVE": Records for active storms only
        - "last3years": Records from the past three years
        - "EP": Eastern North Pacific basin
        - "NA": North Atlantic basin
        - "NI": North Indian basin
        - "SA": South Atlantic basin
        - "SI": South Indian basin
        - "SP": South Pacific basin
        - "WP": Western North Pacific basin

        Only used if file_path is None.

    Returns
    -------
    xarray.Dataset
        Dataset containing IBTrACS data with dimensions (storm, date_time, quadrant)
    """
    if file_path is None:
        # Use a temporary directory that automatically cleans up
        with tempfile.TemporaryDirectory(prefix="ibtracs_data_") as temp_dir:
            file_path = download_ibtracs(dataset=dataset, save_dir=temp_dir)

            # Load the dataset and ensure it's fully loaded into memory
            # since temp_dir will be removed after this block
            ds = xr.open_dataset(file_path).load()
            return ds
    else:
        return xr.open_dataset(file_path)


def get_storms(ds: xr.Dataset) -> pd.DataFrame:
    """
    Extract storm metadata from IBTrACS dataset.

    Creates a dataset with one row per storm containing identifying information.
    This provides a summary of all storms in the dataset with their basic metadata.

    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset containing storm track data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing storm metadata with one row per storm

    Notes
    -----
    The function takes the first available metadata for each storm when multiple
    records exist. This works because storm metadata is generally consistent
    across a storm's lifetime.
    """
    storm_cols = [
        "sid",
        "usa_atcf_id",
        "number",
        "season",
        "name",
        "track_type",
        "basin",
    ]
    str_vars = ["sid", "name", "track_type", "usa_atcf_id", "basin"]

    ds_subset = ds[storm_cols]
    ds_subset[str_vars] = ds_subset[str_vars].astype(str)
    df = ds_subset.to_dataframe().reset_index()
    df = df.replace(b"", pd.NA).replace("", pd.NA)
    cols = storm_cols
    df = df[cols]

    df["provisional"] = df["track_type"].apply(lambda x: x == "PROVISIONAL")
    df_grouped = (
        df.groupby("sid").first().reset_index().drop(columns=["track_type"])
    )

    df_grouped["name"] = df_grouped["name"].replace("UNNAMED", pd.NA)
    df_grouped = df_grouped.rename(
        columns={"usa_atcf_id": "atcf_id", "basin": "genesis_basin"}
    )
    df_grouped["season"] = df_grouped["season"].astype(int)
    df_grouped["storm_id"] = df_grouped.apply(_create_storm_id, axis=1)

    return STORM_SCHEMA.validate(df_grouped)


def get_tracks(ds: xr.Dataset, track_type: str = "all") -> gpd.GeoDataFrame:
    """
    Extract track data from IBTrACS source data. Users should be cautious of
    comparing wind speed measurements from storms with different providers
    (eg. as may be the case in provisional vs best tracks), as different
    providers use different averaging periods.

    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset containing storm track data
    track_type: {"provisional", "best", "all"}
        Which subset of tracks to return

    Returns
    -------
    pandas.DataFrame
        DataFrame containing track data with standardized column names
    """
    if track_type == "provisional":
        return _get_provisional_tracks(ds)
    elif track_type == "best":
        return _get_best_tracks(ds)
    elif track_type == "all":
        df_provisional = _get_provisional_tracks(ds)
        df_best = _get_best_tracks(ds)
        return TRACK_SCHEMA.validate(pd.concat([df_provisional, df_best]))
    else:
        logger.error(
            f"Invalid track type: {track_type}. Must be either `provisional` or `best`."
        )
        return


def normalize_radii(df, radii_cols=None):
    """
    Convert radii data from separate quadrant rows to list format.

    This function converts radius data that's stored with separate rows for each quadrant
    into a single row per storm point with radius values stored as lists.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing storm track data with radii columns and quadrant information
    radii_cols : list of str, optional
        List of column names containing radii data. If None, defaults to
        ["r34", "r50", "r64"]

    Returns
    -------
    pandas.DataFrame
        DataFrame with radii data converted to lists for each point where each list
        contains values for the 4 quadrants (TODO - Confirm the ordering)
    """
    if not radii_cols:
        radii_cols = ["r34", "r50", "r64"]

    dff = df.copy()
    group_columns = [
        col
        for col in dff.columns
        if col not in radii_cols and col != "quadrant"
    ]

    result_df = (
        dff.drop(columns=radii_cols + ["quadrant"])
        .drop_duplicates(subset=group_columns)
        .reset_index(drop=True)
    )

    for col in radii_cols:
        pivot_df = dff.pivot(
            index=group_columns, columns="quadrant", values=col
        )
        lists = pivot_df.apply(lambda x: x.tolist(), axis=1)
        result_df[col] = (
            result_df.set_index(group_columns).index.map(lists).values
        )
    return result_df


def _get_provisional_tracks(ds: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Extract provisional storm tracks from the IBTrACS dataset.

    Extracts tracks marked as PROVISIONAL in the dataset, which typically
    contain USA agency data only. These are often recent storms that haven't
    yet been fully processed by the WMO.

    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset containing storm track data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing provisional track data with standardized column names
    """
    usa_cols = [
        "usa_r34",
        "usa_r50",
        "usa_r64",
        "usa_lat",
        "usa_lon",
        "usa_wind",
        "usa_pres",
        "usa_rmw",
        "usa_roci",
        "usa_poci",
        "usa_gust",
    ]

    other_cols = ["sid", "track_type", "usa_agency", "basin", "nature"]
    string_cols = ["sid", "track_type", "usa_agency", "basin", "nature"]

    ds_ = ds[usa_cols + other_cols]
    provisional_mask = ds_.track_type == b"PROVISIONAL"  # If stored as bytes

    if not provisional_mask.any():
        return pd.DataFrame(columns=list(TRACK_SCHEMA.columns.keys()))

    ds_ = ds_.where(provisional_mask, drop=True)
    df = ds_.to_dataframe().reset_index()
    df = _convert_string_columns(df, string_cols)
    # Dropping interpolated points that don't have an assigned usa_agency
    df = (
        df.replace(b"", pd.NA)
        .replace("", pd.NA)
        .dropna(subset=["time", "usa_agency"])
    )
    cols = usa_cols + other_cols + ["time", "quadrant"]
    df = df[cols]

    r_columns = ["usa_r34", "usa_r50", "usa_r64"]
    result_df = normalize_radii(df, r_columns)

    result_df.rename(
        columns={
            "time": "valid_time",
            "usa_lat": "latitude",
            "usa_lon": "longitude",
            "usa_wind": "wind_speed",
            "usa_pres": "pressure",
            "usa_gust": "gust_speed",
            "usa_rmw": "max_wind_radius",
            "usa_roci": "last_closed_isobar_radius",
            "usa_poci": "last_closed_isobar_pressure",
            "usa_agency": "provider",
            "usa_r34": "quadrant_radius_34",
            "usa_r50": "quadrant_radius_50",
            "usa_r64": "quadrant_radius_64",
        },
        inplace=True,
    )
    result_df = result_df.drop(columns=["track_type"])
    # TODO: Probably a bit overkill in the ids here (and also not really readable)
    # Should think about how to best improve
    result_df["point_id"] = [str(uuid.uuid4()) for _ in range(len(result_df))]

    # Need to get the storms df to join in the storm_id that we created
    storms = get_storms(ds)
    merged_df = result_df.merge(storms[["sid", "storm_id"]], how="left")
    assert len(merged_df) == len(result_df)

    df = _convert_track_column_types(merged_df)
    df = _normalize_longitude(df)
    gdf = _to_gdf(df)
    if len(gdf) == 0:
        logger.warning("Returning empty geodataframe of provisional tracks")
        return gdf
    return TRACK_SCHEMA.validate(gdf)


def _get_best_tracks(ds: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Extract the "best" storm tracks from the IBTrACS dataset.

    Extracts the main tracks that have been assigned a wmo_agency and
    are not marked as PROVISIONAL. These are the official, quality-controlled
    tracks used for most analyses.

    Parameters
    ----------
    ds : xarray.Dataset
        IBTrACS dataset containing storm track data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing best track data with standardized column names

    Notes
    -----
    The function handles agency-specific variables by prioritizing data from the
    agency designated as the WMO agency for each storm. For agencies that are part
    of the USA system, the function uses the corresponding USA data.
    """
    variables = [
        "wind",
        "gust",
        "pres",
        "rmw",
        "roci",
        "poci",
        "lat",
        "lon",
        "r34",
        "r50",
        "r64",
    ]

    base_cols = [
        "sid",
        "wmo_agency",
        "time",
        "quadrant",
        "basin",
        "nature",
        "track_type",
    ]

    string_cols = ["sid", "wmo_agency", "basin", "nature", "track_type"]

    base_ds = ds[base_cols]
    df = base_ds.to_dataframe().reset_index()

    # Seems like checking the track_type is redundant here since provisional tracks
    # also don't have an assigned wmo_agency, but still good to be sure
    df = df[(df["wmo_agency"] != b"") & (df["track_type"] != b"PROVISIONAL")]
    if len(df) == 0:
        return pd.DataFrame(columns=list(TRACK_SCHEMA.columns.keys()))

    df = _convert_string_columns(df, string_cols)

    dff = df[base_cols].copy()
    usa_agencies = [x.decode("utf-8") for x in np.unique(ds.usa_agency.values)]

    for var_suffix in variables:
        matching_vars = [
            v for v in ds.data_vars if v.endswith(f"_{var_suffix}")
        ]
        dff[var_suffix] = np.nan

        for var in matching_vars:
            # We assume that vars are named `agency`_`var`
            agency = var.split("_")[0]
            if agency == "usa":
                # The USA agencies don't each have columns in their own right
                # This data is under the usa_* prefix
                mask = df["wmo_agency"].isin(usa_agencies)
            else:
                mask = df["wmo_agency"].str.lower() == agency

            if not mask.any():
                continue

            # Get the data from the original dataset and add to result
            # Need to map back to original indices for selection
            for idx in df[mask].index:
                storm_val = df.loc[idx, "storm"]
                dt_val = df.loc[idx, "date_time"]

                # Only the radii variables are also indexed by quadrant
                if any(substr in var for substr in ["r34", "r50", "r64"]):
                    quadrant_val = df.loc[idx, "quadrant"]
                    value = (
                        ds[var]
                        .sel(
                            storm=storm_val,
                            date_time=dt_val,
                            quadrant=quadrant_val,
                        )
                        .values.item()
                    )
                else:
                    value = (
                        ds[var]
                        .sel(storm=storm_val, date_time=dt_val)
                        .values.item()
                    )
                if not np.isnan(value):
                    dff.loc[idx, var_suffix] = value

    result_df = normalize_radii(dff)
    result_df.rename(
        columns={
            "time": "valid_time",
            "lat": "latitude",
            "lon": "longitude",
            "wind": "wind_speed",
            "gust": "gust_speed",
            "pres": "pressure",
            "rmw": "max_wind_radius",
            "roci": "last_closed_isobar_radius",
            "poci": "last_closed_isobar_pressure",
            "wmo_agency": "provider",
            "r34": "quadrant_radius_34",
            "r50": "quadrant_radius_50",
            "r64": "quadrant_radius_64",
        },
        inplace=True,
    )
    result_df = result_df.drop(columns=["track_type"])
    result_df["point_id"] = [str(uuid.uuid4()) for _ in range(len(result_df))]
    # Drop points that have null values in non-nullable columns
    result_df = result_df.dropna(
        subset=["latitude", "longitude", "valid_time", "sid"]
    )

    # Need to get the storms df to join in the storm_id that we created
    storms = get_storms(ds)
    merged_df = result_df.merge(storms[["sid", "storm_id"]], how="left")
    assert len(merged_df) == len(result_df)

    df = _convert_track_column_types(merged_df)
    df = _normalize_longitude(df)
    gdf = _to_gdf(df)
    if len(gdf) == 0:
        logger.warning("Returning empty geodataframe of best tracks")
        return gdf
    return TRACK_SCHEMA.validate(gdf)


def _convert_string_columns(
    df: pd.DataFrame, string_cols: List[str]
) -> pd.DataFrame:
    """
    Convert byte columns to strings and handle empty values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns that need conversion
    string_cols : list of str
        List of column names to convert from bytes to strings

    Returns
    -------
    pandas.DataFrame
        DataFrame with converted string columns
    """
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    return df


def _convert_track_column_types(df):
    df_ = df.copy()
    df_ = df_.round({"latitude": 2, "longitude": 2})
    df_["valid_time"] = df_["valid_time"].dt.round("min")
    return df_
