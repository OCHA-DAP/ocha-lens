import io
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import lxml.etree as et
import ocha_stratus as stratus
import pandas as pd
import pandera.pandas as pa
import requests
from dateutil import rrule

from ocha_lens.utils.storm import (
    _convert_season,
    _create_storm_id,
    _normalize_longitude,
    _to_gdf,
    check_coordinate_bounds,
    check_crs,
    check_unique_when_storm_id_not_null,
)

logger = logging.getLogger(__name__)

BASIN_MAPPING = {
    "Northwest Pacific": "WP",
    "Northeast Pacific": "EP",
    "North Atlantic": "NA",
    "North Indian": "NI",
    "South Indian": "SI",
    "South Pacific": "SP",
}

# Conversion factor from m/s to knots
KTS_CONVERSION = 1.944

UNIQUE_COLS = [
    "forecast_id",
    "valid_time",
    "leadtime",
    "issued_time",
    "number",
    "basin",
    "pressure",
    "wind_speed",
    "storm_id",
    "geometry",
]

STORM_SCHEMA = pa.DataFrameSchema(
    {
        "name": pa.Column(str, nullable=True),
        "number": pa.Column(str, nullable=False),
        "storm_id": pa.Column(str, nullable=False),
        "provider": pa.Column(str, nullable=True),
        "season": pa.Column(int, pa.Check.between(2005, 2050)),
        "genesis_basin": pa.Column(
            str, pa.Check.isin(list(BASIN_MAPPING.values())), nullable=True
        ),
    },
    strict=True,
    coerce=True,
    unique=["storm_id"],
    report_duplicates="all",
)

TRACK_SCHEMA = pa.DataFrameSchema(
    {
        "issued_time": pa.Column(pd.Timestamp),
        "provider": pa.Column(str, nullable=False),
        "forecast_id": pa.Column(str, nullable=False),
        "number": pa.Column(str, nullable=True),
        "basin": pa.Column(
            str, pa.Check.isin(list(BASIN_MAPPING.values())), nullable=False
        ),
        "leadtime": pa.Column("Int64", pa.Check.ge(0)),
        "valid_time": pa.Column(pd.Timestamp),
        "pressure": pa.Column(
            float, pa.Check.between(800, 1100), nullable=True
        ),
        "wind_speed": pa.Column(
            float, pa.Check.between(0, 300), nullable=True
        ),
        "storm_id": pa.Column(str, nullable=True),
        "point_id": pa.Column(str, nullable=False),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=False),
    },
    strict=True,
    coerce=True,
    unique=UNIQUE_COLS,
    checks=[
        pa.Check(
            lambda gdf: check_crs(gdf, "EPSG:4326"),
            error="CRS must be EPSG:4326",
        ),
        pa.Check(
            lambda gdf: check_coordinate_bounds(gdf),
            error="All coordinates must be within valid lat/lon bounds",
        ),
        pa.Check(
            check_unique_when_storm_id_not_null,
            raise_warning=True,
            error="""
                Duplicate combination of storm_id, valid_time, leadtime found (excluding null storm_ids).
                This likely means that there are duplicate forecasts, but with slightly different
                position or intensity values for some points. These 'near duplicates' are maintained in the
                output dataset. Use outputs with caution.
                """,
        ),
    ],
)


CXML2CSV_XSL = Path(__file__).parent / "data/cxml_ecmwf_transformation.xsl"


def download_forecasts(
    date: datetime,
    cache_dir: str = "storm",
    use_cache: bool = False,
    skip_if_missing: bool = False,
    stage: Literal["dev", "prod", "local"] = "local",
) -> Optional[Path]:
    """
    Download historical ECMWF data from TIGGE in XML format
    from https://rda.ucar.edu/datasets/d330003/dataaccess/#

    Data can be saved locally or uploaded to Azure blob storage
    depending on the stage parameter.

    Parameters
    ----------
    date : datetime
        The datetime for which to download forecast data
    cache_dir : str, default "storm"
        Directory or container name to store raw cxml files. Refers to
        a container name if stage is "dev" or "prod". Assumed to be a single
        string rather than a full path. (#TODO: consider allowing full paths?).
        If writing to Azure, the container must already exist.
    use_cache : bool, default False
        Whether to check for existing files before downloading
    skip_if_missing : bool, default False
        If True, skip download if file doesn't exist on server rather than downloading
    stage : {"dev", "prod", "local"}, default "local"
        Where to save the downloaded data:
        - "local": Save to local filesystem
        - "dev": Upload to development Azure blob storage
        - "prod": Upload to production Azure blob storage

    Returns
    -------
    Path or None
        Path to the downloaded file if successful, None if download failed
    """

    filename = _get_raw_filename(date)
    base_filename = os.path.basename(filename)
    base_download_path = f"xml/raw/{os.path.basename(base_filename)}"
    download_path = Path(cache_dir) / base_download_path

    # Don't download if exists already
    if use_cache:
        logger.debug(f"using cache for {download_path}")
        if stage == "local":
            if download_path.exists():
                logger.debug(f"{base_filename} already exists locally")
                return download_path
        elif stage == "dev" or stage == "prod":
            if (
                stratus.get_container_client(cache_dir, stage=stage)
                .get_blob_client(base_download_path)
                .exists()
            ):
                logger.debug(f"{base_filename} already exists in blob")
                return download_path
        else:
            logger.error(f"Invalid stage: {stage}")
            return

    # If file doesn't exist and we don't want to check the server
    if skip_if_missing:
        logger.debug(f"file isn't saved! {base_filename}")
        return

    # Now download
    try:
        logger.debug(f"{base_filename} downloading")
        req = requests.get(filename, timeout=(10, None))
    except Exception as e:
        logger.error(e)
        return
    if req.status_code != 200:
        logger.debug(f"{base_filename} invalid URL")
        return

    if stage == "local":
        logger.debug("saving locally")
        download_path.parent.mkdir(parents=True, exist_ok=True)
        open(download_path, "wb").write(req.content)
    elif stage == "dev" or stage == "prod":
        logger.debug(f"Saving to {stage} blob")
        stratus.upload_blob_data(
            req.content,
            base_download_path,
            container_name=cache_dir,
            stage=stage,
        )
    else:
        logger.error(f"Invalid stage: {stage}")
        return
    return download_path


def load_forecasts(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    cache_dir: str = "storm",
    use_cache: bool = True,
    skip_if_missing: bool = False,
    stage: Literal["dev", "prod", "local"] = "local",
) -> Optional[pd.DataFrame]:
    """
    Load ECMWF tropical cyclone hindcast data for a date range.

    Downloads and processes ECMWF forecast data from TIGGE for the specified
    date range. Data is downloaded at 12-hour intervals and processed into
    a standardized format.

    Default behaviour is to locally save downloaded files to "storm/" directory,
    and load from there if they already exist. Optionally, data can be saved to
    or loaded from Azure blob storage containers by setting the stage parameter.

    Parameters
    ----------
    start_date : datetime, optional
        Start date for data retrieval. If None, defaults to yesterday
    end_date : datetime, optional
        End date for data retrieval. If None, defaults to yesterday
    cache_dir : str, default "storm"
        Directory or container name to store raw cxml files. Refers to
        a container name if stage is "dev" or "prod". Assumed to be a single
        string rather than a full path. (#TODO: consider allowing full paths?)
        If writing to Azure, the container must already exist.
    use_cache : bool, default True
        Whether to use cached files if they exist
    skip_if_missing : bool, default False
        Whether to skip dates where files are missing on the server. Set to True if
        you're pulling from what you know is a full cache.
    stage : {"dev", "prod", "local"}, default "local"
        Storage location for downloaded files. "dev" or "prod" refer to
        internal Azure blob storage containers.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing processed forecast data with columns including
        issued_time, valid_time, latitude, longitude, pressure, wind_speed, etc.
        Returns None if no data is available for the specified date range
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1)).date()
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).date()

    date_list = rrule.rrule(
        rrule.HOURLY,
        dtstart=start_date,
        until=end_date + timedelta(hours=12),
        interval=12,
    )

    dfs = []
    for date in date_list:
        logger.info(f"Processing for {date}...")
        raw_file = download_forecasts(
            date, cache_dir, use_cache, skip_if_missing, stage
        )
        if raw_file:
            df = _process_cxml_to_df(raw_file, stage)
            if df is not None:
                dfs.append(df)
    if len(dfs) > 0:
        return pd.concat(dfs)
    logger.error("No data available for input dates")
    return


def get_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes ECMWF tropical cyclone forecast data to create a forecasts dataset
    with one row per forecast containing identifying information. Only storms with
    names are included in the output.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing raw ECMWF forecast data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing storm metadata with with one row per storm

    Notes
    -----
    Storm IDs are created using the format "{name/number}_{basin}_{season}".
    For storms with multiple forecasts, metadata is taken from the most recent forecast.
    Season calculation accounts for Southern Hemisphere cyclone seasons.
    """
    df_ = df.copy()
    df_["name"] = df_["name"].str.upper()
    df_["season"] = df_.apply(_convert_season, axis=1)

    # Make sure time is all in a consistent format
    # in 2022 ECMWF switched from timezone-aware to not
    df_["issued_time"] = pd.to_datetime(
        df_["issued_time"].astype(str), utc=True, format="mixed"
    )

    # We're not grouping by just id,
    # since some forecast id's aren't unique
    df_sorted = df_.sort_values(["issued_time", "id"])
    df_forecasts = (
        df_sorted.groupby(["id", "number", "basin"])[
            [
                "provider",
                "season",
                "name",
                "issued_time",
                "latitude",
                "longitude",
            ]
        ]
        .first()
        .reset_index()
    )
    df_forecasts["genesis_basin"] = df_forecasts.apply(_convert_basin, axis=1)
    df_forecasts = df_forecasts.drop(
        columns=["basin", "latitude", "longitude"]
    )
    df_forecasts.loc[:, "storm_id"] = df_forecasts.apply(
        _create_storm_id, axis=1
    )
    return df_forecasts


def get_storms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes ECMWF tropical cyclone forecast data to create a storms dataset
    with one row per storm containing identifying information. Only storms with
    names are included in the output.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing raw ECMWF forecast data

    Returns
    -------
    pandas.DataFrame
        DataFrame containing storm metadata with with one row per storm
    """
    df_forecasts = get_forecasts(df)
    # Only keep named storms
    df_forecasts_ = df_forecasts.dropna(subset="name").copy()
    # Note that a single storm may have different numbers during its forecast lifecycle
    # We're picking the one from the last forecast
    # TODO: Check that we're not dropping different storms that have ended up with the same id??
    df_storms = df_forecasts_.drop_duplicates(subset=["storm_id"])
    df_storms = df_storms.drop(columns=["id", "issued_time"])
    return STORM_SCHEMA.validate(df_storms)


def get_tracks(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Extract tropical cyclone track data from ECMWF forecast data.

    Processes ECMWF forecast data to create a tracks dataset with individual
    forecast points as rows. Each point contains storm information, forecast
    metadata, and geometric location data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing processed ECMWF forecast data

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing track data with standardized column names and
        geometry points for each location
    """
    df_ = df.copy()

    # Merge in the storm_ids
    df_forecasts = get_forecasts(df_)
    df_["basin"] = df_.apply(_convert_basin, axis=1)
    df_tracks = df_.merge(
        df_forecasts[["id", "number", "storm_id", "genesis_basin"]],
        left_on=["id", "number", "basin"],
        right_on=["id", "number", "genesis_basin"],
        how="left",
    )
    assert len(df_tracks) == len(df_)

    # Basic column transformation
    # Keep the genesis_basin since the basin info is at the forecast level
    df_tracks = df_tracks.drop(columns=["name", "genesis_basin"])
    df_tracks = df_tracks.rename(columns={"id": "forecast_id"})
    df_tracks["point_id"] = [str(uuid.uuid4()) for _ in range(len(df_tracks))]

    # Make sure time is all in a consistent format
    # in 2022 ECMWF switched from timezone-aware to not
    df_tracks["issued_time"] = pd.to_datetime(
        df_tracks["issued_time"].astype(str), utc=True, format="mixed"
    )

    # Drop all rows where our primary variables of interest are 0
    # assuming this is some data quality issue
    mask = (
        (df_tracks["pressure"] == 0)
        & (df_tracks["wind_speed"] == 0)
        & (df_tracks["latitude"] == 0)
        & (df_tracks["longitude"] == 0)
    )
    df_tracks_dropped = df_tracks.drop(df_tracks[mask].index)

    diff = len(df_tracks) - len(df_tracks_dropped)
    if diff > 0:
        logger.warning(
            f"Dropped {diff} tracks with invalid positional and intensity values"
        )

    # Normalize coordinates and convert to gdf
    df_tracks_dropped = _normalize_longitude(df_tracks_dropped)
    gdf_tracks = _to_gdf(df_tracks_dropped)

    assert len(gdf_tracks == len(df_))
    # Need to handle duplicates here. Some archival ECMWF forecasts
    # have duplicates coming from the raw CXML files. Duplicates may
    # be in the same source file, or as result of some forecasts being
    # in incorrectly labelled files. We'll drop duplicates in cases where
    # ALL timing, position, and intensity values are the same.
    gdf_tracks_dropped = gdf_tracks.drop_duplicates(
        subset=UNIQUE_COLS, keep="first"
    )
    duplicated = gdf_tracks[
        gdf_tracks.duplicated(subset=UNIQUE_COLS, keep=False)
    ]
    diff = len(gdf_tracks) - len(gdf_tracks_dropped)
    if diff > 0:
        logger.warning(
            f"Dropped {diff} duplicate tracks based on unique columns"
        )
        logger.warning(
            f"Duplicated forecast IDs: {duplicated['forecast_id'].unique()}"
        )
    return TRACK_SCHEMA.validate(gdf_tracks_dropped)


def _process_cxml_to_df(
    cxml_path: Union[str, Path],
    stage: Literal["dev", "prod", "local"],
    xsl_path: Optional[Union[str, Path]] = None,
) -> Optional[pd.DataFrame]:
    """
    Process TIGGE CXML (cyclone XML) file to DataFrame format. Input data
    may either be local or in Azure blob storage. Adapted from
    https://github.com/CLIMADA-project/climada_petals/blob/6381a3c90dc9f1acd1e41c95f826d7dd7f623fff/climada_petals/hazard/tc_tracks_forecast.py#L627.  # noqa

    Parameters
    ----------
    cxml_path : str or Path
        Path to the CXML file to process
    stage : {"dev", "prod", "local"}
        Source location of the file
    xsl_path : str or Path, optional
        Path to XSL transformation file. If None, uses default transformation

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing processed cyclone data, or None if processing failed

    Notes
    -----
    Removes ensemble forecasts, keeping only deterministic forecasts.
    """
    if xsl_path is None:
        xsl_path = CXML2CSV_XSL
    xsl = et.parse(str(xsl_path))
    transformer = et.XSLT(xsl)

    try:
        if stage == "local":
            xml = et.parse(str(cxml_path))
        elif stage == "dev" or stage == "prod":
            # Remove the first directory level since this is the container
            container, path = str(cxml_path).split("/", 1)
            cxml_data = stratus.load_blob_data(path, container_name=container)
            xml = et.parse(io.BytesIO(cxml_data))
        else:
            logger.error(f"Invalid stage: {stage}")
            return
    except Exception as e:
        logger.error(f"Error parsing cxml: {e}")
        return

    csv_string = str(transformer(xml))

    df = pd.read_csv(
        io.StringIO(csv_string),
        dtype={
            "member": "Int64",
            "cycloneNumber": "object",
            "hour": "Int64",
            "cycloneName": "object",
            "id": "object",
        },
    )

    df["baseTime"] = pd.to_datetime(df["baseTime"])
    df["validTime"] = pd.to_datetime(df["validTime"])
    df["maximumWind"] = df["maximumWind"] * KTS_CONVERSION
    df.dropna(
        subset=["validTime", "latitude", "longitude"], how="any", inplace=True
    )
    # Remove all ensemble forecasts and "analysis"
    df = df[df.type == "forecast"]

    # TODO: Confirm that lastClosedIsobar and maximumWindRadius are always null
    df = df.rename(
        columns={
            "baseTime": "issued_time",
            "validTime": "valid_time",
            "hour": "leadtime",
            "cycloneName": "name",
            "cycloneNumber": "number",
            "minimumPressure": "pressure",
            "maximumWind": "wind_speed",
            "origin": "provider",
        }
    ).drop(
        columns=[
            "disturbance_no",
            "type",
            "member",
            "perturb",
            "lastClosedIsobar",
            "maximumWindRadius",
        ]
    )
    return df


def _get_raw_filename(date):
    dspath = "https://data.rda.ucar.edu/d330003/"
    ymd = date.strftime("%Y%m%d")
    ymdhms = date.strftime("%Y%m%d%H%M%S")
    server = "test" if date < datetime(2008, 8, 1) else "prod"
    file = (
        f"ecmf/{date.year}/{ymd}/z_tigge_c_ecmf_{ymdhms}_"
        f"ifs_glob_{server}_all_glo.xml"
    )
    return dspath + file


def _convert_basin(row):
    basin = row["basin"]
    if row["basin"] == "Southwest Pacific":
        basin = (
            "South Indian"
            if (row["longitude"] > 0 and row["longitude"] <= 135)
            else "South Pacific"
        )
    try:
        standard_basin = BASIN_MAPPING[basin]
    except Exception:
        logger.warning(f"Unexpected input basin: {basin}")
        standard_basin = basin
    return standard_basin
