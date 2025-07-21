import io
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import lxml.etree as et
import ocha_stratus as stratus
import pandas as pd
import requests
from dateutil import rrule
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# TEMP
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# TODO: Keep confirming this
BASIN_MAPPING = {
    "Northwest Pacific": "WP",
    "Northeast Pacific": "EP",
    "North Atlantic": "NA",
    "North Indian": "NI",
    "South Indian": "SI",
    "South Pacific": "SP",
}


CXML2CSV_XSL = Path(__file__).parent / "data/cxml_ecmwf_transformation.xsl"


def download_hindcasts(
    date,
    save_dir="storm",
    use_cache=False,
    skip_if_missing=True,
    stage="local",  # dev, prod, or local
):
    """
    Downloads historical ECMWF data from TIGGE in XML format
    https://rda.ucar.edu/datasets/d330003/dataaccess/#
    """

    filename = _get_raw_filename(date)
    base_filename = os.path.basename(filename)
    base_outfile = f"xml/raw/{os.path.basename(base_filename)}"
    outfile = Path(save_dir) / base_outfile

    # Don't download if exists already
    if use_cache:
        logger.debug(f"using cache for {outfile}")
        if stage == "local":
            if outfile.exists():
                logger.debug(f"{base_filename} already exists locally")
                return outfile
        elif stage == "dev" or stage == "prod":
            if (
                stratus.get_container_client(save_dir, stage=stage)
                .get_blob_client(base_outfile)
                .exists()
            ):
                logger.debug(f"{base_filename} already exists in blob")
                return outfile
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
        outfile.parent.mkdir(parents=True, exist_ok=True)
        open(outfile, "wb").write(req.content)
    elif stage == "dev" or stage == "prod":
        logger.debug(f"Saving to {stage} blob")
        stratus.upload_blob_data(
            req.content, base_outfile, container_name=save_dir, stage=stage
        )
    else:
        logger.error(f"Invalid stage: {stage}")
        return
    return outfile


def get_storms(df):
    df = df.copy()
    df["name"] = df["name"].str.upper()
    df["season"] = df.apply(_convert_season, axis=1)
    df["basin"] = df.apply(_convert_basin, axis=1)

    # We're only identifying storms that have names
    df_ = df.dropna(subset="name").copy()
    df_.loc[:, "storm_id"] = df_.apply(_create_storm_id, axis=1)
    df_ = df_.sort_values("issued_time")
    df_forecasts = (
        df_.groupby(["id", "issued_time", "name", "number"])[
            ["storm_id", "provider", "season", "basin"]
        ]
        .first()
        .reset_index()
    )

    # Note that a single storm may have different numbers during its forecast lifecycle
    # We're picking the one from the last forecast
    # TODO: Check that we're not dropping different storms that have ended up with the same id??
    df_storms = df_forecasts.sort_values(
        "issued_time", ascending=False
    ).drop_duplicates(subset=["storm_id"])
    df_storms = df_storms.drop(columns=["id", "issued_time"])
    df_storms = df_storms.rename(columns={"basin": "genesis_basin"})
    return df_storms


def load_hindcasts(
    start_date: datetime = None,
    end_date: datetime = None,
    temp_dir="storm",
    use_cache=True,
    skip_if_missing=False,
    stage="dev",
):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=1)).date()
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).date()

    save_dir = Path(temp_dir) if temp_dir else Path("temp")
    os.makedirs(save_dir, exist_ok=True)

    date_list = rrule.rrule(
        rrule.HOURLY,
        dtstart=start_date,
        until=end_date + timedelta(hours=12),
        interval=12,
    )

    dfs = []
    for date in date_list:
        logger.info(f"Processing for {date}...")
        raw_file = download_hindcasts(
            date, save_dir, use_cache, skip_if_missing, stage
        )
        if raw_file:
            df = _process_cxml_to_df(raw_file, stage, save_dir)
            if df is not None:
                dfs.append(df)
    if len(dfs) > 0:
        return pd.concat(dfs)
    logger.error("No data available for input dates")
    return


def get_tracks(df):
    # Some duplication of effort here, but want to keep API similar with IBTrACS
    df_storms = get_storms(df)
    df_tracks = df.merge(df_storms[["name", "season", "storm_id"]], how="left")
    assert len(df_tracks) == len(df)
    df_tracks["basin"] = df_tracks.apply(_convert_basin, axis=1)
    df_tracks = df_tracks.drop(columns=["season", "name", "number"])
    df_tracks = df_tracks.rename(columns={"id": "forecast_id"})
    df_tracks["point_id"] = [str(uuid.uuid4()) for _ in range(len(df_tracks))]
    df_tracks["pressure"] = df_tracks["pressure"].astype("float64")
    gdf_tracks = gpd.GeoDataFrame(
        df_tracks,
        geometry=[
            Point(xy) for xy in zip(df_tracks.longitude, df_tracks.latitude)
        ],
        crs="EPSG:4326",
    )
    gdf_tracks = gdf_tracks.drop(["latitude", "longitude"], axis=1)
    assert len(gdf_tracks == len(df))

    return gdf_tracks


def _process_cxml_to_df(cxml_path: str, stage, save_dir, xsl_path: str = None):
    """Adapted from
    https://github.com/CLIMADA-project/climada_petals/blob/6381a3c90dc9f1acd1e41c95f826d7dd7f623fff/climada_petals/hazard/tc_tracks_forecast.py#L627.  # noqa
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
            cxml_path = str(Path(*Path(cxml_path).parts[1:]))
            cxml_data = stratus.load_blob_data(
                cxml_path, container_name=save_dir
            )
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
    df.dropna(
        subset=["validTime", "latitude", "longitude"], how="any", inplace=True
    )
    # Remove all ensemble forecasts
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


def _create_storm_id(row):
    name_part = str(row["number"]) if pd.isna(row["name"]) else row["name"]
    storm_id = f"{name_part}_{row['basin']}_{row['season']}".lower()
    return storm_id


def _convert_basin(row):
    basin = row["basin"]
    if row["basin"] == "Southwest Pacific":
        basin = "South Indian" if row["longitude"] <= 135 else "South Pacific"
    try:
        standard_basin = BASIN_MAPPING[basin]
    except Exception:
        logger.warning(f"Unexpected input basin: {basin}")
        standard_basin = basin
    return standard_basin


def _convert_season(row):
    season = row["valid_time"].year
    basin = row["basin"]
    is_southern_hemisphere = (
        "south" in basin.lower() if isinstance(basin, str) else False
    )
    is_july_or_later = row["valid_time"].month >= 7
    if is_southern_hemisphere and is_july_or_later:
        season += 1
    return season
