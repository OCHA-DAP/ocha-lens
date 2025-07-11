import io
import logging
import os
import uuid
from datetime import datetime
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

# TODO -- or is this even feasible?
BASIN_MAPPING = {
    "Northwest Pacific": "WP",
    "Southwest Pacific": "SP",
    "Northeast Pacific": "EP",
    "North Atlantic": "NA",
    "North Indian": "NI",
}


CXML2CSV_XSL = Path(__file__).parent / "data/cxml_ecmwf_transformation.xsl"


def download_hindcasts(
    date,
    save_dir="xml",
    use_cache=False,
    skip_if_missing=False,
    blob_container="storm",
):
    """
    Downloads historical ECMWF data from TIGGE in XML format
    https://rda.ucar.edu/datasets/d330003/dataaccess/#
    """
    dspath = "https://data.rda.ucar.edu/d330003/"
    ymd = date.strftime("%Y%m%d")
    ymdhms = date.strftime("%Y%m%d%H%M%S")
    server = "test" if date < datetime(2008, 8, 1) else "prod"
    file = (
        f"ecmf/{date.year}/{ymd}/z_tigge_c_ecmf_{ymdhms}_"
        f"ifs_glob_{server}_all_glo.xml"
    )
    filename = dspath + file
    outfile = Path(save_dir) / "raw" / os.path.basename(filename)

    # Don't download if exists already
    if use_cache:
        logger.debug(f"using cache for {outfile}")
        if outfile.exists():
            logger.debug(f"{file} already exists locally")
            return outfile
        elif (
            stratus.get_container_client(blob_container)
            .get_blob_client(str(outfile))
            .exists()
        ):
            logger.debug(f"{file} already exists")
            return outfile

    # Exit if we don't want to download from the server
    if skip_if_missing:
        logger.debug(f"file isn't saved! {file}")
        return

    # Now download
    try:
        req = requests.get(filename, timeout=(10, None))
    except Exception as e:
        logger.error(e)
        return
    if req.status_code != 200:
        logger.debug(f"{file} invalid URL")
        return
    logger.debug(f"{file} downloading")
    open(outfile, "wb").write(req.content)
    if blob_container:
        logger.debug("Saving to blob")
        stratus.upload_blob_data(
            req.content, str(outfile), container_name=blob_container
        )
    return outfile


# TODO: Give option to load from blob as well
def load_hindcasts(
    start_date: datetime = datetime(2025, 1, 1).date(),
    end_date=datetime.now().date(),
    temp_dir="xml",
    use_cache=True,
    skip_if_missing=False,
    blob_container="storm",
):
    save_dir = Path(temp_dir) if temp_dir else Path("temp")
    os.makedirs(save_dir, exist_ok=True)

    date_list = rrule.rrule(
        rrule.HOURLY,
        dtstart=start_date,
        until=end_date,
        interval=12,
    )

    dfs = []
    for date in date_list:
        logger.info(f"Processing for {date}...")
        raw_file = download_hindcasts(
            date, save_dir, use_cache, skip_if_missing, blob_container
        )
        if raw_file:
            df = _process_cxml_to_df(raw_file)
            if df is not None:
                dfs.append(df)

    return pd.concat(dfs)


def get_storms(df):
    # First let's get all the distinct forecasts
    df = df.sort_values(by="issued_time", ascending=False)
    df_forecasts = (
        df.groupby("id")[
            [
                "number",
                "name",
                "basin",
                "valid_time",
                "issued_time",
                "provider",
            ]
        ]
        .first()
        .reset_index()
    )
    df_forecasts = _convert_season(df_forecasts).rename(
        columns={"basin": "genesis_basin"}
    )
    df_forecasts = df_forecasts.sort_values(by="issued_time", ascending=False)
    df_forecasts["name"] = df_forecasts["name"].fillna(df_forecasts["number"])

    # Now group together any forecasts that look like they're from the same storm
    df_storms = (
        df_forecasts.groupby(["season", "name", "genesis_basin"])
        .first()
        .reset_index()
    ).drop(columns="valid_time")
    df_storms["storm_id"] = df_storms["name"].str.cat(
        [df_storms["genesis_basin"], df_storms["season"].astype(str)], sep="_"
    )
    return df_storms


def get_forecast_tracks(df, df_storms):
    df_tracks = df.merge(df_storms[["id", "storm_id"]])
    df_tracks = df_tracks.drop(columns=["provider", "name", "number"])
    df_tracks["point_id"] = [str(uuid.uuid4()) for _ in range(len(df_tracks))]
    gdf_tracks = gpd.GeoDataFrame(
        df_tracks,
        geometry=[
            Point(xy) for xy in zip(df_tracks.longitude, df_tracks.latitude)
        ],
        crs="EPSG:4326",
    )
    gdf_tracks = gdf_tracks.drop(["latitude", "longitude"], axis=1)
    return gdf_tracks


def _process_cxml_to_df(cxml_path: str, xsl_path: str = None):
    """Adapted from
    https://github.com/CLIMADA-project/climada_petals/blob/6381a3c90dc9f1acd1e41c95f826d7dd7f623fff/climada_petals/hazard/tc_tracks_forecast.py#L627.  # noqa
    """
    if xsl_path is None:
        xsl_path = CXML2CSV_XSL

    cxml_data = cxml_path
    if not cxml_path.exists():
        cxml_data = stratus.load_blob_data(
            str(cxml_path), container_name="storm"
        )

    xsl = et.parse(str(xsl_path))
    # Handle bytes, string path, and Path object inputs
    try:
        if isinstance(cxml_data, bytes):
            xml = et.parse(io.BytesIO(cxml_data))
        elif isinstance(cxml_data, (str, Path)):
            xml = et.parse(str(cxml_data))
        else:
            raise ValueError(
                "cxml_data must be bytes, a file path string, or a Path object"
            )
    except Exception as e:
        logger.error(f"Error parsing cxml: {e}")
        return

    transformer = et.XSLT(xsl)
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
        parse_dates=["baseTime", "validTime"],
    )

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


def _convert_season(df):
    df_ = df.copy()
    df_["season"] = df_["valid_time"].dt.year
    southern_hemisphere_mask = (
        df_["basin"].str.lower().str.contains("south", na=False)
    )
    july_1_mask = df_["valid_time"].dt.month >= 7
    df_.loc[southern_hemisphere_mask & july_1_mask, "season"] = (
        df_.loc[southern_hemisphere_mask & july_1_mask, "season"] + 1
    )
    return df_
