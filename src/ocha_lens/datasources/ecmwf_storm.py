import io
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import lxml.etree as et
import ocha_stratus as stratus
import pandas as pd
import requests
from dateutil import rrule

logger = logging.getLogger(__name__)

BASIN_MAPPING = {
    "Northwest Pacific": "WP",
    "Southwest Pacific": "SP",
    "Northeast Pacific": "EP",
}


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
        print(f"using cache for {outfile}")
        if outfile.exists():
            print(f"{file} already exists locally")
            return outfile
        elif (
            stratus.get_container_client(blob_container)
            .get_blob_client(str(outfile))
            .exists()
        ):
            print(f"{file} already exists")
            return outfile

    # Exit if we don't want to download from the server
    if skip_if_missing:
        print(f"file isn't saved! {file}")
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
        logger.debug(f"Processing for {date}...")
        raw_file = download_hindcasts(
            date, save_dir, use_cache, skip_if_missing, blob_container
        )
        if raw_file:
            dfs.append(
                _process_cxml_to_df(
                    raw_file, xsl_path="temp/cxml_ecmwf_transformation.xsl"
                )
            )

    return pd.concat(dfs)


def get_storms(df):
    df_storms = (
        df.groupby("storm_id")[
            ["number", "name", "basin", "valid_time", "provider"]
        ]
        .first()
        .reset_index()
    )
    df_storms["season"] = df_storms.valid_time.dt.year
    df_storms = df_storms.rename(columns={"basin": "genesis_basin"}).drop(
        columns=["valid_time"]
    )
    return df_storms


def get_forecast_tracks(df):
    df_tracks = df.drop(columns=["provider", "name", "number"])
    df_tracks["point_id"] = [str(uuid.uuid4()) for _ in range(len(df_tracks))]
    return df_tracks


def _process_cxml_to_df(cxml_path: str, xsl_path: str = None):
    """Read a cxml v1.1 file; may not work on newer specs."""
    if xsl_path is None:
        xsl_path = "data/cxml_ecmwf_transformation.xsl"

    cxml_data = cxml_path
    if not cxml_path.exists():
        cxml_data = stratus.load_blob_data(
            str(cxml_path), container_name="storm"
        )

    xsl = et.parse(str(xsl_path))
    # Handle bytes, string path, and Path object inputs
    if isinstance(cxml_data, bytes):
        xml = et.parse(io.BytesIO(cxml_data))
    elif isinstance(cxml_data, (str, Path)):
        xml = et.parse(str(cxml_data))
    else:
        raise ValueError(
            "cxml_data must be bytes, a file path string, or a Path object"
        )

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

    df["basin"] = df["basin"].map(BASIN_MAPPING)

    df = df.rename(
        columns={
            "id": "storm_id",
            "baseTime": "issued_time",
            "validTime": "valid_time",
            "hour": "leadtime",
            "cycloneName": "name",
            "cycloneNumber": "number",
            "minimumPressure": "pressure",
            "lastClosedIsobar": "last_closed_isobar_radius",
            "maximumWind": "wind_speed",
            "maximumWindRadius": "maximum_wind_radius",
            "origin": "provider",
        }
    ).drop(columns=["disturbance_no", "type", "member", "perturb"])

    return df
