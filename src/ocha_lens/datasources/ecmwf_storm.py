import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import lxml.etree as et
import pandas as pd
import requests
from dateutil import rrule

logger = logging.getLogger(__name__)

BASIN_MAPPING = {
    "Northwest Pacific": "WP",
    "Southwest Pacific": "SP",
    "Northeast Pacific": "EP",
}


def download_hindcasts(date, save_dir):
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
    outfile = save_dir / "xml" / os.path.basename(filename)
    # Don't download if exists already
    if outfile.exists():
        logger.debug(f"{file} already exists")
        return outfile
    req = requests.get(filename, allow_redirects=True)
    if req.status_code != 200:
        logger.debug(f"{file} invalid URL")
        return
    logger.debug(f"{file} downloading")
    open(outfile, "wb").write(req.content)
    return outfile


# TODO: Give option to load from blob as well
def load_hindcasts(
    start_date: datetime = datetime(2025, 1, 1).date(),
    end_date=datetime.now().date(),
    temp_dir: Optional[str] = None,
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
        raw_file = download_hindcasts(date, save_dir)
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
    return df_tracks


def _process_cxml_to_df(cxml_path: str, xsl_path: str = None):
    """Read a cxml v1.1 file; may not work on newer specs."""
    if xsl_path is None:
        xsl_path = "data/cxml_ecmwf_transformation.xsl"

    xsl = et.parse(str(xsl_path))
    xml = et.parse(str(cxml_path))
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
