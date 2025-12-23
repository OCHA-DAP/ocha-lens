"""
National Hurricane Center (NHC) Datasource Module.

This module provides functions to download, load, and process tropical
cyclone forecast and observation data from the National Hurricane Center
(NHC) and Central Pacific Hurricane Center (CPHC).
"""

import gzip
import io
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from lat_lon_parser import parse as parse_lat_lon

from ocha_lens.utils.storm import (
    _create_storm_id,
    _to_gdf,
    check_coordinate_bounds,
    check_crs,
)

# Set up logging
logger = logging.getLogger(__name__)

# Basin code mapping from ATCF to standard codes
BASIN_CODE_MAPPING = {
    "al": "NA",  # Atlantic → North Atlantic (matches ECMWF/IBTrACS)
    "ep": "EP",  # Eastern Pacific
    "cp": "CP",  # Central Pacific (CPHC jurisdiction)
}

# NHC API URL
NHC_CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"

# ATCF Archive URL pattern
ATCF_ARCHIVE_URL = (
    "https://ftp.nhc.noaa.gov/atcf/archive/{year}/"
    "a{basin}{number:02d}{year}.dat.gz"
)

# ATCF A-deck column names (based on ATCF documentation)
ATCF_ADECK_COLUMNS = [
    "basin",  # 0: Basin identifier (AL, EP, CP, etc.)
    "cy",  # 1: Annual cyclone number (01-99)
    "yyyymmddhh",  # 2: Warning Date-Time-Group
    "technum",  # 3: Objective technique sorting number
    "tech",  # 4: Technique (model/method identifier)
    "tau",  # 5: Forecast period (hours, 0-168)
    "lat",  # 6: Latitude (tenths of degrees, N/S)
    "lon",  # 7: Longitude (tenths of degrees, E/W)
    "vmax",  # 8: Maximum sustained wind speed (knots)
    "mslp",  # 9: Minimum sea level pressure (mb)
    "ty",  # 10: Level of tropical cyclone development
    "rad",  # 11: Wind intensity for radii (34, 50, 64 kt)
    "windcode",  # 12: Radius code (AAA, NEQ, etc.)
    "rad1",  # 13: Radius in NE quadrant (nm)
    "rad2",  # 14: Radius in SE quadrant (nm)
    "rad3",  # 15: Radius in SW quadrant (nm)
    "rad4",  # 16: Radius in NW quadrant (nm)
    "pouter",  # 17: Pressure of outermost closed isobar
    "router",  # 18: Radius of outermost closed isobar
    "rmw",  # 19: Radius of maximum winds
    "gusts",  # 20: Gust speed (knots)
    "eye",  # 21: Eye diameter (nm)
    "subregion",  # 22: Sub-region code
    "maxseas",  # 23: Maximum seas (ft)
    "initials",  # 24: Forecaster's initials
    "dir",  # 25: Storm direction (degrees)
    "speed",  # 26: Storm speed (knots)
    "stormname",  # 27: Storm name
    "depth",  # 28: Depth (D=deep, M=medium, S=shallow)
    "seas",  # 29: Seas (Wave height, ft)
    "seascode",  # 30: Seas radius code
    "seas1",  # 31: Seas radius in NE quadrant
    "seas2",  # 32: Seas radius in SE quadrant
    "seas3",  # 33: Seas radius in SW quadrant
    "seas4",  # 34: Seas radius in NW quadrant
]

# Official forecast technique identifiers
OFFICIAL_FORECAST_TECHS = [
    "OFCL",  # Official NHC forecast
    "OFCP",  # Official forecast (alternate)
    "HFIP",  # HFIP consensus (sometimes used as official)
]

# Schema for storm metadata
STORM_SCHEMA = pa.DataFrameSchema(
    {
        "atcf_id": pa.Column(str, nullable=False),
        "name": pa.Column(str, nullable=True),
        "number": pa.Column(str, nullable=False),
        "season": pa.Column(
            "int64", pa.Check.between(2000, 2050), nullable=False
        ),
        "genesis_basin": pa.Column(
            str,
            pa.Check.isin(list(BASIN_CODE_MAPPING.values())),
            nullable=False,
        ),
        "provider": pa.Column(str, nullable=True),
        "storm_id": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
    unique=["atcf_id", "storm_id"],
    report_duplicates="all",
)

# Schema for track data
TRACK_SCHEMA = pa.DataFrameSchema(
    {
        "atcf_id": pa.Column(str, nullable=False),
        "provider": pa.Column(str, nullable=False),
        "basin": pa.Column(
            str,
            pa.Check.isin(list(BASIN_CODE_MAPPING.values())),
            nullable=False,
        ),
        "issued_time": pa.Column(pd.Timestamp, nullable=False),
        "valid_time": pa.Column(pd.Timestamp, nullable=False),
        "leadtime": pa.Column("Int64", pa.Check.ge(0), nullable=False),
        "forecast_type": pa.Column(
            str,
            pa.Check.isin(["observation", "forecast", "outlook"]),
            nullable=False,
        ),
        "wind_speed": pa.Column(
            float, pa.Check.between(0, 300), nullable=True
        ),
        "pressure": pa.Column(
            float, pa.Check.between(800, 1100), nullable=True
        ),
        "number": pa.Column(str, nullable=True),
        "storm_id": pa.Column(str, nullable=True),
        "point_id": pa.Column(str, nullable=False),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=False),
    },
    strict=True,
    coerce=True,
    # unique=["atcf_id", "valid_time", "leadtime", "issued_time", "forecast_type"],
    # report_duplicates="all",
    checks=[
        pa.Check(
            lambda gdf: check_crs(gdf, "EPSG:4326"),
            error="CRS must be EPSG:4326",
        ),
        pa.Check(
            lambda gdf: check_coordinate_bounds(gdf),
            error="All coordinates must be within valid bounds",
        ),
    ],
)


# Helper Functions


def _get_basin_code(atcf_id: str) -> str:
    """
    Extract and convert ATCF basin code to standard basin code.

    Parameters
    ----------
    atcf_id : str
        ATCF storm identifier (e.g., "ep092023", "al142024")

    Returns
    -------
    str
        Standard basin code (NA, EP, or CP)

    Examples
    --------
    >>> _get_basin_code("ep092023")
    'EP'
    >>> _get_basin_code("al142024")
    'NA'
    """
    basin_prefix = atcf_id[:2].lower()
    return BASIN_CODE_MAPPING.get(basin_prefix, basin_prefix.upper())


def _parse_valid_time(valid_time_str: str, issuance: str) -> pd.Timestamp:
    """
    Convert relative forecast time to absolute timestamp.

    NHC advisories use relative time format like "29/1200Z" which needs
    to be converted to an absolute timestamp based on the issuance time.

    Parameters
    ----------
    valid_time_str : str
        Forecast time in format "DD/HHMMZ" (e.g., "29/1200Z")
    issuance : str
        Advisory issuance timestamp (ISO format or parseable string)

    Returns
    -------
    pd.Timestamp
        UTC timestamp for the valid time

    Examples
    --------
    >>> _parse_valid_time("29/1200Z", "2023-08-28T18:00:00Z")
    Timestamp('2023-08-29 12:00:00+0000', tz='UTC')
    """
    # Parse issuance time
    issuance_dt = dateparser.parse(issuance)

    # Clean up the valid time string
    time_part = valid_time_str.replace("Z", "").strip()

    # Parse day and time components
    if "/" in time_part:
        day_str, hhmm = time_part.split("/")
    else:
        # Sometimes just "HHMM" without day
        day_str = str(issuance_dt.day)
        hhmm = time_part

    day = int(day_str)
    hour = int(hhmm[:2]) if len(hhmm) >= 2 else 0
    minute = int(hhmm[2:4]) if len(hhmm) >= 4 else 0

    # Create forecast datetime with the specified day/time
    forecast_dt = issuance_dt.replace(
        day=day, hour=hour, minute=minute, second=0, microsecond=0
    )

    # Handle month rollover (if forecast day < issuance day, add 1 month)
    if day < issuance_dt.day:
        forecast_dt = forecast_dt + relativedelta(months=1)

    return pd.Timestamp(forecast_dt, tz="UTC")


def _get_provider(basin: str) -> str:
    """
    Determine the provider (NHC or CPHC) based on basin.

    Parameters
    ----------
    basin : str
        Basin code (NA, EP, or CP)

    Returns
    -------
    str
        Provider name ("NHC" or "CPHC")
    """
    return "CPHC" if basin == "CP" else "NHC"


def _parse_atcf_lat_lon(lat_str: str, lon_str: str) -> Tuple[float, float]:
    """
    Parse ATCF latitude and longitude format.

    ATCF coordinates are in tenths of degrees with N/S/E/W suffix.
    Examples: "253N" = 25.3°N, "755W" = 75.5°W

    Parameters
    ----------
    lat_str : str
        Latitude string (e.g., "253N", "185S")
    lon_str : str
        Longitude string (e.g., "755W", "1234E")

    Returns
    -------
    tuple of (float, float)
        Latitude and longitude in decimal degrees

    Examples
    --------
    >>> _parse_atcf_lat_lon("253N", "755W")
    (25.3, -75.5)
    >>> _parse_atcf_lat_lon("185S", "1234E")
    (-18.5, 123.4)
    """
    # Parse latitude
    lat_value = float(lat_str[:-1]) / 10.0
    lat_dir = lat_str[-1]
    if lat_dir == "S":
        lat_value = -lat_value

    # Parse longitude
    lon_value = float(lon_str[:-1]) / 10.0
    lon_dir = lon_str[-1]
    if lon_dir == "W":
        lon_value = -lon_value

    return lat_value, lon_value


def _parse_atcf_adeck(file_path: Path) -> pd.DataFrame:
    """
    Parse ATCF A-deck file to standardized DataFrame format.

    ATCF A-deck files contain forecast data in comma-delimited format.
    This function reads the file, filters for official forecasts (OFCL),
    and converts to the standard NHC DataFrame schema.

    Parameters
    ----------
    file_path : Path
        Path to ATCF A-deck file (.dat or .dat.gz)

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with NHC schema fields

    Notes
    -----
    - Filters for TECH in OFFICIAL_FORECAST_TECHS (primarily OFCL)
    - TAU=0 represents analysis (current position)
    - TAU>0 represents forecast positions
    - Handles both compressed (.gz) and uncompressed files
    - Missing values in optional fields are handled gracefully

    Examples
    --------
    >>> df = _parse_atcf_adeck(Path("aal142023.dat.gz"))
    >>> df[['atcf_id', 'issued_time', 'valid_time', 'leadtime']]
    """
    # Determine if file is gzipped
    if file_path.suffix == ".gz":
        open_func = gzip.open
        mode = "rt"  # Read as text
    else:
        open_func = open
        mode = "r"

    # Read CSV with specified columns (no header in ATCF files)
    # Note: ATCF files can have more than 35 columns (extended fields)
    # We only read the first 35 standard columns
    try:
        with open_func(file_path, mode) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=ATCF_ADECK_COLUMNS,
                usecols=range(len(ATCF_ADECK_COLUMNS)),
                skipinitialspace=True,
                na_values=["", " ", "null"],
                keep_default_na=True,
            )
    except Exception as e:
        logger.error(f"Failed to read ATCF file {file_path}: {e}")
        return pd.DataFrame(
            columns=[
                "atcf_id",
                "name",
                "number",
                "basin",
                "provider",
                "issued_time",
                "valid_time",
                "leadtime",
                "forecast_type",
                "wind_speed",
                "pressure",
                "latitude",
                "longitude",
            ]
        )

    logger.debug(f"Read {len(df)} records from {file_path}")

    # Filter for official forecasts only
    df = df[df["tech"].isin(OFFICIAL_FORECAST_TECHS)].copy()

    if df.empty:
        logger.warning(f"No official forecasts found in {file_path}")
        return pd.DataFrame(
            columns=[
                "atcf_id",
                "name",
                "number",
                "basin",
                "provider",
                "issued_time",
                "valid_time",
                "leadtime",
                "forecast_type",
                "wind_speed",
                "pressure",
                "latitude",
                "longitude",
            ]
        )

    logger.debug(f"Filtered to {len(df)} official forecast records")

    # Parse issued time (YYYYMMDDHH)
    df["issued_time"] = pd.to_datetime(
        df["yyyymmddhh"], format="%Y%m%d%H", utc=True
    )

    # Calculate valid time (issued_time + tau hours)
    df["valid_time"] = df.apply(
        lambda row: row["issued_time"] + pd.Timedelta(hours=row["tau"]), axis=1
    )

    # Parse coordinates
    coords = df.apply(
        lambda row: _parse_atcf_lat_lon(row["lat"], row["lon"])
        if pd.notna(row["lat"]) and pd.notna(row["lon"])
        else (None, None),
        axis=1,
    )
    df["latitude"] = coords.apply(lambda x: x[0])
    df["longitude"] = coords.apply(lambda x: x[1])

    # Create ATCF ID (basin + number + year)
    year = df["yyyymmddhh"].astype(str).str[:4]
    df["atcf_id"] = (
        df["basin"].str.lower() + df["cy"].astype(str).str.zfill(2) + year
    )

    # Map basin to standard codes
    df["basin_std"] = df["basin"].str.lower().map(BASIN_CODE_MAPPING)

    # Get provider
    df["provider"] = df["basin_std"].apply(_get_provider)

    # Extract storm number
    df["number"] = df["cy"].astype(str).str.zfill(2)

    # Get storm name (if available)
    df["name"] = df["stormname"].replace(["", " ", "UNNAMED"], None)

    # Determine forecast type
    df["forecast_type"] = df["tau"].apply(
        lambda tau: "observation"
        if tau == 0
        else ("outlook" if tau > 120 else "forecast")
    )

    # Wind speed and pressure
    df["wind_speed"] = pd.to_numeric(df["vmax"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["mslp"], errors="coerce")
    df.loc[df["pressure"] == 0, "pressure"] = None
    df.loc[df["wind_speed"] == 0, "wind_speed"] = None

    # ATCF uses 0 as missing value indicator - convert to None
    df.loc[df["pressure"] == 0, "pressure"] = None
    df.loc[df["wind_speed"] == 0, "wind_speed"] = None

    # Leadtime
    df["leadtime"] = df["tau"].astype("int64")

    # Select and rename columns to match schema
    result = df[
        [
            "atcf_id",
            "name",
            "number",
            "basin_std",
            "provider",
            "issued_time",
            "valid_time",
            "leadtime",
            "forecast_type",
            "wind_speed",
            "pressure",
            "latitude",
            "longitude",
        ]
    ].rename(columns={"basin_std": "basin"})

    # Drop rows with missing coordinates
    result = result.dropna(subset=["latitude", "longitude"])

    logger.info(
        f"Parsed {len(result)} valid forecast points from {file_path.name}"
    )

    return result


def _fetch_forecast_advisory(url: str) -> Optional[str]:
    """
    Fetch forecast advisory HTML from NHC and extract text content.

    Parameters
    ----------
    url : str
        URL to the forecast advisory HTML page

    Returns
    -------
    str or None
        Extracted text content from <pre> tag, or None if fetch fails

    Notes
    -----
    Uses a 10-second connection timeout and 30-second read timeout.
    Returns None on any HTTP or parsing errors.
    """
    try:
        response = requests.get(url, timeout=(10, 30))
        response.raise_for_status()

        # Parse HTML and extract <pre> tag content
        soup = BeautifulSoup(response.content, "html.parser")
        pre_tag = soup.find("pre")

        if pre_tag is None:
            logger.warning(f"No <pre> tag found in advisory at {url}")
            return None

        return pre_tag.get_text()

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching advisory from {url}")
        return None
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch advisory from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing advisory from {url}: {e}")
        return None


def _parse_forecast_advisory(
    advisory_text: str,
    storm_id: str,
    storm_name: str,
    issuance: str,
    basin: str,
) -> List[dict]:
    """
    Parse NHC forecast advisory text to extract forecast points.

    Parses lines starting with "FORECAST VALID" or "OUTLOOK VALID" to
    extract position and wind speed forecasts. Also handles "REMNANTS OF
    CENTER LOCATED NEAR" lines for dissipating storms.

    Parameters
    ----------
    advisory_text : str
        Raw text from <pre> tag in advisory HTML
    storm_id : str
        ATCF storm identifier (e.g., "ep092023")
    storm_name : str
        Storm name
    issuance : str
        Issuance timestamp of advisory
    basin : str
        Basin code (NA, EP, or CP)

    Returns
    -------
    list of dict
        Forecast points with valid_time, latitude, longitude, wind_speed

    Notes
    -----
    Only includes forecast points where all three values (lat, lon, wind)
    are successfully parsed and wind speed > 0.
    """
    forecast_points = []
    lines = advisory_text.split("\n")

    # Preprocessing: remove "..." and extra spaces
    lines = [ln.replace("...", " ").replace("  ", " ") for ln in lines]

    latitude = longitude = maxwind = valid_time = None

    for ln in lines:
        forecast_line = ln.split(" ")

        # Parse FORECAST VALID or OUTLOOK VALID lines
        if (
            ln.startswith("FORECAST VALID") or ln.startswith("OUTLOOK VALID")
        ) and len(forecast_line) >= 5:
            try:
                valid_time = _parse_valid_time(forecast_line[2], issuance)
                latitude = parse_lat_lon(forecast_line[3])
                longitude = parse_lat_lon(forecast_line[4])
            except Exception:
                logger.debug(f"Could not parse position from: {ln[:50]}")
                latitude = longitude = valid_time = None
                continue

        # Parse REMNANTS OF CENTER LOCATED NEAR lines
        elif (
            ln.startswith("REMNANTS OF CENTER LOCATED NEAR")
            and len(forecast_line) >= 8
        ):
            try:
                valid_time = _parse_valid_time(forecast_line[8], issuance)
                latitude = parse_lat_lon(forecast_line[5])
                longitude = parse_lat_lon(forecast_line[6])
            except Exception:
                logger.debug(f"Could not parse remnants from: {ln[:50]}")
                latitude = longitude = valid_time = None
                continue

        # Parse MAX WIND lines
        if ln.startswith("MAX WIND") and len(forecast_line) >= 3:
            try:
                maxwind = int(forecast_line[2])
            except ValueError:
                logger.debug(f"Could not parse wind from: {ln[:50]}")
                maxwind = None

        # If we have all components, save the point
        if (
            latitude is not None
            and longitude is not None
            and maxwind is not None
            and valid_time is not None
        ):
            if maxwind > 0:  # Only include valid wind speeds
                forecast_points.append(
                    {
                        "atcf_id": storm_id,
                        "name": storm_name,
                        "basin": basin,
                        "valid_time": valid_time,
                        "latitude": latitude,
                        "longitude": longitude,
                        "wind_speed": maxwind,
                        "pressure": None,  # Forecasts don't have pressure
                    }
                )

            # Reset for next forecast point
            latitude = longitude = maxwind = valid_time = None

    logger.debug(
        f"Parsed {len(forecast_points)} forecast points for {storm_id}"
    )
    return forecast_points


def _fetch_current_storms_json() -> Optional[dict]:
    """
    Fetch current storms JSON from NHC API.

    Returns
    -------
    dict or None
        Parsed JSON response, or None if fetch fails

    Notes
    -----
    Fetches from https://www.nhc.noaa.gov/CurrentStorms.json
    Uses a 10-second timeout.
    """
    try:
        response = requests.get(NHC_CURRENT_STORMS_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {NHC_CURRENT_STORMS_URL}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch current storms: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return None


def _extract_current_observation(storm_dict: dict) -> dict:
    """
    Extract current observation from storm dictionary.

    Parameters
    ----------
    storm_dict : dict
        Storm dictionary from NHC CurrentStorms.json

    Returns
    -------
    dict
        Observation record with standardized fields

    Notes
    -----
    The current observation includes both wind and pressure data,
    unlike forecasts which only have wind.
    """
    atcf_id = storm_dict["id"]
    basin = _get_basin_code(atcf_id)
    provider = _get_provider(basin)

    # Parse last update time
    last_update = pd.Timestamp(storm_dict["lastUpdate"], tz="UTC")

    return {
        "atcf_id": atcf_id,
        "name": storm_dict.get("name"),
        "number": atcf_id[2:4],  # Extract "09" from "ep09"
        "basin": basin,
        "provider": provider,
        "issued_time": last_update,
        "valid_time": last_update,
        "leadtime": 0,
        "forecast_type": "observation",
        "wind_speed": float(storm_dict.get("intensity", 0)),
        "pressure": float(storm_dict.get("pressure", 0))
        if storm_dict.get("pressure")
        else None,
        "latitude": float(storm_dict.get("latitudeNumeric", 0)),
        "longitude": float(storm_dict.get("longitudeNumeric", 0)),
    }


def _process_nhc_to_df(
    raw_data: dict,
    include_observations: bool = True,
    include_forecasts: bool = True,
) -> pd.DataFrame:
    """
    Process raw NHC JSON data to DataFrame.

    Extracts current observations and fetches/parses forecast advisories
    for each active storm.

    Parameters
    ----------
    raw_data : dict
        Parsed JSON from NHC CurrentStorms.json
    include_observations : bool, default True
        Whether to include current observations
    include_forecasts : bool, default True
        Whether to fetch and include forecast advisories

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with observations and/or forecasts

    Notes
    -----
    If no active storms or all fetches fail, returns an empty DataFrame
    with the correct column structure.
    """
    all_records = []

    # Check if there are active storms
    active_storms = raw_data.get("activeStorms", [])

    if not active_storms:
        logger.info("No active storms currently")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(
            columns=[
                "atcf_id",
                "name",
                "number",
                "basin",
                "provider",
                "issued_time",
                "valid_time",
                "leadtime",
                "forecast_type",
                "wind_speed",
                "pressure",
                "latitude",
                "longitude",
            ]
        )

    logger.info(f"Processing {len(active_storms)} active storms")

    for storm in active_storms:
        atcf_id = storm["id"]
        storm_name = storm.get("name", "Unnamed")
        basin = _get_basin_code(atcf_id)

        logger.debug(f"Processing storm {atcf_id} ({storm_name})")

        # Extract current observation
        if include_observations:
            try:
                obs = _extract_current_observation(storm)
                all_records.append(obs)
                logger.debug(f"Extracted current observation for {atcf_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to extract observation for {atcf_id}: {e}"
                )

        # Fetch and parse forecast advisory
        if include_forecasts:
            forecast_advisory = storm.get("forecastAdvisory")
            if forecast_advisory:
                advisory_url = forecast_advisory.get("url")
                issuance = forecast_advisory.get("issuance")

                if advisory_url and issuance:
                    logger.debug(
                        f"Fetching forecast advisory from {advisory_url}"
                    )

                    advisory_text = _fetch_forecast_advisory(advisory_url)

                    if advisory_text:
                        forecast_points = _parse_forecast_advisory(
                            advisory_text, atcf_id, storm_name, issuance, basin
                        )

                        # Add metadata and leadtime to forecast points
                        provider = _get_provider(basin)
                        issued_time = pd.Timestamp(issuance, tz="UTC")

                        for point in forecast_points:
                            point["provider"] = provider
                            point["number"] = atcf_id[2:4]
                            point["issued_time"] = issued_time

                            # Calculate leadtime in hours
                            leadtime = int(
                                (
                                    point["valid_time"] - issued_time
                                ).total_seconds()
                                / 3600
                            )
                            point["leadtime"] = leadtime

                            # Determine forecast type
                            if leadtime > 120:  # > 5 days
                                point["forecast_type"] = "outlook"
                            else:
                                point["forecast_type"] = "forecast"

                        all_records.extend(forecast_points)
                        logger.debug(
                            f"Added {len(forecast_points)} forecast points"
                        )
                    else:
                        logger.warning(
                            f"Failed to fetch advisory for {atcf_id}"
                        )
                else:
                    logger.debug(f"No advisory URL or issuance for {atcf_id}")

    if not all_records:
        logger.warning("No records extracted from active storms")
        return pd.DataFrame(
            columns=[
                "atcf_id",
                "name",
                "number",
                "basin",
                "provider",
                "issued_time",
                "valid_time",
                "leadtime",
                "forecast_type",
                "wind_speed",
                "pressure",
                "latitude",
                "longitude",
            ]
        )

    logger.info(f"Extracted {len(all_records)} total records")
    return pd.DataFrame(all_records)


# Public API Functions


def download_nhc_archive(
    year: int,
    storm_number: Optional[Union[int, List[int]]] = None,
    basin: str = "AL",
    cache_dir: str = "storm",
    use_cache: bool = True,
) -> List[Path]:
    """
    Download ATCF archive files for historical storm data.

    Downloads A-deck (forecast) files from the ATCF archive at
    ftp.nhc.noaa.gov for a specific year and basin. Can download
    all storms for a year or specific storm numbers.

    Parameters
    ----------
    year : int
        Year to download (e.g., 2023, 2024)
    storm_number : int or list of int, optional
        Specific storm number(s) to download (1-30).
        If None, downloads all storms for the year
    basin : str, default "AL"
        Basin code: "AL" (Atlantic), "EP" (Eastern Pacific),
        or "CP" (Central Pacific)
    cache_dir : str, default "storm"
        Directory to store downloaded files
    use_cache : bool, default True
        Whether to use existing cached files if available

    Returns
    -------
    list of Path
        Paths to downloaded ATCF files

    Notes
    -----
    Files are saved as: atcf_{basin}_{number}_{year}.dat in
    the {cache_dir}/raw/atcf/ subdirectory.

    Files are automatically decompressed from .dat.gz to .dat format.

    Examples
    --------
    >>> # Download all Atlantic storms from 2023
    >>> paths = download_nhc_archive(2023, basin="AL")
    >>> # Download specific storms
    >>> paths = download_nhc_archive(2023, storm_number=[14, 15], basin="AL")
    >>> # Download Eastern Pacific storm 9 from 2023
    >>> paths = download_nhc_archive(2023, storm_number=9, basin="EP")
    """
    # Create cache directory
    cache_path = Path(cache_dir) / "raw" / "atcf"
    os.makedirs(cache_path, exist_ok=True)

    # Determine which storms to download
    if storm_number is None:
        # Download all storms (typically 1-30)
        storm_numbers = range(1, 31)
    elif isinstance(storm_number, int):
        storm_numbers = [storm_number]
    else:
        storm_numbers = storm_number

    downloaded_files = []
    basin_lower = basin.lower()

    for num in storm_numbers:
        # Construct filename and URL
        filename = f"atcf_{basin_lower}_{num:02d}_{year}.dat"
        file_path = cache_path / filename

        # Check cache
        if use_cache and file_path.exists():
            logger.debug(f"Using cached file: {file_path}")
            downloaded_files.append(file_path)
            continue

        # Construct download URL
        url = ATCF_ARCHIVE_URL.format(year=year, basin=basin_lower, number=num)

        logger.debug(f"Downloading from {url}")

        try:
            # Download file
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Decompress gzip content and save
            with gzip.open(io.BytesIO(response.content), "rt") as gz_file:
                content = gz_file.read()

            with open(file_path, "w") as f:
                f.write(content)

            logger.info(f"Downloaded and decompressed {filename}")
            downloaded_files.append(file_path)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(
                    f"Storm {basin_lower}{num:02d}{year} not found (404)"
                )
            else:
                logger.warning(f"HTTP error downloading {url}: {e}")
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")

    if not downloaded_files:
        logger.warning(f"No files downloaded for {basin} basin, year {year}")
    else:
        logger.info(
            f"Downloaded {len(downloaded_files)} ATCF files for {basin} {year}"
        )

    return downloaded_files


def _load_nhc_archive(
    year: int,
    storm_number: Optional[Union[int, List[int]]] = None,
    basin: str = "AL",
    cache_dir: str = "storm",
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Load NHC archive data from ATCF files.

    Downloads (if needed) and parses ATCF A-deck files for historical
    storm forecast data.

    Parameters
    ----------
    year : int
        Year to load
    storm_number : int or list of int, optional
        Specific storm number(s) to load. If None, loads all storms
    basin : str, default "AL"
        Basin code (AL, EP, or CP)
    cache_dir : str, default "storm"
        Directory for cached files
    use_cache : bool, default True
        Whether to use existing cached files

    Returns
    -------
    pd.DataFrame or None
        Combined DataFrame with all storm data, or None if no data

    Notes
    -----
    This function handles multiple storms by concatenating their data
    into a single DataFrame.
    """
    # Download ATCF files
    logger.info(f"Loading archive data for {basin} basin, year {year}")

    file_paths = download_nhc_archive(
        year=year,
        storm_number=storm_number,
        basin=basin,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    if not file_paths:
        logger.warning(f"No ATCF files available for {basin} {year}")
        return None

    # Parse each file and combine
    all_dfs = []
    for file_path in file_paths:
        df = _parse_atcf_adeck(file_path)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning(f"No data parsed from {basin} {year} files")
        return None

    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    logger.info(
        f"Loaded {len(combined_df)} records from "
        f"{len(all_dfs)} storms in {basin} {year}"
    )

    return combined_df


def download_nhc(
    cache_dir: str = "storm",
    use_cache: bool = False,
) -> Optional[Path]:
    """
    Download current NHC storm data in JSON format.

    Fetches active storm data from the National Hurricane Center's
    CurrentStorms.json API and saves to local cache directory.

    Parameters
    ----------
    cache_dir : str, default "storm"
        Directory to store raw JSON files
    use_cache : bool, default False
        Whether to use existing cached file if available from today

    Returns
    -------
    Path or None
        Path to downloaded JSON file, None if download failed

    Notes
    -----
    Files are saved with timestamp: nhc_{YYYYMMDD}_{HHMM}.json
    in the {cache_dir}/raw/ subdirectory.

    Examples
    --------
    >>> path = download_nhc()
    >>> path = download_nhc(cache_dir="data/storms", use_cache=True)
    """
    # Create cache directory
    cache_path = Path(cache_dir) / "raw"
    os.makedirs(cache_path, exist_ok=True)

    # Generate filename with current timestamp
    now = datetime.utcnow()
    filename = f"nhc_{now.strftime('%Y%m%d_%H%M')}.json"
    file_path = cache_path / filename

    # Check if we should use cache
    if use_cache:
        # Look for any file from today
        today_pattern = f"nhc_{now.strftime('%Y%m%d')}_*.json"
        existing_files = list(cache_path.glob(today_pattern))

        if existing_files:
            # Use the most recent file from today
            latest_file = max(existing_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using cached file: {latest_file}")
            return latest_file

    # Fetch data from NHC
    logger.info("Downloading current storms from NHC")
    data = _fetch_current_storms_json()

    if data is None:
        logger.error("Failed to download NHC data")
        return None

    # Save to file
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved NHC data to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save NHC data: {e}")
        return None


def load_nhc(
    file_path: Optional[str] = None,
    cache_dir: str = "storm",
    use_cache: bool = True,
    year: Optional[int] = None,
    storm_number: Optional[Union[int, List[int]]] = None,
    basin: str = "AL",
) -> Optional[pd.DataFrame]:
    """
    Load and process NHC storm data from current API or historical archive.

    Supports two modes:
    1. **Current mode** (default): Downloads current storms from NHC API
    2. **Archive mode**: Downloads historical ATCF data when year is specified

    Parameters
    ----------
    file_path : str, optional
        Path to NHC JSON file. If None, downloads data
    cache_dir : str, default "storm"
        Directory for caching downloaded files
    use_cache : bool, default True
        Whether to use existing cached file if available
    year : int, optional
        Year for archive mode (e.g., 2023). If specified, loads
        historical ATCF data instead of current API data
    storm_number : int or list of int, optional
        Specific storm number(s) for archive mode (1-30).
        If None in archive mode, loads all storms for the year
    basin : str, default "AL"
        Basin code for archive mode: "AL" (Atlantic), "EP" (Eastern
        Pacific), or "CP" (Central Pacific)

    Returns
    -------
    pd.DataFrame or None
        DataFrame with combined observations and forecasts, or None if
        loading fails

    Notes
    -----
    **Current Mode:**
    - Returns current observations (leadtime=0) and forecast points
    - Pressure is only available for observations, not forecasts
    - Fetches from https://www.nhc.noaa.gov/CurrentStorms.json

    **Archive Mode:**
    - Returns historical forecast data from ATCF archive
    - Coverage: 1850-2024
    - Includes official NHC forecasts (OFCL) only
    - Pressure available for both observations and forecasts

    Examples
    --------
    >>> # Current storms
    >>> df = load_nhc()
    >>> df = load_nhc(file_path="storm/raw/nhc_20231225_1200.json")

    >>> # Archive mode: All Atlantic storms from 2023
    >>> df = load_nhc(year=2023, basin="AL")

    >>> # Specific storm from archive
    >>> df = load_nhc(year=2023, storm_number=14, basin="AL")

    >>> # Multiple storms
    >>> df = load_nhc(year=2023, storm_number=[14, 15, 16], basin="AL")

    >>> # Eastern Pacific storms
    >>> df = load_nhc(year=2024, basin="EP")
    """
    # Archive mode: year is specified
    if year is not None:
        logger.info(f"Loading archive data for {basin} {year}")
        return _load_nhc_archive(
            year=year,
            storm_number=storm_number,
            basin=basin,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

    # Current mode: download from API or load from file
    if file_path is None:
        logger.info("No file path provided, downloading current storms")
        file_path = download_nhc(cache_dir=cache_dir, use_cache=use_cache)

        if file_path is None:
            logger.error("Failed to download NHC data")
            return None
    else:
        file_path = Path(file_path)

    # Load JSON data
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded NHC data from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return None

    # Process to DataFrame
    df = _process_nhc_to_df(data)

    if df.empty:
        logger.warning("No data extracted from NHC JSON")
        return df

    logger.info(f"Loaded {len(df)} records from NHC")
    return df


def get_storms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract storm metadata from NHC DataFrame.

    Creates a dataset with one row per storm containing identifying
    information. This provides a summary of all storms in the dataset
    with their basic metadata.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from load_nhc()

    Returns
    -------
    pd.DataFrame
        Storm metadata with schema validation applied

    Notes
    -----
    The storm_id field is created from name, basin, and season using
    the format "{name}_{basin}_{season}" (lowercase). Unnamed storms
    will have storm_id set to None.

    Examples
    --------
    >>> df = load_nhc()
    >>> storms = get_storms(df)
    >>> storms[['atcf_id', 'name', 'genesis_basin', 'season']]
    """
    if df.empty:
        logger.warning("Input DataFrame is empty")
        return pd.DataFrame(columns=list(STORM_SCHEMA.columns.keys()))

    df_ = df.copy()

    # Calculate season from valid_time
    df_["season"] = df_["valid_time"].dt.year

    # Group by atcf_id to get one row per storm
    df_storms = (
        df_.groupby("atcf_id")
        .agg(
            {
                "name": "first",
                "number": "first",
                "basin": "first",
                "season": "first",
                "provider": "first",
            }
        )
        .reset_index()
    )

    # Rename basin to genesis_basin
    df_storms = df_storms.rename(columns={"basin": "genesis_basin"})

    # Create storm_id for named storms
    df_storms["storm_id"] = df_storms.apply(_create_storm_id, axis=1)

    logger.info(f"Extracted {len(df_storms)} unique storms")

    # Validate against schema
    return STORM_SCHEMA.validate(df_storms)


def get_tracks(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Extract track-level data from NHC DataFrame.

    Creates a GeoDataFrame with one row per track point (observation or
    forecast), including geometry for spatial analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from load_nhc()

    Returns
    -------
    gpd.GeoDataFrame
        Track data with Point geometry and schema validation applied

    Notes
    -----
    - Each point is assigned a unique UUID in the point_id field
    - Longitude is normalized to [-180, 180] range
    - Geometry is created in EPSG:4326 (WGS84)
    - Pressure field will be None for forecast points
    - The storm_id field links to the storms table

    Examples
    --------
    >>> df = load_nhc()
    >>> tracks = get_tracks(df)
    >>> tracks.plot()  # Spatial plot of storm tracks
    >>> tracks[tracks.forecast_type == 'observation']  # Current positions
    >>> tracks[tracks.forecast_type == 'forecast']  # Forecast tracks
    """
    if df.empty:
        logger.warning("Input DataFrame is empty")
        # Return empty GeoDataFrame with correct schema
        empty_df = pd.DataFrame(columns=list(TRACK_SCHEMA.columns.keys()))
        return gpd.GeoDataFrame(empty_df, geometry="geometry", crs="EPSG:4326")

    df_ = df.copy()

    # Get storms to merge in storm_id (only select needed columns)
    df_storms = get_storms(df)
    df_tracks = df_.merge(
        df_storms[["atcf_id", "storm_id"]], on="atcf_id", how="left"
    )

    # Drop any extra columns not in schema (like 'name' from merge)
    schema_columns = list(TRACK_SCHEMA.columns.keys())
    extra_columns = [
        col
        for col in df_tracks.columns
        if col not in schema_columns
        and col != "latitude"
        and col != "longitude"
    ]
    if extra_columns:
        df_tracks = df_tracks.drop(columns=extra_columns)

    # Generate point_id for each record
    df_tracks["point_id"] = [str(uuid.uuid4()) for _ in range(len(df_tracks))]

    # Normalize longitude to [-180, 180]
    df_tracks["longitude"] = df_tracks["longitude"].apply(
        lambda lon: lon - 360 if lon > 180 else lon
    )

    # Convert to GeoDataFrame using utility function
    gdf_tracks = _to_gdf(df_tracks)

    logger.info(f"Created GeoDataFrame with {len(gdf_tracks)} track points")

    # Validate against schema
    return TRACK_SCHEMA.validate(gdf_tracks)
