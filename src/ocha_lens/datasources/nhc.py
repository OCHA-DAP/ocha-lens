"""
National Hurricane Center (NHC) Datasource Module.

This module provides functions to download, load, and process tropical
cyclone forecast and observation data from the National Hurricane Center
(NHC) and Central Pacific Hurricane Center (CPHC).

Important Notes
---------------
Basin Designation vs. Geographic Location:
    The 'basin' field represents a storm's designation basin (where it
    originated) from the ATCF ID, not its current geographic location.
    Storms retain their original basin designation throughout their lifecycle,
    even when crossing basin boundaries (e.g., Eastern Pacific storms crossing
    140°W into the Central Pacific still have basin='EP').
"""

import gzip
import io
import json
import logging
import os
import uuid
import zipfile
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
    check_quadrant_list,
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

# NHC storm name lookup table
NHC_STORM_TABLE_URL = "https://ftp.nhc.noaa.gov/atcf/archive/storm.table"

# ATCF Archive URL pattern
ATCF_ARCHIVE_URL = (
    "https://ftp.nhc.noaa.gov/atcf/archive/{year}/"
    "a{basin}{number:02d}{year}.dat.gz"
)

# NHC GIS Shapefile URL patterns
NHC_GIS_FORECAST_URL = (
    "https://www.nhc.noaa.gov/gis/forecast/archive/"
    "{basin}{number:02d}{year}_5day_latest.zip"
)
NHC_GIS_BESTTRACK_URL = (
    "https://www.nhc.noaa.gov/gis/best_track/"
    "{basin}{number:02d}{year}_best_track.zip"
)

# ATCF A-deck column names (first 35 standard columns)
# Format documentation: https://ftp.nhc.noaa.gov/atcf/README
# See also: https://github.com/palewire/atcf-data-parser
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

# Extend with additional columns for newer ATCF formats (following HDX pattern)
# These handle extended ATCF data that can have up to 75+ columns
for i in range(1, 21):
    ATCF_ADECK_COLUMNS.append(f"userdefine{i}")
    ATCF_ADECK_COLUMNS.append(f"userdata{i}")

# Official forecast technique identifiers
OFFICIAL_FORECAST_TECHS = [
    "OFCL",  # Official NHC forecast
    "OFCP",  # Official forecast (alternate)
    "HFIP",  # HFIP consensus (sometimes used as official)
]

# Wind radii column names
WIND_RADII_COLUMNS = [
    "quadrant_radius_34",
    "quadrant_radius_50",
    "quadrant_radius_64",
]

# Schema for storm metadata
# Note: genesis_basin represents the storm's designation basin (where it originated)
# and remains constant throughout the storm's lifecycle
STORM_SCHEMA = pa.DataFrameSchema(
    {
        "atcf_id": pa.Column(str, nullable=False),
        "name": pa.Column(
            str,
            nullable=True,
            checks=[
                pa.Check(
                    lambda s: s.upper() == s,
                    error="County must be uppercase",
                    element_wise=True,
                )
            ],
        ),
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
# Note: basin represents the storm's designation basin (from ATCF ID) and is
# constant for all track points. It does NOT represent the storm's current
# geographic location. Storms that cross basin boundaries (e.g., EP→CP at 140°W)
# retain their original basin designation throughout their lifecycle.
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
        "wind_speed": pa.Column(
            float, pa.Check.between(0, 300), nullable=True
        ),
        "pressure": pa.Column(
            float, pa.Check.between(800, 1100), nullable=True
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
        "nature": pa.Column(str, nullable=True),
        "quadrant_radius_34": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=True
        ),
        "quadrant_radius_50": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=True
        ),
        "quadrant_radius_64": pa.Column(
            "object", checks=pa.Check(check_quadrant_list), nullable=True
        ),
        "number": pa.Column(str, nullable=True),
        "storm_id": pa.Column(str, nullable=True),
        "point_id": pa.Column(str, nullable=False),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=False),
    },
    strict=True,
    coerce=True,
    # unique=[
    #     "atcf_id",
    #     "valid_time",
    #     "leadtime",
    #     "issued_time",
    # ],
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


# Derive base columns from TRACK_SCHEMA for raw data functions
# Raw data has name, latitude, longitude instead of storm_id, point_id, geometry
def _get_base_columns():
    """Get column list for raw data (before get_tracks transformation)."""
    cols = list(TRACK_SCHEMA.columns.keys())
    # Remove columns added by get_tracks
    for col in ["storm_id", "point_id", "geometry"]:
        if col in cols:
            cols.remove(col)
    # Add columns used by raw data but removed by get_tracks
    cols.extend(["name", "latitude", "longitude"])
    return cols


BASE_COLUMNS = _get_base_columns()


# Helper Functions


def _fetch_storm_names() -> dict:
    """
    Fetch storm name lookup table from NHC archive.

    Returns
    -------
    dict
        Mapping of ATCF ID (uppercase) to storm name
        Example: {'AL132023': 'LEE', 'EP092023': 'HILARY'}

    Notes
    -----
    Downloads and parses the official NHC storm.table file which contains
    the authoritative mapping of storm IDs to names. This is more reliable
    than extracting names from ATCF A-deck files where OFCL forecasts don't
    include storm names.
    """
    try:
        response = requests.get(NHC_STORM_TABLE_URL, timeout=30)
        response.raise_for_status()

        # Parse the table
        storm_names = {}
        for line in response.text.strip().split("\n"):
            if not line.strip():
                continue

            # Split by comma
            cols = [c.strip() for c in line.split(",")]

            if len(cols) < 21:
                continue

            name = cols[0]  # Column 1: Storm name
            atcf_id = cols[20]  # Column 21: ATCF ID

            if name and atcf_id:
                # Convert to uppercase to match output ATCF ID format
                storm_names[atcf_id.upper()] = name

        logger.info(f"Loaded {len(storm_names)} storm names from NHC archive")
        return storm_names

    except Exception as e:
        logger.warning(f"Failed to fetch storm name table: {e}")
        return {}


def _get_basin_code(atcf_id: str) -> str:
    """
    Extract and convert ATCF basin code to standard basin code.

    Parameters
    ----------
    atcf_id : str
        ATCF storm identifier (e.g., "EP092023", "AL142024")

    Returns
    -------
    str
        Standard basin code (NA, EP, or CP)

    Examples
    --------
    >>> _get_basin_code("EP092023")
    'EP'
    >>> _get_basin_code("AL142024")
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

    # Convert to pandas Timestamp (forecast_dt already has timezone from issuance_dt)
    return pd.Timestamp(forecast_dt)


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
    - Validates that ATCF ID in filename matches data

    Examples
    --------
    >>> df = _parse_atcf_adeck(Path("aal142023.dat.gz"))
    >>> df[['atcf_id', 'issued_time', 'valid_time', 'leadtime']]
    """
    # Extract expected ATCF ID from filename (remove leading 'a' and extension)
    # Filename format: aal012023.dat or aal012023.dat.gz
    # Uppercase basin letters to match IBTrACS format (AL012023 not al012023)
    filename = (
        file_path.stem
        if file_path.suffix != ".gz"
        else file_path.stem.replace(".dat", "")
    )
    atcf_id = (
        filename[1:3].upper() + filename[3:]
    )  # Uppercase basin code (first 2 chars)

    # Determine if file is gzipped
    if file_path.suffix == ".gz":
        open_func = gzip.open
        mode = "rt"  # Read as text
    else:
        open_func = open
        mode = "r"

    # Read CSV with pre-defined columns (no header in ATCF files)
    # Note: ATCF files can have variable number of columns per line (30-75+)
    # Using extended column list (75 columns) to handle all ATCF formats
    try:
        with open_func(file_path, mode) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=ATCF_ADECK_COLUMNS,
                skipinitialspace=True,
                na_values=["", " ", "null"],
                keep_default_na=True,
                engine="python",  # Python engine handles variable-length rows
            )
    except Exception as e:
        logger.error(f"Failed to read ATCF file {file_path}: {e}")
        return pd.DataFrame(columns=BASE_COLUMNS)

    logger.debug(f"Read {len(df)} records from {file_path}")

    # Filter for official forecasts only (OFCL, OFCP, etc.)
    # Excludes CARQ (best track observations) and model forecasts
    df = df[df["tech"].isin(OFFICIAL_FORECAST_TECHS)].copy()

    if df.empty:
        logger.warning(
            f"No official forecast data (OFCL) found in {file_path}. "
            f"Storm may not have had forecasts issued."
        )
        return pd.DataFrame(columns=BASE_COLUMNS)

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

    df["atcf_id"] = atcf_id

    # Validate ATCF ID from filename matches data
    year = df["yyyymmddhh"].astype(str).str[:4]
    expected_ids = (
        df["basin"].str.upper() + df["cy"].astype(str).str.zfill(2) + year
    )
    if expected_ids.unique()[0] != atcf_id:
        logger.error(
            f"ATCF ID mismatch in {file_path.name}: "
            f"filename={atcf_id}, data={expected_ids.unique()[0]}. "
            f"Skipping file."
        )
        return None
    if len(expected_ids.unique()) > 1:
        logger.warning(
            f"Multiple ATCF IDs found in {file_path.name}: {expected_ids.unique()}"
        )

    # Map basin to standard codes
    df["basin_std"] = df["basin"].str.lower().map(BASIN_CODE_MAPPING)

    # Get provider
    df["provider"] = df["basin_std"].apply(_get_provider)

    # Extract storm number
    df["number"] = df["cy"].astype(str).str.zfill(2)

    # Get storm name from official NHC storm.table
    # This is more reliable than ATCF stormname field which is blank for OFCL forecasts
    storm_name_lookup = _fetch_storm_names()

    # Map storm names using ATCF ID
    df["name"] = df["atcf_id"].map(storm_name_lookup)

    # Fallback to stormname field if lookup fails (for very recent storms)
    if df["name"].isna().any():
        df["name"] = df["name"].fillna(
            df["stormname"].replace(["", " ", "UNNAMED"], None)
        )
        # Forward-fill within storm if still missing
        df["name"] = df.groupby("atcf_id")["name"].ffill().bfill()

    # Wind speed and pressure (float columns - NaN is appropriate)
    df["wind_speed"] = pd.to_numeric(df["vmax"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["mslp"], errors="coerce")
    # ATCF uses 0 as missing value indicator - replace with NaN
    df.loc[df["pressure"] == 0, "pressure"] = pd.NA
    df.loc[df["wind_speed"] == 0, "wind_speed"] = pd.NA

    # Additional fields from ATCF to match IBTrACS schema
    # These are Int64 columns - use pd.NA for consistency
    # Convert to Int64 and replace 0 with pd.NA (ATCF uses 0 for missing values)
    for new_col, atcf_col in [
        ("max_wind_radius", "rmw"),
        ("last_closed_isobar_radius", "router"),
        ("last_closed_isobar_pressure", "pouter"),
        ("gust_speed", "gusts"),
    ]:
        df[new_col] = pd.to_numeric(df[atcf_col], errors="coerce").astype(
            "Int64"
        )
        df.loc[df[new_col] == 0, new_col] = pd.NA

    # Nature field (ty column) - storm type/classification (string column)
    df["nature"] = df["ty"].replace(["", " "], pd.NA)

    # Leadtime
    df["leadtime"] = df["tau"].astype("int64")

    # Parse wind radii from ATCF format
    # Wind radii are stored in separate rows with rad, rad1-rad4 columns
    # Group by forecast key (atcf_id + issued_time + tau) and collect radii
    df["rad"] = pd.to_numeric(df["rad"], errors="coerce")
    df["rad1"] = pd.to_numeric(df["rad1"], errors="coerce")
    df["rad2"] = pd.to_numeric(df["rad2"], errors="coerce")
    df["rad3"] = pd.to_numeric(df["rad3"], errors="coerce")
    df["rad4"] = pd.to_numeric(df["rad4"], errors="coerce")

    # Create forecast key for grouping
    df["forecast_key"] = (
        df["atcf_id"]
        + "_"
        + df["issued_time"].astype(str)
        + "_"
        + df["tau"].astype(str)
    )

    # Extract wind radii rows (where rad is 34, 50, or 64)
    radii_mask = df["rad"].isin([34, 50, 64])
    radii_df = df[radii_mask].copy()

    # Create dictionary to store radii by forecast_key
    radii_dict = {}
    for _, row in radii_df.iterrows():
        key = row["forecast_key"]
        rad_val = int(row["rad"])

        if key not in radii_dict:
            radii_dict[key] = {}

        # Create list [NE, SE, SW, NW]
        radii_list = [row["rad1"], row["rad2"], row["rad3"], row["rad4"]]

        # Convert to int list if all values are valid
        if all(pd.notna(r) and r > 0 for r in radii_list):
            radii_list = [int(r) for r in radii_list]
        else:
            radii_list = None

        # Store in dictionary by wind threshold
        radii_dict[key][f"quadrant_radius_{rad_val}"] = radii_list

    # Initialize wind radii columns with object dtype to hold lists
    for col in WIND_RADII_COLUMNS:
        df[col] = pd.Series([None] * len(df), dtype="object")

    # Merge radii back to main dataframe
    for key, radii in radii_dict.items():
        indices = df[df["forecast_key"] == key].index
        # Assign to each row individually to avoid pandas broadcasting issues
        for radii_col in WIND_RADII_COLUMNS:
            if radii_col in radii:
                for idx in indices:
                    df.at[idx, radii_col] = radii[radii_col]

    # Remove duplicate rows after merging wind radii
    # Keep only one row per unique forecast (atcf_id + issued_time + tau)
    df = df.drop_duplicates(subset=["forecast_key"], keep="first")

    # Select and rename columns to match schema
    # Use BASE_COLUMNS but substitute basin_std for basin during selection
    select_cols = [
        col if col != "basin" else "basin_std" for col in BASE_COLUMNS
    ]
    result = df[select_cols].rename(columns={"basin_std": "basin"})

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
        ATCF storm identifier (e.g., "EP092023")
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

    # Validate ATCF ID matches advisory text
    # Look for pattern like "MIAMI FL       AL132023" in header
    import re

    atcf_pattern = r"MIAMI FL\s+([A-Z]{2}\d{6})"
    for line in lines[:10]:  # Check first 10 lines only
        match = re.search(atcf_pattern, line)
        if match:
            advisory_atcf_id = match.group(1).upper()
            if advisory_atcf_id != storm_id.upper():
                logger.error(
                    f"ATCF ID mismatch: parameter={storm_id}, "
                    f"advisory text={advisory_atcf_id}. Skipping advisory."
                )
                return []
            break

    latitude = longitude = maxwind = valid_time = gust_speed = None
    radii_34 = radii_50 = radii_64 = None

    for ln in lines:
        # Save original line for pattern matching before preprocessing
        original_ln = ln

        # Preprocessing the line (following HDX pattern)
        ln = ln.replace("...", " ")
        ln = ln.replace("  ", " ")
        forecast_line = ln.split()

        # Parse FORECAST VALID or OUTLOOK VALID lines
        if (
            ln.startswith("FORECAST VALID") or ln.startswith("OUTLOOK VALID")
        ) and len(forecast_line) >= 5:
            # Save previous forecast point before starting new one
            if all([latitude, longitude, maxwind, valid_time]) and maxwind > 0:
                forecast_points.append(
                    {
                        "atcf_id": storm_id,
                        "name": storm_name.upper(),
                        "basin": basin,
                        "valid_time": valid_time,
                        "latitude": latitude,
                        "longitude": longitude,
                        "wind_speed": maxwind,
                        "pressure": pd.NA,
                        "max_wind_radius": pd.NA,  # Not in forecast advisory text
                        "last_closed_isobar_radius": pd.NA,
                        "last_closed_isobar_pressure": pd.NA,
                        "gust_speed": gust_speed
                        if gust_speed is not None
                        else pd.NA,
                        "nature": pd.NA,
                        "quadrant_radius_34": radii_34,
                        "quadrant_radius_50": radii_50,
                        "quadrant_radius_64": radii_64,
                    }
                )
            # Reset for next forecast point
            latitude = longitude = maxwind = valid_time = gust_speed = None
            radii_34 = radii_50 = radii_64 = None

            try:
                valid_time = _parse_valid_time(forecast_line[2], issuance)
                latitude = parse_lat_lon(forecast_line[3])
                longitude = parse_lat_lon(forecast_line[4])
            except Exception as e:
                logger.debug(
                    f"Could not parse position from: {ln[:80]} | Error: {e}"
                )
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

        # Parse MAX WIND lines and optional GUSTS
        # Example: "MAX WIND 115 KT GUSTS 140 KT"
        if ln.startswith("MAX WIND") and len(forecast_line) >= 3:
            try:
                maxwind = int(forecast_line[2])
                # Check for GUSTS on same line
                if "GUSTS" in forecast_line:
                    gusts_idx = forecast_line.index("GUSTS")
                    if gusts_idx + 1 < len(forecast_line):
                        gust_speed = int(forecast_line[gusts_idx + 1])
            except ValueError:
                logger.debug(f"Could not parse wind from: {ln[:50]}")
                maxwind = None

        # Parse wind radii lines: "64 KT... 25NE  20SE  15SW  15NW"
        # Check original line (before preprocessing removed "...")
        if "KT..." in original_ln and any(
            original_ln.startswith(f"{kt} KT") for kt in ["64", "50", "34"]
        ):
            try:
                # Extract wind threshold (e.g., "64" from ["64", "KT", "50NE", ...])
                kt_val = int(forecast_line[0])

                # After preprocessing, line is: ["64", "KT", "50NE", "40SE", "35SW", "50NW."]
                # Radii values start at index 2
                if len(forecast_line) >= 6:
                    # Extract numerical values from strings like "25NE"
                    ne = int("".join(filter(str.isdigit, forecast_line[2])))
                    se = int("".join(filter(str.isdigit, forecast_line[3])))
                    sw = int("".join(filter(str.isdigit, forecast_line[4])))
                    nw = int("".join(filter(str.isdigit, forecast_line[5])))

                    radii_list = [ne, se, sw, nw]

                    # Assign to appropriate threshold
                    if kt_val == 64:
                        radii_64 = radii_list
                    elif kt_val == 50:
                        radii_50 = radii_list
                    elif kt_val == 34:
                        radii_34 = radii_list
            except (ValueError, IndexError) as e:
                logger.debug(
                    f"Could not parse wind radii from: {ln[:50]} - {e}"
                )

    # Save the last forecast point after loop ends
    if all([latitude, longitude, maxwind, valid_time]) and maxwind > 0:
        forecast_points.append(
            {
                "atcf_id": storm_id,
                "name": storm_name.upper(),
                "basin": basin,
                "valid_time": valid_time,
                "latitude": latitude,
                "longitude": longitude,
                "wind_speed": maxwind,
                "pressure": pd.NA,
                "max_wind_radius": pd.NA,  # Not in forecast advisory text
                "last_closed_isobar_radius": pd.NA,
                "last_closed_isobar_pressure": pd.NA,
                "gust_speed": gust_speed if gust_speed is not None else pd.NA,
                "nature": pd.NA,
                "quadrant_radius_34": radii_34,
                "quadrant_radius_50": radii_50,
                "quadrant_radius_64": radii_64,
            }
        )

    logger.debug(
        f"Parsed {len(forecast_points)} forecast points for {storm_id}"
    )
    return forecast_points


def _fetch_current_storms_json() -> Optional[dict]:
    """
    Fetch current storms JSON from NHC CurrentStorms.json file.

    Returns
    -------
    dict or None
        Parsed JSON response, or None if fetch fails
    """
    try:
        response = requests.get(NHC_CURRENT_STORMS_URL, timeout=10)
        response.raise_for_status()
        print(response.json())
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
    # Uppercase basin code to match IBTrACS format (AL012023 not al012023)
    atcf_id_raw = storm_dict["id"]
    atcf_id = atcf_id_raw[:2].upper() + atcf_id_raw[2:]

    basin = _get_basin_code(atcf_id)
    provider = _get_provider(basin)

    # Parse last update time
    last_update = pd.Timestamp(storm_dict["lastUpdate"])

    return {
        "atcf_id": atcf_id,
        "name": storm_dict.get("name").upper(),
        "number": atcf_id[2:4],  # Extract "09" from "EP092023"
        "basin": basin,
        "provider": provider,
        "issued_time": last_update,
        "valid_time": last_update,
        "leadtime": 0,
        "wind_speed": float(storm_dict.get("intensity", 0)),
        "pressure": float(storm_dict.get("pressure", 0))
        if storm_dict.get("pressure")
        else pd.NA,
        "latitude": float(storm_dict.get("latitudeNumeric", 0)),
        "longitude": float(storm_dict.get("longitudeNumeric", 0)),
        "max_wind_radius": pd.NA,  # Not available in CurrentStorms.json
        "last_closed_isobar_radius": pd.NA,
        "last_closed_isobar_pressure": pd.NA,
        "gust_speed": pd.NA,
        "nature": pd.NA,
        "quadrant_radius_34": None,  # Not available in CurrentStorms.json
        "quadrant_radius_50": None,
        "quadrant_radius_64": None,
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
        return pd.DataFrame(columns=BASE_COLUMNS)

    logger.info(f"Processing {len(active_storms)} active storms")

    for storm in active_storms:
        # Uppercase basin code to match IBTrACS format (AL012023 not al012023)
        atcf_id_raw = storm["id"]
        atcf_id = atcf_id_raw[:2].upper() + atcf_id_raw[2:]

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
                    logger.info(
                        f"Fetching forecast advisory for {atcf_id} from {advisory_url}"
                    )

                    advisory_text = _fetch_forecast_advisory(advisory_url)

                    if advisory_text:
                        logger.info(
                            f"Successfully fetched advisory for {atcf_id}, parsing..."
                        )
                        forecast_points = _parse_forecast_advisory(
                            advisory_text, atcf_id, storm_name, issuance, basin
                        )
                        logger.info(
                            f"Parsed {len(forecast_points)} forecast points for {atcf_id}"
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

                        all_records.extend(forecast_points)
                        logger.debug(
                            f"Added {len(forecast_points)} forecast points for {atcf_id}"
                        )
                    else:
                        logger.warning(
                            f"Failed to fetch advisory text for {atcf_id} from {advisory_url}"
                        )
                else:
                    logger.debug(
                        f"Missing advisory URL or issuance for {atcf_id}"
                    )
            else:
                logger.debug(f"No forecastAdvisory found for {atcf_id}")

    if not all_records:
        logger.warning("No records extracted from active storms")
        return pd.DataFrame(columns=BASE_COLUMNS)

    logger.info(f"Extracted {len(all_records)} total records")
    return pd.DataFrame(all_records)


# Public API Functions


def _list_available_atcf_files(year: int, basin: str) -> List[int]:
    """
    List available ATCF A-deck files for a given year and basin.

    Parameters
    ----------
    year : int
        Year to check
    basin : str
        Basin code (AL, EP, or CP)

    Returns
    -------
    list of int
        Storm numbers that have available files

    Raises
    ------
    Exception
        If FTP connection or directory listing fails
    """
    import re
    from ftplib import FTP

    basin_lower = basin.lower()
    ftp_host = "ftp.nhc.noaa.gov"
    ftp_path = f"/atcf/archive/{year}/"

    # Connect to FTP server
    with FTP(ftp_host, timeout=30) as ftp:
        ftp.login()  # Anonymous login

        # List files in directory
        files = ftp.nlst(ftp_path)

        # Pattern: aal012023.dat.gz, aep092023.dat.gz, etc.
        pattern = rf"a{basin_lower}(\d{{2}}){year}\.dat\.gz"

        storm_numbers = []
        for file in files:
            # Extract just the filename from full path
            filename = file.split("/")[-1]
            match = re.match(pattern, filename)
            if match:
                storm_numbers.append(int(match.group(1)))

        storm_numbers = sorted(set(storm_numbers))

        logger.info(
            f"Found {len(storm_numbers)} {basin} storms in {year} archive: "
            f"{storm_numbers if len(storm_numbers) <= 10 else f'{storm_numbers[:10]}...'}"
        )

        return storm_numbers


def download_nhc_archive(
    year: int,
    basin: str = "AL",
    cache_dir: str = "storm",
    use_cache: bool = True,
) -> List[Path]:
    """
    Download ATCF archive files for all storms in a given year and basin.

    Parameters
    ----------
    year : int
        Year to download (e.g., 2023, 2024)
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
    Queries the FTP server to find all available storms for the specified
    year and basin, then downloads only those files.

    Files are saved with archive naming: a{basin}{number}{year}.dat
    (e.g., aal012023.dat) in the {cache_dir}/raw/atcf/ subdirectory.

    Files are automatically decompressed from .dat.gz to .dat format.

    Examples
    --------
    >>> # Download all Atlantic storms from 2023
    >>> paths = download_nhc_archive(2023, basin="AL")
    >>> # Example cached file: aal012023.dat

    >>> # Download all Eastern Pacific storms from 2023
    >>> paths = download_nhc_archive(2023, basin="EP")
    >>> # Example cached file: aep012023.dat
    """
    # Create cache directory
    cache_path = Path(cache_dir) / "raw" / "atcf"
    os.makedirs(cache_path, exist_ok=True)

    # Query FTP server to find available storms
    storm_numbers = _list_available_atcf_files(year, basin)

    if not storm_numbers:
        logger.warning(f"No storms found for {basin} {year}")
        return []

    downloaded_files = []
    basin_lower = basin.lower()

    for num in storm_numbers:
        # Construct filename to match archive naming: aal012023.dat
        filename = f"a{basin_lower}{num:02d}{year}.dat"
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
    use_cache: bool = False,
    year: Optional[int] = None,
    basin: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load and process NHC storm data from CurrentStorms.json or historical archive.

    Supports two modes:
    1. **Current mode** (default): Downloads current storms from NHC CurrentStorms.json
    2. **Archive mode**: Downloads historical ATCF data when year is specified

    Parameters
    ----------
    file_path : str, optional
        Path to NHC JSON file. If None, downloads data
    cache_dir : str, default "storm"
        Directory for caching downloaded files
    use_cache : bool, default False
        Whether to use existing cached file if available
    year : int, optional
        Year for archive mode (e.g., 2023). If specified, loads
        historical ATCF data instead of current json data
    basin : str, optional
        Basin code for archive mode: "AL" (Atlantic), "EP" (Eastern
        Pacific), or "CP" (Central Pacific). If None, loads all basins

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

    >>> # Archive mode: All storms from all basins in 2023
    >>> df = load_nhc(year=2023)

    >>> # Archive mode: All Atlantic storms from 2023
    >>> df = load_nhc(year=2023, basin="AL")

    >>> # Eastern Pacific storms
    >>> df = load_nhc(year=2024, basin="EP")
    """
    # Archive mode: year is specified
    if year is not None:
        # If basin is None, load all basins
        if basin is None:
            logger.info(f"Loading archive data for all basins in {year}")
            all_dfs = []
            for basin_code in ["AL", "EP", "CP"]:
                df = _load_nhc_archive(
                    year=year,
                    basin=basin_code,
                    cache_dir=cache_dir,
                    use_cache=use_cache,
                )
                if df is not None and not df.empty:
                    all_dfs.append(df)

            if not all_dfs:
                logger.warning(f"No data loaded for any basin in {year}")
                return None

            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(
                f"Loaded {len(combined_df)} records from "
                f"{len(all_dfs)} basins in {year}"
            )
            return combined_df
        else:
            logger.info(f"Loading archive data for {basin} {year}")
            return _load_nhc_archive(
                year=year,
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


def download_nhc_gis(
    atcf_id: str,
    cache_dir: str = "storm",
    product: str = "forecast",
    use_cache: bool = False,
) -> Optional[Path]:
    """
    Download NHC GIS shapefiles for a storm.

    Downloads ZIP file containing shapefiles for forecast cone, wind radii,
    and other GIS products from NHC.

    Parameters
    ----------
    atcf_id : str
        ATCF storm identifier (e.g., "AL102023", "EP092024")
    cache_dir : str, default "storm"
        Directory to store shapefile ZIP files
    product : str, default "forecast"
        GIS product type: "forecast" or "besttrack"
    use_cache : bool, default False
        Whether to use existing cached file if available

    Returns
    -------
    Path or None
        Path to downloaded ZIP file, None if download failed

    Notes
    -----
    Files are saved as: {cache_dir}/raw/gis/{atcf_id}_{product}.zip

    The ZIP file contains multiple shapefiles:
    - Forecast cone (polygon)
    - Forecast track (line)
    - Wind radii (polygons for 34kt, 50kt, 64kt)
    - Watches/warnings (polygons)

    Examples
    --------
    >>> path = download_nhc_gis("AL102023")
    >>> path = download_nhc_gis("EP092024", product="besttrack")
    """
    # Parse ATCF ID to extract components
    basin = atcf_id[:2].lower()  # "al", "ep", "cp"
    number = int(atcf_id[2:4])  # storm number
    year = int(atcf_id[4:])  # 4-digit year

    # Create cache directory
    cache_path = Path(cache_dir) / "raw" / "gis"
    os.makedirs(cache_path, exist_ok=True)

    # Generate filename
    filename = f"{atcf_id}_{product}.zip"
    file_path = cache_path / filename

    # Check if we should use cache
    if use_cache and file_path.exists():
        logger.info(f"Using cached GIS file: {file_path}")
        return file_path

    # Determine URL based on product type
    if product == "forecast":
        url = NHC_GIS_FORECAST_URL.format(
            basin=basin, number=number, year=year
        )
    elif product == "besttrack":
        url = NHC_GIS_BESTTRACK_URL.format(
            basin=basin, number=number, year=year
        )
    else:
        logger.error(f"Unknown product type: {product}")
        return None

    # Download file
    logger.info(f"Downloading GIS {product} shapefile from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save ZIP file
        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Saved GIS shapefile to {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download GIS shapefile: {e}")
        return None


def download_nhc_gis_from_json(
    storm_dict: dict,
    product: str = "cone",
    file_format: str = "shapefile",
    cache_dir: str = "storm",
    use_cache: bool = False,
) -> Optional[Path]:
    """
    Download GIS products using URLs from CurrentStorms.json.

    This is the preferred method for active storms as it uses NHC's provided
    URLs rather than constructing them manually. Supports multiple GIS products
    including forecast cones, wind radii, track lines, and storm surge.

    Parameters
    ----------
    storm_dict : dict
        Storm dictionary from CurrentStorms.json activeStorms array
    product : str, default "cone"
        GIS product to download:
        - "cone" - Forecast cone of uncertainty (polygon)
        - "track" - Forecast track line
        - "wind_radii" - Forecast wind radii (34/50/64kt polygons)
        - "initial_wind" - Current wind extent (polygons)
        - "wind_prob" - Wind speed probabilities (grids)
        - "surge_watch_warning" - Storm surge watches/warnings
        - "surge_flooding" - Potential storm surge flooding
        - "wind_watch_warning" - Wind watches/warnings
    file_format : str, default "shapefile"
        File format: "shapefile" (zip) or "kmz"
    cache_dir : str, default "storm"
        Directory to store downloaded files
    use_cache : bool, default False
        Whether to use existing cached file if available

    Returns
    -------
    Path or None
        Path to downloaded file, None if download failed or URL not available

    Notes
    -----
    - Files are saved as: {cache_dir}/raw/gis/{atcf_id}_{product}_{advnum}.{ext}
    - Advisory number is included to track forecast evolution
    - Not all products are available for all storms (e.g., surge only for US threats)
    - URLs are advisory-specific, matching the tabular forecast data

    Examples
    --------
    >>> data = nhc._fetch_current_storms_json()
    >>> storm = data["activeStorms"][0]
    >>> cone_path = nhc.download_nhc_gis_from_json(storm, product="cone")
    >>> radii_path = nhc.download_nhc_gis_from_json(storm, product="wind_radii")
    """
    atcf_id = storm_dict.get("id")
    if not atcf_id:
        logger.error("Storm dictionary missing 'id' field")
        return None

    # Map product names to JSON fields and file keys
    product_mapping = {
        "cone": ("trackCone", "kmzFile", "zipFile"),
        "track": ("forecastTrack", "kmzFile", "zipFile"),
        "wind_radii": ("forecastWindRadiiGIS", "kmzFile", "zipFile"),
        "initial_wind": ("initialWindExtent", "kmzFile", "zipFile"),
        "wind_prob": ("windSpeedProbabilitiesGIS", "kmzFile", "zipFile0p5deg"),
        "surge_watch_warning": (
            "stormSurgeWatchWarningGIS",
            "kmzFile",
            "zipFile",
        ),
        "surge_flooding": (
            "potentialStormSurgeFloodingGIS",
            "kmzFile",
            "zipFile",
        ),
        "wind_watch_warning": ("windWatchesWarnings", "kmzFile", "zipFile"),
    }

    if product not in product_mapping:
        logger.error(
            f"Unknown product: {product}. Choose from: {list(product_mapping.keys())}"
        )
        return None

    # Get the JSON field and file key for this product
    json_field, kmz_key, zip_key = product_mapping[product]
    file_key = kmz_key if file_format == "kmz" else zip_key

    # Extract URL from storm dictionary
    product_dict = storm_dict.get(json_field)
    if not product_dict:
        logger.warning(
            f"Product '{product}' not available for storm {atcf_id}"
        )
        return None

    url = product_dict.get(file_key)
    if not url:
        logger.warning(
            f"No {file_format} URL found for product '{product}' in storm {atcf_id}"
        )
        return None

    # Get advisory number for filename
    adv_num = product_dict.get("advNum", "unknown")

    # Create cache directory
    cache_path = Path(cache_dir) / "raw" / "gis"
    os.makedirs(cache_path, exist_ok=True)

    # Generate filename
    ext = "kmz" if file_format == "kmz" else "zip"
    filename = f"{atcf_id}_{product}_adv{adv_num}.{ext}"
    file_path = cache_path / filename

    # Check if we should use cache
    if use_cache and file_path.exists():
        logger.info(f"Using cached GIS file: {file_path}")
        return file_path

    # Download file
    logger.info(f"Downloading {product} {file_format} from {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save file
        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Saved GIS {product} to {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download GIS {product}: {e}")
        return None


def get_current_storm_gis(
    atcf_id: Optional[str] = None,
    products: Optional[List[str]] = None,
    cache_dir: str = "storm",
    use_cache: bool = False,
) -> dict:
    """
    Download all GIS products for current active storms.

    Fetches CurrentStorms.json and downloads requested GIS products
    for all active storms or a specific storm.

    Parameters
    ----------
    atcf_id : str, optional
        Specific storm to download (e.g., "AL102023"). If None, downloads
        for all active storms.
    products : list of str, optional
        GIS products to download. If None, downloads default set:
        ["cone", "track", "wind_radii", "initial_wind"]
    cache_dir : str, default "storm"
        Directory to store downloaded files
    use_cache : bool, default False
        Whether to use existing cached files

    Returns
    -------
    dict
        Dictionary mapping storm IDs to product paths:
        {
            "AL102023": {
                "cone": Path(...),
                "track": Path(...),
                "wind_radii": Path(...)
            }
        }

    Examples
    --------
    >>> # Get all GIS for all active storms
    >>> gis_data = nhc.get_current_storm_gis()

    >>> # Get specific products for one storm
    >>> gis_data = nhc.get_current_storm_gis(
    ...     atcf_id="AL102023",
    ...     products=["cone", "wind_radii"]
    ... )

    >>> # Load the cone
    >>> cone = nhc.load_nhc_gis(gis_data["AL102023"]["cone"])
    """
    # Default products
    if products is None:
        products = ["cone", "track", "wind_radii", "initial_wind"]

    # Fetch current storms
    data = _fetch_current_storms_json()
    if not data:
        logger.error("Failed to fetch current storms")
        return {}

    active_storms = data.get("activeStorms", [])
    if not active_storms:
        logger.info("No active storms")
        return {}

    # Filter to specific storm if requested
    if atcf_id:
        active_storms = [s for s in active_storms if s.get("id") == atcf_id]
        if not active_storms:
            logger.warning(f"Storm {atcf_id} not found in active storms")
            return {}

    # Download GIS for each storm
    results = {}
    for storm in active_storms:
        storm_id = storm.get("id")
        if not storm_id:
            continue

        logger.info(f"Downloading GIS products for {storm_id}")
        results[storm_id] = {}

        for product in products:
            path = download_nhc_gis_from_json(
                storm,
                product=product,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            if path:
                results[storm_id][product] = path

    return results


def load_nhc_gis(
    file_path: Union[str, Path],
    layer_name: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load NHC GIS shapefile from ZIP archive.

    Extracts and loads shapefiles from NHC GIS ZIP file. The ZIP typically
    contains multiple layers (cone, track, wind radii, watches/warnings).

    Parameters
    ----------
    file_path : str or Path
        Path to NHC GIS ZIP file
    layer_name : str, optional
        Specific layer to load. If None, loads first available layer.
        Common layer names:
        - "al{XX}{YYYY}_5day_pgn" - Forecast cone polygon
        - "al{XX}{YYYY}_5day_lin" - Forecast track line
        - "al{XX}{YYYY}_5day_pts" - Forecast points
        - "al{XX}{YYYY}_ww_wwlin" - Watches/warnings

    Returns
    -------
    GeoDataFrame
        Loaded shapefile data with geometry and attributes

    Examples
    --------
    >>> gdf = load_nhc_gis("storm/raw/gis/AL102023_forecast.zip")
    >>> cone = load_nhc_gis("AL102023_forecast.zip", layer_name="al102023_5day_pgn")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"GIS file not found: {file_path}")

    try:
        # Extract ZIP to temporary directory
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # List available layers
            shp_files = [f for f in zip_ref.namelist() if f.endswith(".shp")]

            if not shp_files:
                logger.error(f"No shapefiles found in {file_path}")
                return gpd.GeoDataFrame()

            # If layer_name specified, find matching shapefile
            if layer_name:
                target_shp = None
                for shp in shp_files:
                    if layer_name in shp:
                        target_shp = shp
                        break

                if not target_shp:
                    logger.error(
                        f"Layer {layer_name} not found in {file_path}. "
                        f"Available layers: {shp_files}"
                    )
                    return gpd.GeoDataFrame()
            else:
                # Use first shapefile
                target_shp = shp_files[0]
                logger.info(
                    f"Loading first available layer: {target_shp}. "
                    f"All layers: {shp_files}"
                )

            # Read shapefile directly from ZIP
            gdf = gpd.read_file(f"zip://{file_path}!{target_shp}")

            logger.info(f"Loaded {len(gdf)} features from {target_shp}")
            return gdf

    except Exception as e:
        logger.error(f"Failed to load GIS shapefile: {e}")
        return gpd.GeoDataFrame()


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
    >>> tracks[tracks.leadtime == 0]  # Current positions (observations)
    >>> tracks[tracks.leadtime > 0]  # Forecast tracks
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
