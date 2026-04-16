import io
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
import requests
from bs4 import BeautifulSoup

from ocha_lens.datasources.nhc import _fetch_current_storms_json

logger = logging.getLogger(__name__)


# GIS archive base URL (5km product, 120-hr cumulative)
# Available 2017-04-19 — present (only issued when storms are active)
GIS_ARCHIVE_INDEX_URL = "https://www.nhc.noaa.gov/gis/forecast/archive/"
GIS_ARCHIVE_ZIP_URL = (
    "https://www.nhc.noaa.gov/gis/forecast/archive/{issuance}_wsp_120hr5km.zip"
)

# Official NHC probability bin labels — matches the shapefile PERCENTAGE column
WSP_PERCENTAGE_BINS = [
    "<5%",
    "5-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    ">90%",
]

WSP_POLYGON_SCHEMA = pa.DataFrameSchema(
    {
        "issuance": pa.Column(pd.Timestamp, nullable=False),
        "wind_threshold_kt": pa.Column(
            int,
            pa.Check.isin([34, 50, 64]),
            nullable=False,
        ),
        "percentage": pa.Column(
            str,
            pa.Check.isin(WSP_PERCENTAGE_BINS),
            nullable=False,
        ),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=True),
    },
    strict=True,
    coerce=True,
)


def load_nhc_wsp(
    issuance: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: str = "storm",
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load NHC 5km wind speed probability polygons.

    Three modes:

    1. **Current** (default, no arguments): fetches the latest issuance from
       ``CurrentStorms.json``. Only available when storms are active.
    2. **Single issuance**: pass ``issuance`` as YYYYMMDDHH or ISO timestamp.
    3. **Archive range**: pass ``start`` (and optionally ``end``) to fetch all
       available issuances in the date range from the NHC GIS archive.

    Parameters
    ----------
    issuance : str, optional
        Single issuance timestamp (YYYYMMDDHH or ISO format, e.g. '2023082200').
    start : str, optional
        Start of date range for archive mode (YYYYMMDDHH or ISO format).
    end : str, optional
        End of date range. Defaults to now. Only used with ``start``.
    cache_dir : str, default "storm"
        Directory for cached zip files (used in archive mode).
    use_cache : bool, default True
        Whether to use cached zip files if available.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: issuance, wind_threshold_kt, percentage, geometry (EPSG:4326).
        One row per (issuance, wind threshold, probability band). Empty if no
        data is available.
    """
    if start is not None:
        return _load_nhc_wsp_archive(
            start=pd.Timestamp(start, tz="UTC"),
            end=(
                pd.Timestamp(end, tz="UTC")
                if end
                else pd.Timestamp.now(tz="UTC")
            ),
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

    if issuance is not None:
        if len(issuance) == 10 and issuance.isdigit():
            ts = pd.Timestamp(
                datetime.strptime(issuance, "%Y%m%d%H").replace(
                    tzinfo=timezone.utc
                )
            )
        else:
            ts = pd.Timestamp(issuance, tz="UTC")
        ts_str = ts.strftime("%Y%m%d%H")
        url = GIS_ARCHIVE_ZIP_URL.format(issuance=ts_str)
        logger.info(f"Fetching WSP shapefile for issuance {ts_str}")
        gdf = _load_wsp_from_url(url, ts)
        if gdf.empty:
            return gdf
        return WSP_POLYGON_SCHEMA.validate(gdf)

    return _load_nhc_wsp_current()


def _load_nhc_wsp_current() -> gpd.GeoDataFrame:
    """Fetch the latest WSP issuance from the active storm JSON."""
    data = _fetch_current_storms_json()
    if data is None:
        logger.info("No active storms — no WSP product available")
        return _empty_wsp_gdf()

    active_storms = data.get("activeStorms", [])
    # All active storms share the same basin-wide WSP product
    gis_meta = next(
        (
            s["windSpeedProbabilitiesGIS"]
            for s in active_storms
            if s.get("windSpeedProbabilitiesGIS")
            and s["windSpeedProbabilitiesGIS"].get("zipFile5km")
        ),
        None,
    )
    if gis_meta is None:
        logger.info("No active storms — no WSP product available")
        return _empty_wsp_gdf()

    issuance = pd.Timestamp(gis_meta["issuance"], tz="UTC")
    zip_url = gis_meta["zipFile5km"]
    logger.info(f"Fetching current WSP from {zip_url} (issuance {issuance})")
    gdf = _load_wsp_from_url(zip_url, issuance)
    if gdf.empty:
        return gdf
    return WSP_POLYGON_SCHEMA.validate(gdf)


def _load_nhc_wsp_archive(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str,
    use_cache: bool,
) -> gpd.GeoDataFrame:
    """Fetch all available WSP shapefiles in the archive between start and end."""
    logger.info(
        f"Scanning archive for WSP issuances between {start} and {end}"
    )
    issuances = _list_archive_issuances(start, end)

    if not issuances:
        logger.warning(f"No WSP archive files found between {start} and {end}")
        return _empty_wsp_gdf()

    logger.info(f"Found {len(issuances)} archive issuances to fetch")

    cache_path = Path(cache_dir) / "raw" / "nhc_wsp"
    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)

    all_gdfs = []
    for ts in issuances:
        ts_str = ts.strftime("%Y%m%d%H")
        cache_file = cache_path / f"{ts_str}_wsp_120hr5km.zip"

        if use_cache and cache_file.exists():
            logger.debug(f"Using cached file {cache_file.name}")
            zip_bytes = cache_file.read_bytes()
        else:
            url = GIS_ARCHIVE_ZIP_URL.format(issuance=ts_str)
            try:
                zip_bytes = _fetch_wsp_zip(url)
            except Exception as e:
                logger.warning(f"Failed to fetch {ts_str}: {e}")
                continue
            if use_cache:
                cache_file.write_bytes(zip_bytes)

        gdf = _parse_wsp_zip(zip_bytes, ts)
        if not gdf.empty:
            all_gdfs.append(gdf)

    if not all_gdfs:
        return _empty_wsp_gdf()

    combined = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    return WSP_POLYGON_SCHEMA.validate(combined)


def _load_wsp_from_url(url: str, issuance: pd.Timestamp) -> gpd.GeoDataFrame:
    """Fetch a single WSP zip by URL and return parsed polygons."""
    zip_bytes = _fetch_wsp_zip(url)
    return _parse_wsp_zip(zip_bytes, issuance)


def _fetch_wsp_zip(url: str) -> bytes:
    """Download a NHC WSP GIS zip and return raw bytes."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _parse_wsp_zip(
    zip_bytes: bytes, issuance: pd.Timestamp
) -> gpd.GeoDataFrame:
    """
    Parse a NHC 5km WSP zip into a flat GeoDataFrame.

    Each zip contains 3 shapefiles (34, 50, 64 kt), each with up to 11 rows
    (one per probability band). Returns all three thresholds combined into a
    single GeoDataFrame.

    Parameters
    ----------
    zip_bytes : bytes
        Raw bytes of a NHC 5km WSP zip file.
    issuance : pd.Timestamp
        Issuance timestamp to attach to every row.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: issuance, wind_threshold_kt, percentage, geometry (EPSG:4326).
    """
    gdfs = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)

            for kt in [34, 50, 64]:
                candidates = [
                    f
                    for f in z.namelist()
                    if f.endswith(".shp") and f"wsp{kt}knt" in f
                ]
                if not candidates:
                    logger.warning(f"No {kt}kt shapefile found in zip")
                    continue

                gdf = gpd.read_file(os.path.join(tmp, candidates[0])).to_crs(
                    "EPSG:4326"
                )
                gdf = gdf[["PERCENTAGE", "geometry"]].rename(
                    columns={"PERCENTAGE": "percentage"}
                )
                gdf["issuance"] = issuance
                gdf["wind_threshold_kt"] = kt
                gdfs.append(
                    gdf[
                        [
                            "issuance",
                            "wind_threshold_kt",
                            "percentage",
                            "geometry",
                        ]
                    ]
                )

    if not gdfs:
        return _empty_wsp_gdf()

    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )


def _list_archive_issuances(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[pd.Timestamp]:
    """
    Scrape the NHC GIS archive index for available 5km WSP issuances.

    Parameters
    ----------
    start : pd.Timestamp
        Start of the date range (inclusive).
    end : pd.Timestamp
        End of the date range (inclusive).

    Returns
    -------
    list of pd.Timestamp
        Available issuance timestamps in [start, end], sorted chronologically.
    """
    response = requests.get(GIS_ARCHIVE_INDEX_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    issuances = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "_wsp_120hr5km.zip" not in href:
            continue
        filename = href.split("/")[-1]
        ts_str = filename.split("_wsp_")[0]
        try:
            ts = pd.Timestamp(
                datetime.strptime(ts_str, "%Y%m%d%H").replace(
                    tzinfo=timezone.utc
                )
            )
        except ValueError:
            continue
        if start <= ts <= end:
            issuances.append(ts)

    return sorted(issuances)


def _empty_wsp_gdf() -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame with the WSP polygon schema columns."""
    empty_df = pd.DataFrame(columns=list(WSP_POLYGON_SCHEMA.columns.keys()))
    return gpd.GeoDataFrame(empty_df, geometry="geometry", crs="EPSG:4326")
