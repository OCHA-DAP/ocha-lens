"""GDACS Tropical Cyclone API client.

Fetches active and historical tropical cyclone events, advisory
timelines, and country-level impact data from the GDACS REST API.
No authentication required.

API docs: https://www.gdacs.org/gdacsapi/swagger/index.html

Endpoints
---------
- Event search (``/events/geteventlist/SEARCH``): paginated, filterable
  by date range and alert level. One row per TC event.
- Event/episode detail (``/events/geteventdata`` and
  ``/events/getepisodedata``): full JSON for one event or episode,
  including resource URLs (timeline, per-buffer impact).
- Timeline: per-advisory snapshots with position, wind speed, and
  population exposure (``pop39``/``pop74``). Available 2015+.
- Impact buffers: country-level cumulative population exposure within
  the cumulative wind footprint. Returns data ~2022+.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

import geopandas as gpd
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.gdacs.org/gdacsapi/api"

_TIMEOUT = 30
_PAGE_SIZE = 100

# Alert levels are queried one at a time and results deduped on
# eventid. The combined-alertlevel API filter is unreliable, and a
# storm's alert level can change during its lifetime — without
# per-level iteration + dedup, the same event would appear multiple
# times or be missed entirely.
_ALERT_LEVELS = ("Green", "Orange", "Red")

_TIMELINE_NUMERIC_COLS = [
    "latitude",
    "longitude",
    "wind_speed",
    "wind_gusts",
    "pressure",
    "pop",
    "pop39",
    "pop74",
    "popstormsurge",
    "alertscore",
    "windrad_nm_34kt_ne",
    "windrad_nm_34kt_se",
    "windrad_nm_34kt_sw",
    "windrad_nm_34kt_nw",
    "windrad_nm_50kt_ne",
    "windrad_nm_50kt_se",
    "windrad_nm_50kt_sw",
    "windrad_nm_50kt_nw",
    "windrad_nm_64kt_ne",
    "windrad_nm_64kt_se",
    "windrad_nm_64kt_sw",
    "windrad_nm_64kt_nw",
]

_EVENT_COLS = [
    "eventid",
    "name",
    "alert_level",
    "episode_alert",
    "iso3",
    "country",
    "from_date",
    "to_date",
    "is_current",
    "severity_kmh",
    "severity_text",
    "source",
    "geometry",
]


def get_events(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    alert_levels: Optional[List[Literal["Green", "Orange", "Red"]]] = None,
    source: Optional[Literal["NOAA", "JTWC"]] = None,
    page_size: int = _PAGE_SIZE,
) -> gpd.GeoDataFrame:
    """Fetch GDACS tropical cyclone events.

    Auto-paginates the GDACS event search endpoint, iterating over
    alert levels separately and deduplicating on ``eventid``.

    Parameters
    ----------
    from_date, to_date : str, optional
        ISO date strings ``"YYYY-MM-DD"``. Default: GDACS API defaults.
    alert_levels : list of {"Green", "Orange", "Red"}, optional
        Alert levels to query. Default: all three.
    source : {"NOAA", "JTWC"}, optional
        Filter to events tagged with this source. Applied client-side
        because the API source filter is unreliable. Default: no
        filter.
    page_size : int, default 100
        GDACS caps page size at 100; do not exceed.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per event. Empty GeoDataFrame with the expected
        columns if no events match.
    """
    levels = list(alert_levels) if alert_levels else list(_ALERT_LEVELS)
    seen: Dict[int, Dict[str, Any]] = {}

    for level in levels:
        page = 1
        while True:
            params: Dict[str, Any] = {
                "eventlist": "TC",
                "alertlevel": level,
                "pageSize": page_size,
                "pageNumber": page,
            }
            if from_date:
                params["fromDate"] = from_date
            if to_date:
                params["toDate"] = to_date

            r = requests.get(
                f"{BASE_URL}/events/geteventlist/SEARCH",
                params=params,
                timeout=_TIMEOUT,
            )
            r.raise_for_status()
            if not r.text.strip():
                break
            features = r.json().get("features", [])
            if not features:
                break

            for f in features:
                row = _event_feature_to_row(f)
                if source and row["source"] != source:
                    continue
                if row["eventid"] in seen:
                    continue
                seen[row["eventid"]] = row

            page += 1

    if not seen:
        return gpd.GeoDataFrame(
            columns=_EVENT_COLS, geometry="geometry", crs="EPSG:4326"
        )

    df = pd.DataFrame(list(seen.values()))
    return gpd.GeoDataFrame(
        df.drop(columns=["lon", "lat"]),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


def get_event_detail(eventid: int) -> Dict[str, Any]:
    """Fetch full event detail JSON.

    Includes the episode list and resource URLs for the timeline and
    per-buffer impact endpoints.
    """
    r = requests.get(
        f"{BASE_URL}/events/geteventdata",
        params={"eventtype": "TC", "eventid": eventid},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_episode_detail(eventid: int, episodeid: int) -> Dict[str, Any]:
    """Fetch full episode detail JSON.

    Each episode is one model run (issuance), typically every 6 hours.
    The structure mirrors :func:`get_event_detail`, but resource URLs
    point to episode-specific data.
    """
    r = requests.get(
        f"{BASE_URL}/events/getepisodedata",
        params={
            "eventtype": "TC",
            "eventid": eventid,
            "episodeid": episodeid,
        },
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_timeline(eventid: int) -> pd.DataFrame:
    """Fetch the advisory timeline for one TC event.

    Each row is one advisory with position, wind speed, population
    exposure (``pop39``/``pop74``), and quadrant wind radii.
    ``pop39``/``pop74`` are instantaneous snapshots — not cumulative.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``advisory_number`` ascending. Empty DataFrame if
        no timeline exists for the event.
    """
    detail = get_event_detail(eventid)
    impacts = detail.get("properties", {}).get("impacts", [])

    timeline_url = None
    for imp in impacts:
        url = imp.get("resource", {}).get("timeline")
        if url:
            timeline_url = url
            break

    if timeline_url is None:
        logger.warning("No timeline found for event %s", eventid)
        return pd.DataFrame()

    r = requests.get(timeline_url, timeout=_TIMEOUT)
    r.raise_for_status()
    items = r.json().get("channel", {}).get("item", [])

    keep_cols = [
        "advisory_number",
        "name",
        "advisory_datetime",
        "actual",
        "current",
        "alertcolor",
        "country",
    ] + _TIMELINE_NUMERIC_COLS

    rows = []
    for item in items:
        row = {k: item.get(k) for k in keep_cols}
        status = item.get("storm_status", [])
        row["storm_status"] = status[0] if status else None
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in _TIMELINE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["advisory_datetime"] = pd.to_datetime(df["advisory_datetime"])
    df["advisory_number"] = pd.to_numeric(
        df["advisory_number"], errors="coerce"
    )
    return df.sort_values("advisory_number").reset_index(drop=True)


def get_impact_by_country(
    eventid: int, aggregate: bool = True
) -> Dict[str, pd.DataFrame]:
    """Fetch population exposure per wind buffer for one TC event.

    Returns country-level (or admin-level) population exposed within
    the cumulative wind footprint across the full storm track.

    Note: only returns data for events ~2022 onward. Earlier events
    return zeros. Use :func:`get_timeline` for 2015+ coverage.

    Parameters
    ----------
    eventid : int
    aggregate : bool, default True
        Sum to country level if True; per-admin-unit rows if False.

    Returns
    -------
    dict
        Maps buffer key (``"buffer39"`` or ``"buffer74"``) to a
        DataFrame with columns ``iso3``, ``country``, ``pop_admin``,
        ``pop_affected``, ``distance_km``.
    """
    detail = get_event_detail(eventid)
    impacts = detail.get("properties", {}).get("impacts", [])

    result: Dict[str, pd.DataFrame] = {}
    for imp in impacts:
        resource = imp.get("resource", {})
        for key in ("buffer39", "buffer74"):
            if key not in resource:
                continue
            r = requests.get(resource[key], timeout=_TIMEOUT)
            r.raise_for_status()
            datums = r.json().get("datums", [])

            countries = []
            for datum_group in datums:
                if datum_group.get("alias") != "alert":
                    continue
                for d in datum_group.get("datum", []):
                    scalars = {
                        s["name"]: s["value"]
                        for s in d.get("scalars", {}).get("scalar", [])
                    }
                    if "GMI_CNTRY" not in scalars:
                        continue
                    countries.append(
                        {
                            "iso3": scalars.get("GMI_CNTRY"),
                            "country": scalars.get("CNTRY_NAME"),
                            "pop_admin": int(
                                float(scalars.get("POP_ADMIN", 0))
                            ),
                            "pop_affected": int(
                                float(scalars.get("POP_AFFECTED", 0))
                            ),
                            "distance_km": round(
                                float(scalars.get("distance", 0)), 1
                            ),
                        }
                    )

            df = pd.DataFrame(countries)
            if aggregate and len(df) > 0:
                df = (
                    df.groupby(["iso3", "country"], as_index=False)
                    .agg({"pop_affected": "sum"})
                    .sort_values("pop_affected", ascending=False)
                    .reset_index(drop=True)
                )
            result[key] = df

    return result


def _event_feature_to_row(
    feature: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten a GDACS event feature JSON to a row dict."""
    p = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates") or [
        None,
        None,
    ]
    severity = p.get("severitydata") or {}
    return {
        "eventid": int(p["eventid"]),
        "name": p.get("eventname") or p.get("name") or "",
        "alert_level": p.get("alertlevel", ""),
        "episode_alert": p.get("episodealertlevel", ""),
        "iso3": p.get("iso3", ""),
        "country": p.get("country", ""),
        "from_date": p.get("fromdate", ""),
        "to_date": p.get("todate", ""),
        "is_current": str(p.get("iscurrent", "")).lower() == "true",
        "severity_kmh": severity.get("severity"),
        "severity_text": severity.get("severitytext"),
        "source": p.get("source", ""),
        "lon": coords[0],
        "lat": coords[1],
    }
