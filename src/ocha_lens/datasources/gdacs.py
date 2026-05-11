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
from typing import Any, Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests

from ocha_lens.utils.storm import _to_gdf

logger = logging.getLogger(__name__)

BASE_URL = "https://www.gdacs.org/gdacsapi/api"

_TIMEOUT = 30
_PAGE_SIZE = 100
_EVENT_TYPE = "TC"
_BUFFER_KEYS = ("buffer39", "buffer74")

AlertLevel = Literal["Green", "Orange", "Red"]

# Alert levels are queried one at a time and results deduped on
# eventid. The combined-alertlevel API filter is unreliable, and a
# storm's alert level can change during its lifetime — without
# per-level iteration + dedup, the same event would appear multiple
# times or be missed entirely.
_ALERT_LEVELS: Tuple[AlertLevel, ...] = ("Green", "Orange", "Red")

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

_TIMELINE_COLS = (
    [
        "advisory_number",
        "name",
        "advisory_datetime",
        "actual",
        "current",
        "alertcolor",
        "country",
    ]
    + _TIMELINE_NUMERIC_COLS
    + ["storm_status"]
)

_EVENT_ROW_KEYS = [
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
    "longitude",
    "latitude",
]


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    allow_empty: bool = False,
) -> Any:
    """GET a URL and return parsed JSON, raising on HTTP error.

    If ``allow_empty`` is True, returns None for an empty response
    body instead of raising — used by paginated endpoints where an
    empty body signals end-of-pages.
    """
    r = _get_session().get(url, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    if allow_empty and not r.text.strip():
        return None
    return r.json()


def get_events(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    alert_levels: Optional[List[AlertLevel]] = None,
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
                "eventlist": _EVENT_TYPE,
                "alertlevel": level,
                "pageSize": page_size,
                "pageNumber": page,
            }
            if from_date:
                params["fromDate"] = from_date
            if to_date:
                params["toDate"] = to_date

            data = _get_json(
                f"{BASE_URL}/events/geteventlist/SEARCH",
                params=params,
                allow_empty=True,
            )
            if data is None:
                break
            features = data.get("features", [])
            if not features:
                break

            for f in features:
                row = _event_feature_to_row(f)
                if source and row["source"] != source:
                    continue
                if row["eventid"] in seen:
                    continue
                seen[row["eventid"]] = row

            if len(features) < page_size:
                break
            page += 1

    if not seen:
        return _to_gdf(pd.DataFrame(columns=_EVENT_ROW_KEYS))
    return _to_gdf(pd.DataFrame(list(seen.values())))


def get_event_detail(eventid: int) -> Dict[str, Any]:
    """Fetch full event detail JSON.

    Includes the episode list and resource URLs for the timeline and
    per-buffer impact endpoints.
    """
    return _get_json(
        f"{BASE_URL}/events/geteventdata",
        params={"eventtype": _EVENT_TYPE, "eventid": eventid},
    )


def get_episode_detail(eventid: int, episodeid: int) -> Dict[str, Any]:
    """Fetch full episode detail JSON.

    Each episode is one model run (issuance), typically every 6 hours.
    The structure mirrors :func:`get_event_detail`, but resource URLs
    point to episode-specific data.
    """
    return _get_json(
        f"{BASE_URL}/events/getepisodedata",
        params={
            "eventtype": _EVENT_TYPE,
            "eventid": eventid,
            "episodeid": episodeid,
        },
    )


def get_timeline(eventid: int) -> pd.DataFrame:
    """Fetch the advisory timeline for one TC event.

    Each row is one advisory with position, wind speed, population
    exposure (``pop39``/``pop74``), and quadrant wind radii.
    ``pop39``/``pop74`` are instantaneous snapshots — not cumulative.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``advisory_number`` ascending. Empty DataFrame with
        the expected columns if no timeline exists for the event.
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
        return pd.DataFrame(columns=_TIMELINE_COLS)

    items = _get_json(timeline_url).get("channel", {}).get("item", [])
    if not items:
        return pd.DataFrame(columns=_TIMELINE_COLS)

    rows = []
    for item in items:
        row = {k: item.get(k) for k in _TIMELINE_COLS if k != "storm_status"}
        status = item.get("storm_status", [])
        row["storm_status"] = status[0] if status else None
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in _TIMELINE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["advisory_datetime"] = pd.to_datetime(df["advisory_datetime"])
    df["advisory_number"] = pd.to_numeric(
        df["advisory_number"], errors="coerce"
    )
    return df.sort_values("advisory_number").reset_index(drop=True)


_ADM0_COLUMNS = ["iso3", "country", "pop_affected", "distance_km"]

_ADM1_COLUMNS = [
    "iso3",
    "country",
    "fips_admin",
    "gmi_admin",
    "admin_name",
    "admin_type",
    "pop_admin",
    "pop_affected",
    "distance_km",
]


def get_exposure_adm0(
    eventid: int, episodeid: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """Fetch country-level population exposure per wind buffer.

    Reads ``datums[alias='country']`` from each buffer's impact JSON —
    the canonical ADM0 rollup. One row per affected country, per
    buffer.

    Note: only returns data for events ~2022 onward. Earlier events
    return empty results. Use :func:`get_timeline` for 2015+ coverage.

    Parameters
    ----------
    eventid : int
    episodeid : int, optional
        If provided, fetches that specific episode's snapshot via
        :func:`get_episode_detail`. Otherwise uses the event-level
        resource URLs from :func:`get_event_detail` (latest snapshot).

    Returns
    -------
    dict
        Maps buffer key (``"buffer39"`` or ``"buffer74"``) to a
        DataFrame with columns ``iso3``, ``country``, ``pop_affected``,
        ``distance_km``.
    """
    return _exposure_per_buffer(eventid, episodeid, _parse_adm0)


def get_exposure_adm1(
    eventid: int, episodeid: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """Fetch ADM1-grain population exposure per wind buffer.

    Reads ``datums[alias='alert']`` from each buffer's impact JSON.
    One row per affected sub-national admin unit, per buffer. Country
    identifiers are attached so callers can roll up to ADM0 if needed.

    Parameters
    ----------
    eventid : int
    episodeid : int, optional
        Same semantics as :func:`get_exposure_adm0`.

    Returns
    -------
    dict
        Maps buffer key to a DataFrame with columns ``iso3``,
        ``country``, ``fips_admin``, ``gmi_admin``, ``admin_name``,
        ``admin_type``, ``pop_admin``, ``pop_affected``,
        ``distance_km``.
    """
    return _exposure_per_buffer(eventid, episodeid, _parse_adm1)


def _exposure_per_buffer(
    eventid: int,
    episodeid: Optional[int],
    parse,
) -> Dict[str, pd.DataFrame]:
    if episodeid is None:
        detail = get_event_detail(eventid)
    else:
        detail = get_episode_detail(eventid, episodeid)
    impacts = detail.get("properties", {}).get("impacts", [])

    result: Dict[str, pd.DataFrame] = {}
    for imp in impacts:
        resource = imp.get("resource", {})
        for key in _BUFFER_KEYS:
            url = resource.get(key)
            if not url:
                continue
            result[key] = parse(_get_json(url))
    return result


def _parse_adm0(data: Dict[str, Any]) -> pd.DataFrame:
    """Read ``alias='country'`` (ADM0 rollups) from a buffer JSON."""
    rows = []
    for dg in data.get("datums", []):
        if dg.get("alias") != "country":
            continue
        for d in dg.get("datum", []):
            scalars = _scalars(d)
            iso3 = scalars.get("ISO_3DIGIT")
            if not iso3:
                continue
            rows.append(
                {
                    "iso3": iso3,
                    "country": scalars.get("CNTRY_NAME"),
                    "pop_affected": int(float(scalars.get("POP_AFFECTED", 0))),
                    "distance_km": round(float(scalars.get("distance", 0)), 1),
                }
            )
    return pd.DataFrame(rows, columns=_ADM0_COLUMNS)


def _parse_adm1(data: Dict[str, Any]) -> pd.DataFrame:
    """Read ``alias='alert'`` (ADM1 grain) from a buffer JSON."""
    rows = []
    for dg in data.get("datums", []):
        if dg.get("alias") != "alert":
            continue
        for d in dg.get("datum", []):
            scalars = _scalars(d)
            if not scalars.get("GMI_ADMIN"):
                continue
            rows.append(
                {
                    "iso3": scalars.get("GMI_CNTRY"),
                    "country": scalars.get("CNTRY_NAME"),
                    "fips_admin": scalars.get("FIPS_ADMIN"),
                    "gmi_admin": scalars.get("GMI_ADMIN"),
                    "admin_name": scalars.get("ADMIN_NAME"),
                    "admin_type": scalars.get("TYPE_ENG"),
                    "pop_admin": int(float(scalars.get("POP_ADMIN", 0))),
                    "pop_affected": int(float(scalars.get("POP_AFFECTED", 0))),
                    "distance_km": round(float(scalars.get("distance", 0)), 1),
                }
            )
    return pd.DataFrame(rows, columns=_ADM1_COLUMNS)


def _scalars(datum: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a GDACS ``scalars`` container into a name→value dict."""
    return {
        s["name"]: s["value"]
        for s in datum.get("scalars", {}).get("scalar", [])
    }


def _event_feature_to_row(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a GDACS event feature JSON to a row dict."""
    p = feature.get("properties", {})
    coords = feature.get("geometry", {}).get("coordinates") or [None, None]
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
        "longitude": coords[0],
        "latitude": coords[1],
    }
