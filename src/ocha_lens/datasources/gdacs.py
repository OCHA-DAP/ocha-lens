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
import numpy as np
import pandas as pd
import pandera.pandas as pa
import requests

from ocha_lens.utils.storm import _to_gdf

logger = logging.getLogger(__name__)

BASE_URL = "https://www.gdacs.org/gdacsapi/api"

_TIMEOUT = 30
_PAGE_SIZE = 100
_EVENT_TYPE = "TC"
_BUFFER_KEYS = ("buffer39", "buffer74")

# Mapping of GDACS proprietary country codes to standard ISO 3166-1
# alpha-3. Most GDACS GMI_CNTRY values ARE valid ISO3 (USA, MEX, BHS,
# etc.); the entries here are the exceptions — territories and
# dependencies where GDACS uses an X-prefixed internal code. Add new
# entries here as new GDACS proprietary codes surface in real data.
GDACS_PROPRIETARY_TO_ISO3 = {
    "XJE": "JEY",   # Jersey (British Crown Dependency)
    # "XGG": "GGY",  # Guernsey — add if encountered
    # "XIM": "IMN",  # Isle of Man — add if encountered
}


def to_iso3(gdacs_country_code: str) -> str:
    """Map a GDACS GMI_CNTRY code to standard ISO 3166-1 alpha-3.

    Most GDACS codes are already valid ISO3 — pass-through. Only the
    X-prefixed proprietary codes (see ``GDACS_PROPRIETARY_TO_ISO3``)
    get remapped.
    """
    return GDACS_PROPRIETARY_TO_ISO3.get(
        gdacs_country_code, gdacs_country_code
    )

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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
#
# Types chosen to map cleanly to the SQL columns the pipeline writes
# (see ds-storms-pipeline/src/schemas/sql/gdacs_exposure.sql):
#   pop_affected → INTEGER     ⇒  pa.Column(int)
#   distance_km  → DOUBLE      ⇒  pa.Column(float)
#   iso3         → VARCHAR(3)  ⇒  pa.Column(str)
#   valid_time   → TIMESTAMP   ⇒  pa.Column(pd.Timestamp)
#
# Validation is called at the return point of each public function.

EVENT_SCHEMA = pa.DataFrameSchema(
    {
        "eventid": pa.Column(int, nullable=False),
        "name": pa.Column(str, nullable=True),
        "alert_level": pa.Column(
            str,
            pa.Check.isin(["Green", "Orange", "Red", ""]),
            nullable=True,
        ),
        "episode_alert": pa.Column(str, nullable=True),
        "iso3": pa.Column(str, nullable=True),
        "country": pa.Column(str, nullable=True),
        "from_date": pa.Column(str, nullable=True),
        "to_date": pa.Column(str, nullable=True),
        "is_current": pa.Column(bool, nullable=False),
        "severity_kmh": pa.Column(float, nullable=True),
        "severity_text": pa.Column(str, nullable=True),
        "source": pa.Column(
            str,
            pa.Check.isin(["NOAA", "JTWC", ""]),
            nullable=True,
        ),
        "geometry": pa.Column(gpd.array.GeometryDtype, nullable=True),
    },
    strict=True,
    coerce=True,
)

TIMELINE_SCHEMA = pa.DataFrameSchema(
    {
        "advisory_number": pa.Column("Int64", nullable=True),
        "name": pa.Column(str, nullable=True),
        "advisory_datetime": pa.Column(pd.Timestamp, nullable=True),
        "actual": pa.Column(str, nullable=True),
        "current": pa.Column(str, nullable=True),
        "alertcolor": pa.Column(str, nullable=True),
        "country": pa.Column(str, nullable=True),
        **{c: pa.Column(float, nullable=True) for c in _TIMELINE_NUMERIC_COLS},
        "storm_status": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
)

ADM0_EXPOSURE_SCHEMA = pa.DataFrameSchema(
    {
        "iso3": pa.Column(str, nullable=False),
        "country": pa.Column(str, nullable=True),
        # pop_affected legitimately missing when no population is
        # exposed to this wind threshold — preserved as null
        "pop_affected": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "distance_km": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    strict=True,
    coerce=True,
)

ADM1_EXPOSURE_SCHEMA = pa.DataFrameSchema(
    {
        "iso3": pa.Column(str, nullable=False),
        "country": pa.Column(str, nullable=True),
        "fips_admin": pa.Column(str, nullable=True),
        "gmi_admin": pa.Column(str, nullable=False),
        "admin_name": pa.Column(str, nullable=True),
        "admin_type": pa.Column(str, nullable=True),
        "pop_admin": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "pop_affected": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "distance_km": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    strict=True,
    coerce=True,
)


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
    return EVENT_SCHEMA.validate(_to_gdf(pd.DataFrame(list(seen.values()))))


def get_event_detail(eventid: int) -> Dict[str, Any]:
    """Fetch full event detail JSON.

    Includes the episode list and resource URLs for the timeline and
    per-buffer impact endpoints.
    """
    return _get_json(
        f"{BASE_URL}/events/geteventdata",
        params={"eventtype": _EVENT_TYPE, "eventid": eventid},
    )


class NoEpisodesError(ValueError):
    """Raised when a GDACS event has no episodes in its detail."""


class EpisodeUrlFormatError(ValueError):
    """Raised when GDACS's episode `details` URL doesn't contain a
    parseable `episodeid=` query parameter (API contract change)."""


def latest_episode_id(event_detail: Dict[str, Any]) -> int:
    """Pull the latest episode id out of an event-detail JSON.

    GDACS's ``properties.episodes`` is a list of ``{details, ...}``
    dicts; the episode id is embedded in the ``details`` URL as a
    query param. This helper hides that GDACS-internal quirk.

    Raises
    ------
    NoEpisodesError
        Event has no episodes (legitimately new event with no
        advisories yet — callers may want to handle this).
    EpisodeUrlFormatError
        Episode URL is malformed — likely a GDACS API contract
        change; should surface loudly rather than be skipped.
    """
    episodes = event_detail["properties"]["episodes"]
    if not episodes:
        raise NoEpisodesError("event has no episodes")
    url = episodes[-1]["details"]
    if "episodeid=" not in url:
        raise EpisodeUrlFormatError(
            f"`episodeid=` not found in details URL: {url!r}"
        )
    return int(url.split("episodeid=")[-1].split("&")[0])


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


class NoTimelineError(ValueError):
    """Raised when a GDACS event has no timeline resource URL in its
    impact entries. Distinguishable from "timeline exists but empty"
    (which returns an empty DataFrame, validated by TIMELINE_SCHEMA).
    """


def get_timeline(
    eventid: int, detail: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Fetch the advisory timeline for one TC event.

    Each row is one advisory with position, wind speed, population
    exposure (``pop39``/``pop74``), and quadrant wind radii.
    ``pop39``/``pop74`` are instantaneous snapshots — not cumulative.

    Parameters
    ----------
    eventid : int
    detail : dict, optional
        Pre-fetched output of :func:`get_event_detail` for this
        eventid. Pass it to skip the internal detail fetch when the
        caller already has it (saves one HTTP round-trip).

    Raises
    ------
    NoTimelineError
        Event has no timeline resource URL in its impacts list
        (legitimately rare — most TC events have one).

    Returns
    -------
    pandas.DataFrame
        Sorted by ``advisory_number`` ascending. Empty DataFrame
        (with expected columns) if the timeline endpoint returns
        no items — pandera validates the row structure either way.
    """
    if detail is None:
        detail = get_event_detail(eventid)
    impacts = detail["properties"]["impacts"]

    timeline_url = None
    for imp in impacts:
        url = imp.get("resource", {}).get("timeline")
        if url:
            timeline_url = url
            break

    if timeline_url is None:
        raise NoTimelineError(f"no timeline resource for event {eventid}")

    items = _get_json(timeline_url)["channel"]["item"]
    if not items:
        return TIMELINE_SCHEMA.validate(pd.DataFrame(columns=_TIMELINE_COLS))

    rows = []
    for item in items:
        row = {k: item.get(k) for k in _TIMELINE_COLS if k != "storm_status"}
        status = item.get("storm_status", [])
        row["storm_status"] = status[0] if status else None
        rows.append(row)

    df = pd.DataFrame(rows)
    # Numeric coercion is strict (errors propagate); a GDACS field
    # appearing with an unparseable value should surface, not be
    # silently NaN'd. Empty strings are treated as null below.
    for col in _TIMELINE_NUMERIC_COLS:
        df[col] = df[col].replace("", None)
        df[col] = pd.to_numeric(df[col])
    df["advisory_datetime"] = pd.to_datetime(df["advisory_datetime"])
    df["advisory_number"] = df["advisory_number"].replace("", None)
    df["advisory_number"] = pd.to_numeric(df["advisory_number"])
    return TIMELINE_SCHEMA.validate(
        df.sort_values("advisory_number").reset_index(drop=True)
    )


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
    eventid: int,
    episodeid: Optional[int] = None,
    detail: Optional[Dict[str, Any]] = None,
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
    detail : dict, optional
        Pre-fetched output of :func:`get_event_detail`. Pass it to
        skip the internal detail fetch on the event-level path.
        Mutually exclusive with ``episodeid`` — passing both raises
        ``ValueError`` (the two args describe different snapshots).

    Returns
    -------
    dict
        Maps buffer key (``"buffer39"`` or ``"buffer74"``) to a
        DataFrame with columns ``iso3``, ``country``, ``pop_affected``,
        ``distance_km``.
    """
    return _exposure_per_buffer(
        eventid, episodeid, _parse_adm0, detail=detail
    )


def get_exposure_adm1(
    eventid: int,
    episodeid: Optional[int] = None,
    detail: Optional[Dict[str, Any]] = None,
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
    detail : dict, optional
        Same semantics as :func:`get_exposure_adm0`.

    Returns
    -------
    dict
        Maps buffer key to a DataFrame with columns ``iso3``,
        ``country``, ``fips_admin``, ``gmi_admin``, ``admin_name``,
        ``admin_type``, ``pop_admin``, ``pop_affected``,
        ``distance_km``.
    """
    return _exposure_per_buffer(
        eventid, episodeid, _parse_adm1, detail=detail
    )


def match_to_atcf(
    timeline: pd.DataFrame,
    nhc_tracks: pd.DataFrame,
    max_avg_dist_deg: float = 0.1,
) -> Optional[str]:
    """Match a GDACS timeline to an NHC ``atcf_id``.

    One pandas join, no strategy ordering, no mode flags. Inner-join
    GDACS rows to NHC tracks on ``valid_time`` (which is
    ``advisory_datetime`` on the GDACS side — that field is overloaded
    to be the issued time on ``actual=True`` rows and the forecast
    valid_time on ``actual=False`` rows), compute L2 distance per
    joined row, group by ``atcf_id``, pick the storm with the smallest
    mean distance.

    Why it works across A-deck-fed and TCM-fed NHC tables:

    - A-deck mode: A-deck ``leadtime=0`` rows have ``valid_time`` at
      synoptic time (e.g. 18Z), but GDACS ``actual=True`` rows have
      ``advisory_datetime`` at issue time (21Z) — those rows don't
      collide, so A-deck ``leadtime=0`` naturally drops out of the
      join. A-deck forecast rows at the same future ``valid_time``
      as GDACS forecast rows match at ~0 deg.
    - TCM mode: TCM ``leadtime=0`` ``valid_time`` equals GDACS issue
      time (both at 21Z) — those rows DO match, contributing another
      ~0-deg signal. Forecast rows match as in A-deck mode.

    In either mode, the correct storm's joined rows average to
    ~0 deg; wrong storms produce large mean distances. The smallest
    aggregate wins.

    Parameters
    ----------
    timeline : DataFrame
        Output of :func:`get_timeline`. Requires columns
        ``advisory_datetime`` (used as valid_time for the join),
        ``latitude``, ``longitude``.
    nhc_tracks : DataFrame
        Caller-loaded NHC track rows. Requires columns ``atcf_id``,
        ``valid_time``, ``lat``, ``lon``.

        **Caller MUST pre-filter to one row per (atcf_id,
        valid_time) — the freshest issuance.** If the caller passes
        multiple issuances' forecasts for the same valid_time, the
        join averages across stale forecasts and the true match's
        mean distance inflates well above zero (verified against
        MILTON-24: stale-forecast contamination drives the correct
        match's mean from 0.000 to ~3 deg, defeating the tolerance
        check).

        SQL pattern::

            SELECT DISTINCT ON (atcf_id, valid_time)
                   atcf_id, valid_time, ST_Y(geom) lat, ST_X(geom) lon
            FROM storms.nhc_tracks_geo
            WHERE ...
            ORDER BY atcf_id, valid_time, issued_time DESC
    max_avg_dist_deg : float, default 0.1
        Maximum acceptable mean L2 distance across joined rows for
        the winning ``atcf_id``. True matches are ~0.000 deg.

    Returns
    -------
    atcf_id (e.g. ``"AL142024"``) or ``None`` if no candidate
    averages within tolerance (or if the join is empty).
    """
    g = timeline[["advisory_datetime", "latitude", "longitude"]].rename(
        columns={
            "advisory_datetime": "valid_time",
            "latitude": "g_lat",
            "longitude": "g_lon",
        }
    )
    joined = g.merge(
        nhc_tracks[["atcf_id", "valid_time", "lat", "lon"]],
        on="valid_time",
        how="inner",
    )
    if len(joined) == 0:
        return None
    joined["dist"] = np.sqrt(
        (joined["g_lat"] - joined["lat"]) ** 2
        + (joined["g_lon"] - joined["lon"]) ** 2
    )
    agg = joined.groupby("atcf_id")["dist"].mean().sort_values()
    if agg.iloc[0] > max_avg_dist_deg:
        return None
    return agg.index[0]


def _exposure_per_buffer(
    eventid: int,
    episodeid: Optional[int],
    parse,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    if episodeid is not None and detail is not None:
        # The two args describe different snapshots (event-level vs.
        # a specific episode). Accepting both silently would mean
        # the caller's `detail` is wasted; force them to pick one.
        raise ValueError(
            "pass either `episodeid` or `detail`, not both"
        )
    if episodeid is not None:
        detail = get_episode_detail(eventid, episodeid)
    elif detail is None:
        detail = get_event_detail(eventid)
    impacts = detail["properties"]["impacts"]

    result: Dict[str, pd.DataFrame] = {}
    for imp in impacts:
        # `resource` may legitimately omit individual buffer keys for
        # events with no exposure at that wind threshold — handle
        # missing keys explicitly rather than catching it everywhere.
        resource = imp["resource"]
        for key in _BUFFER_KEYS:
            if key not in resource:
                continue
            result[key] = parse(_get_json(resource[key]))
    return result


def _parse_adm0(data: Dict[str, Any]) -> pd.DataFrame:
    """Read ``alias='country'`` (ADM0 rollups) from a buffer JSON.

    The ``iso3`` column holds GDACS' native ``GMI_CNTRY`` (which is
    usually a valid ISO3 like "USA" but is X-prefixed for some
    non-sovereign territories like "XJE"=Jersey). Callers should apply
    :func:`to_iso3` to get a standardized ISO3 code.

    ``GMI_CNTRY`` is required — KeyError if GDACS omits it (contract
    break). Population/distance fields are preserved as None when
    missing (GDACS legitimately omits POP_AFFECTED when no population
    is exposed at that wind threshold).
    """
    rows = []
    for dg in data.get("datums", []):
        if dg.get("alias") != "country":
            continue
        for d in dg.get("datum", []):
            scalars = _scalars(d)
            rows.append(
                {
                    "iso3": scalars["GMI_CNTRY"],
                    "country": scalars.get("CNTRY_NAME"),
                    "pop_affected": _to_nullable_int(scalars.get("POP_AFFECTED")),
                    "distance_km": _to_nullable_round(scalars.get("distance")),
                }
            )
    return ADM0_EXPOSURE_SCHEMA.validate(
        pd.DataFrame(rows, columns=_ADM0_COLUMNS)
    )


def _parse_adm1(data: Dict[str, Any]) -> pd.DataFrame:
    """Read ``alias='alert'`` (ADM1 grain) from a buffer JSON.

    Identifier fields (``GMI_CNTRY``, ``GMI_ADMIN``) are required —
    KeyError if GDACS omits them. Pop/distance preserved as None
    when missing.
    """
    rows = []
    for dg in data.get("datums", []):
        if dg.get("alias") != "alert":
            continue
        for d in dg.get("datum", []):
            scalars = _scalars(d)
            rows.append(
                {
                    "iso3": scalars["GMI_CNTRY"],
                    "country": scalars.get("CNTRY_NAME"),
                    "fips_admin": scalars.get("FIPS_ADMIN"),
                    "gmi_admin": scalars["GMI_ADMIN"],
                    "admin_name": scalars.get("ADMIN_NAME"),
                    "admin_type": scalars.get("TYPE_ENG"),
                    "pop_admin": _to_nullable_int(scalars.get("POP_ADMIN")),
                    "pop_affected": _to_nullable_int(scalars.get("POP_AFFECTED")),
                    "distance_km": _to_nullable_round(scalars.get("distance")),
                }
            )
    return ADM1_EXPOSURE_SCHEMA.validate(
        pd.DataFrame(rows, columns=_ADM1_COLUMNS)
    )


# GDACS uses -99999 as a sentinel for "no data" in numeric scalar
# fields (POP_AFFECTED, POP_ADMIN, distance). Normalized to None
# here so downstream code sees one consistent missing representation.
_GDACS_MISSING_SENTINEL = -99999


def _to_nullable_int(value: Any) -> Optional[int]:
    """Cast a scalar to int, preserving None for missing/empty/sentinel."""
    if value is None or value == "":
        return None
    n = int(float(value))
    if n == _GDACS_MISSING_SENTINEL:
        return None
    return n


def _to_nullable_round(value: Any) -> Optional[float]:
    """Cast a scalar to rounded float, preserving None for missing/empty/sentinel."""
    if value is None or value == "":
        return None
    f = round(float(value), 1)
    if int(f) == _GDACS_MISSING_SENTINEL:
        return None
    return f


def _scalars(datum: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a GDACS ``scalars`` container into a name→value dict."""
    return {
        s["name"]: s["value"]
        for s in datum.get("scalars", {}).get("scalar", [])
    }


def _event_feature_to_row(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a GDACS event feature JSON to a row dict.

    Required fields (eventid, eventname, geometry coordinates,
    fromdate/todate, alertlevel) use direct subscripts — KeyError
    if GDACS drops them. Optional/metadata fields (iso3, country,
    severitydata) use ``.get`` returning None — these are
    legitimately absent for some events and downstream code is
    expected to handle null.
    """
    p = feature["properties"]
    coords = feature["geometry"]["coordinates"]
    severity = p.get("severitydata") or {}
    return {
        "eventid": int(p["eventid"]),
        "name": p["eventname"],
        "alert_level": p["alertlevel"],
        "episode_alert": p["episodealertlevel"],
        "iso3": p.get("iso3"),
        "country": p.get("country"),
        "from_date": p["fromdate"],
        "to_date": p["todate"],
        "is_current": str(p["iscurrent"]).lower() == "true",
        "severity_kmh": severity.get("severity"),
        "severity_text": severity.get("severitytext"),
        "source": p.get("source"),
        "longitude": coords[0],
        "latitude": coords[1],
    }
