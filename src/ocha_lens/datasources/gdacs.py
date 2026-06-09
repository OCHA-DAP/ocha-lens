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
from collections import Counter
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
    "XJE": "JEY",  # Jersey (British Crown Dependency)
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
    # GDACS timelines mix full and abbreviated month names within the same
    # event ("01 January 2025" vs "01 Jun 2025"). Bare pd.to_datetime infers
    # one format from the first row and chokes on the other; format="mixed"
    # parses each value independently.
    df["advisory_datetime"] = pd.to_datetime(
        df["advisory_datetime"],
        format="mixed",
    )
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
    return _exposure_per_buffer(eventid, episodeid, _parse_adm0, detail=detail)


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
    return _exposure_per_buffer(eventid, episodeid, _parse_adm1, detail=detail)


def _is_actual(actual: pd.Series) -> pd.Series:
    """Boolean mask for observed (vs forecast) advisories.

    GDACS stores ``actual`` as a string (``"True"``/``"False"``);
    compare case-insensitively rather than trusting the dtype.
    """
    return actual.astype(str).str.lower() == "true"


def _split_timeline(
    timeline: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a GDACS timeline into ``(observed, forecast)`` advisories.

    Each timeline row is flagged ``actual``: observed positions
    (``actual="True"``) and forecast-cone positions
    (``actual="False"``). The two are matched differently — see
    :func:`match_to_atcf`.
    """
    actual_mask = _is_actual(timeline["actual"])
    return timeline[actual_mask], timeline[~actual_mask]


def _nearest_atcf_at(
    valid_time: pd.Timestamp,
    lat: float,
    lon: float,
    nhc_tracks: pd.DataFrame,
    max_dist_deg: float,
) -> Optional[str]:
    """Closest NHC ``atcf_id`` to ``(lat, lon)`` among rows at exactly
    ``valid_time``.

    The shared primitive behind both matching strategies. Returns the
    ``atcf_id`` of the nearest row within ``max_dist_deg``, or
    ``None`` when no NHC row shares the timestamp or the closest one
    is too far.
    """
    at_time = nhc_tracks[nhc_tracks["valid_time"] == valid_time]
    if at_time.empty:
        return None
    dist = np.sqrt((at_time["lat"] - lat) ** 2 + (at_time["lon"] - lon) ** 2)
    best_idx = dist.idxmin()
    if dist.loc[best_idx] > max_dist_deg:
        return None
    return at_time.loc[best_idx, "atcf_id"]


def _match_by_forecast_cone(
    forecast: pd.DataFrame,
    nhc_tracks: pd.DataFrame,
    max_dist_deg: float,
) -> Optional[str]:
    """Strategy 1 (primary) — match on the shared forecast cone.

    A GDACS forecast point and the NHC forecast for the same
    ``valid_time`` are the *same* NHC forecast package (GDACS scrapes
    the TCM advisory, which carries the synoptic A-deck cone forward
    unchanged), so for the correct storm they agree to ~0°. Each
    forecast point that lands on an NHC row casts a vote for its
    ``atcf_id``; the most-voted ``atcf_id`` wins.

    Voting across the whole cone is what makes this robust: a wrong
    storm would have to share an exact forecast position at the same
    valid_time, which doesn't happen. It also self-heals late
    captures — it never needs the genesis cycle, only the cone of
    whatever advisory we did capture.

    Ties (a basin-crossing storm whose cone touches two ``atcf_id``
    values) go to the earliest ``valid_time``, i.e. the genesis basin.
    """
    votes = [
        atcf
        for _, row in forecast.sort_values("advisory_datetime").iterrows()
        if (
            atcf := _nearest_atcf_at(
                row["advisory_datetime"],
                row["latitude"],
                row["longitude"],
                nhc_tracks,
                max_dist_deg,
            )
        )
        is not None
    ]
    if not votes:
        return None
    counts = Counter(votes)
    most_votes = max(counts.values())
    leaders = {atcf for atcf, n in counts.items() if n == most_votes}
    # votes is in valid_time order, so the first leader encountered is
    # the earliest-matching (genesis-basin) atcf_id.
    return next(atcf for atcf in votes if atcf in leaders)


def _match_by_genesis(
    observed: pd.DataFrame,
    nhc_tracks: pd.DataFrame,
    max_dist_deg: float,
) -> Optional[str]:
    """Strategy 2 (fallback) — match the genesis observed position.

    Used only when the timeline carries no forecast cone (e.g. a
    dissipated storm pulled from the archive). Takes the genesis
    advisory (smallest ``advisory_number``) and requires an NHC row
    at that exact ``valid_time``. It's a single point, so it depends
    on that one genesis cycle being present in ``nhc_tracks`` — which
    is exactly why it's the fallback, not the primary.
    """
    if observed.empty:
        return None
    genesis = observed.sort_values("advisory_number").iloc[0]
    return _nearest_atcf_at(
        genesis["advisory_datetime"],
        genesis["latitude"],
        genesis["longitude"],
        nhc_tracks,
        max_dist_deg,
    )


def match_to_atcf(
    timeline: pd.DataFrame,
    nhc_tracks: pd.DataFrame,
    max_dist_deg: float = 0.05,
) -> Optional[str]:
    """Match a GDACS event to an NHC ``atcf_id``.

    GDACS and our NHC table are two views of the *same* NHC
    forecaster output (GDACS scrapes the TCM Forecast/Advisory; we
    store the A-deck OFCL). They're timestamped 3h apart — A-deck on
    synoptic valid times (00/06/12/18Z), TCM on advisory issue times
    (03/09/15/21Z) — but the **forecast cone (t>0) is identical
    between them at shared valid_times**. Only the observed t=0
    position differs (the storm moved in the intervening 3h).

    That asymmetry drives a two-strategy match:

    1. :func:`_match_by_forecast_cone` (primary) — vote the GDACS
       forecast points onto NHC ``atcf_id`` values by exact valid_time.
       Robust (many points), product-agnostic, and self-healing for
       storms whose first advisories we never captured.
    2. :func:`_match_by_genesis` (fallback) — single-point match on
       the genesis observed advisory, for completed storms whose
       timeline has no forecast cone left.

    Parameters
    ----------
    timeline : DataFrame
        Output of :func:`get_timeline`. Required columns:
        ``advisory_number``, ``actual``, ``advisory_datetime``,
        ``latitude``, ``longitude``.
    nhc_tracks : DataFrame
        NHC tracks deduped to one row per ``(atcf_id, valid_time)``
        at the freshest issuance (all leadtimes kept — the forecast
        cone lives at leadtime>0). Required columns: ``atcf_id``,
        ``valid_time``, ``lat``, ``lon``. See
        ``load_freshest_nhc_tracks`` in the pipeline.
    max_dist_deg : float, default 0.05
        Spatial tolerance in degrees. Cone matches for the correct
        storm are ~0°; 0.05° absorbs only rounding. Robustness comes
        from voting across the cone, not from a loose tolerance.

    Returns
    -------
    atcf_id (e.g. ``"AL142024"``) or ``None`` when neither strategy
    finds a match — the correct answer for a non-NHC (JTWC/RSMC)
    storm or a genuine gap in our NHC table.
    """
    if timeline.empty or nhc_tracks.empty:
        return None
    observed, forecast = _split_timeline(timeline)
    return _match_by_forecast_cone(
        forecast, nhc_tracks, max_dist_deg
    ) or _match_by_genesis(observed, nhc_tracks, max_dist_deg)


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
        raise ValueError("pass either `episodeid` or `detail`, not both")
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
                    "pop_affected": _to_nullable_int(
                        scalars.get("POP_AFFECTED")
                    ),
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
                    "pop_affected": _to_nullable_int(
                        scalars.get("POP_AFFECTED")
                    ),
                    "distance_km": _to_nullable_round(scalars.get("distance")),
                }
            )
    return ADM1_EXPOSURE_SCHEMA.validate(
        pd.DataFrame(rows, columns=_ADM1_COLUMNS)
    )


# GDACS uses a couple of distinct "no data" sentinels in numeric scalar
# fields (POP_AFFECTED, POP_ADMIN, distance). Both normalize to None so
# downstream code sees one consistent missing representation.
#   -99999 : structurally missing (e.g., field absent or zeroed-out)
#   -1     : "data unavailable / not computed" (observed for POP_AFFECTED
#            on some admin1 rows where POP_ADMIN itself is a real number,
#            e.g. MARIO-25 over Baja California Sur)
_GDACS_MISSING_SENTINELS = {-99999, -1}


def _to_nullable(value: Any, coerce) -> Any:
    """Cast a GDACS scalar via ``coerce(float(value))``, normalizing
    None, empty string, and the GDACS missing-data sentinels to None.
    ``coerce`` is the type-specific cast (e.g. ``int`` or
    ``lambda v: round(v, 1)``)."""
    if value is None or value == "":
        return None
    out = coerce(float(value))
    if out in _GDACS_MISSING_SENTINELS:
        return None
    return out


def _to_nullable_int(value: Any) -> Optional[int]:
    return _to_nullable(value, int)


def _to_nullable_round(value: Any) -> Optional[float]:
    return _to_nullable(value, lambda v: round(v, 1))


def _scalars(datum: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a GDACS ``scalars`` container into a name→value dict."""
    return {
        s["name"]: s["value"]
        for s in datum.get("scalars", {}).get("scalar", [])
    }


def _event_feature_to_row(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a GDACS event feature JSON to a row dict.

    Required fields (eventid, eventname, geometry coordinates,
    fromdate/todate, alertlevel, episodealertlevel, iscurrent) use
    direct subscripts — KeyError if GDACS drops them. Optional/
    metadata fields (iso3, country, severitydata, source) use
    ``.get`` returning None — these are legitimately absent for
    some events and downstream code is expected to handle null.
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
