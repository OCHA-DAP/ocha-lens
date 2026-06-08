"""WFP ADAM (Automatic Disaster Analysis and Mapping) — tropical cyclone
exposure datasource.

ADAM publishes per-event/per-episode population exposure CSVs alongside an
OGC API listing endpoint. Each CSV is at ADM2 granularity; this module
aggregates to ADM0 and ADM1 in code (mirroring Hannah's harmonisation
precedent) and returns one long-form DataFrame covering all three levels.

ADAM event_id is the same identifier GDACS uses for the same physical
event, so the join key downstream is just the event_id — no separate
cross-source matching pass needed.

The CSVs are name-only (no admin codes). The boundary set appears to be
FAO GAUL 2015 Level 2 (schema fingerprint + WFP's institutional standard).
No public GAUL → OCHA COD pcode crosswalk exists; pcode enrichment is
deferred to a future geometric-join pipeline using ADAM's per-event
wind-footprint shapefiles.
"""

import base64
import json
import logging
from io import StringIO
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import pandera.pandas as pa
import pycountry
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAM_BASE = "https://api.adam.geospatial.wfp.org/api"
COLLECTION = "adam.adam_ts_events"
_PAGE_SIZE = 100
_TIMEOUT = 30


# ADAM publishes pop counts at three wind bands (km/h). We standardize on
# the NHC kt convention for the wind_speed_kt column to stay parallel with
# GDACS / NHC tables. The km/h → kt conversion is approximate (60 km/h is
# strictly ~32 kt, 90 ≈ 49, 120 ≈ 65) but the conventional NHC thresholds
# 34/50/64 are what every consumer downstream expects.
_KMH_COL_TO_KT = {
    "POP_60_KMH": 34,
    "POP_90_KMH": 50,
    "POP_120_KMH": 64,
}


# WFP renamed the admin columns around the 2026 season: historical CSVs use
# ADM0_NAME/ADM1_NAME/ADM2_NAME, newer ones use LEVEL_0/LEVEL_1/LEVEL_2 (same
# meaning: country / state / district). We alias the new names onto the
# canonical ones so both schemas parse identically — see _csv_to_long.
_ADMIN_COL_ALIASES = {
    "LEVEL_0": "ADM0_NAME",
    "LEVEL_1": "ADM1_NAME",
    "LEVEL_2": "ADM2_NAME",
}


# Hard-coded ADM0_NAME → ISO3 overrides for territories where pycountry
# fuzzy-search either fails or resolves to the wrong country (most common
# failure: territories that report under their parent country's name). The
# list is from Hannah's harmonisation work; add new entries as ADAM throws
# unrecognized names at us.
ADAM_ISO3_OVERRIDES: Dict[str, str] = {
    "Puerto Rico (USA)": "PRI",
    "United States Virgin Islands (USA)": "VIR",
    "Saint Pierre et Miquelon": "SPM",
    "Azores Islands": "PRT",
    "Clipperton Island": "CPT",
}


# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------

_EVENT_COLS = [
    "event_id",
    "episode_id",
    "uid",
    "name",
    "source",
    "from_date",
    "to_date",
    "alert_level",
    "population_csv_url",
]

EVENT_SCHEMA = pa.DataFrameSchema(
    {
        "event_id": pa.Column(int),
        "episode_id": pa.Column(int),
        "uid": pa.Column(str),
        "name": pa.Column(str),
        "source": pa.Column(str),
        "from_date": pa.Column(pa.DateTime),
        "to_date": pa.Column(pa.DateTime),
        "alert_level": pa.Column(str),
        # Some legacy events have no static CSV — caller handles via the
        # NoExposureCSVError raised by get_exposure rather than schema reject.
        "population_csv_url": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
)


_EXPOSURE_COLS = [
    "admin_level",
    "iso3",
    "admin_name",
    "parent_admin_name",
    "wind_speed_kt",
    "pop_exposed",
]

EXPOSURE_SCHEMA = pa.DataFrameSchema(
    {
        "admin_level": pa.Column(int, pa.Check.isin([0, 1, 2])),
        # iso3 nullable because name_to_iso3 returns None for genuinely
        # unresolvable / ambiguous ADM0_NAMEs (e.g., disputed territories).
        "iso3": pa.Column(str, nullable=True),
        "admin_name": pa.Column(str),
        # parent_admin_name: country for adm1 rows, adm1 for adm2 rows,
        # null for adm0 rows. Lets consumers walk the hierarchy without a
        # join.
        "parent_admin_name": pa.Column(str, nullable=True),
        "wind_speed_kt": pa.Column(int, pa.Check.isin([34, 50, 64])),
        # pop_exposed is the cumulative count ≥ wind_speed_kt (post-
        # make_cumulative), nullable when ADAM omits the column entirely.
        "pop_exposed": pa.Column("Int64", nullable=True),
    },
    strict=True,
    coerce=True,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NoExposureCSVError(ValueError):
    """The event's latest episode has no population_csv_url. Legitimately
    rare — some early ADAM events lack the static-file output entirely."""


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


_session = requests.Session()


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET an ADAM /items endpoint and unwrap the base64-encoded JSON body.
    ADAM specifically wraps OGC responses in base64 — most other JSON APIs
    don't, so we don't reuse a generic helper."""
    resp = _session.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return json.loads(base64.b64decode(resp.content))


# ---------------------------------------------------------------------------
# Name → ISO3
# ---------------------------------------------------------------------------


def name_to_iso3(name: str) -> Optional[str]:
    """Resolve an ADM0_NAME to ISO 3166-1 alpha-3. Checks the override dict
    first; falls back to pycountry's fuzzy search. Returns None when neither
    resolves (caller stores iso3 as null and downstream can flag)."""
    if not name:
        return None
    if name in ADAM_ISO3_OVERRIDES:
        return ADAM_ISO3_OVERRIDES[name]
    try:
        return pycountry.countries.search_fuzzy(name)[0].alpha_3
    except LookupError:
        return None


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


def _event_feature_to_row(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten an ADAM /items feature to a row dict.

    Required fields (event_id, episode_id, uid, name, source, from_date,
    to_date, alert_level) use direct subscripts — KeyError if ADAM drops
    them, surfacing a contract change loudly rather than silently coercing
    to None. population_csv_url is optional (legitimately missing on some
    legacy events).
    """
    p = feature["properties"]
    return {
        "event_id": int(p["event_id"]),
        "episode_id": int(p["episode_id"]),
        "uid": p["uid"],
        "name": p["name"],
        "source": p["source"],
        "from_date": p["from_date"],
        "to_date": p["to_date"],
        "alert_level": p["alert_level"],
        "population_csv_url": p.get("population_csv_url"),
    }


def _latest_episode_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe to one row per event_id, keeping the row with the highest
    episode_id. ADAM episode_ids are monotone-increasing per event, so
    max(episode_id) is the latest cumulative snapshot."""
    return (
        df.sort_values("episode_id", ascending=False)
        .drop_duplicates(subset="event_id", keep="first")
        .reset_index(drop=True)
    )


def get_events(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    source: Optional[Literal["NOAA", "JTWC"]] = "NOAA",
    all_episodes: bool = False,
) -> pd.DataFrame:
    """Fetch the ADAM TC event list.

    ADAM's ``/items`` endpoint returns one feature per *(event, episode)*.
    By default we dedupe to the latest episode per ``event_id`` (cumulative
    snapshot at storm end). Pass ``all_episodes=True`` to skip the dedupe
    and return every episode-feature — the per-episode ``to_date`` is the
    snapshot's effective time and each row carries its own
    ``population_csv_url``.

    Date filtering happens client-side after pagination (the server-side
    OGC filter syntax is awkward for date ranges); ``source`` is a
    server-side param.

    Parameters
    ----------
    from_date, to_date : str, optional
        ISO-format strings bounding an inclusive **overlap** window — an
        event is kept if it was active at any point in [from_date, to_date]
        (i.e., its ``to_date`` >= window start AND its ``from_date`` <=
        window end). Pass None on either side to leave that bound open.
        Strict containment would silently drop storms that started before
        the window opened.
    source : "NOAA" | "JTWC" | None, default "NOAA"
        Server-side source filter. ADAM aggregates advisory feeds from
        multiple agencies; NOAA covers Atlantic + EPac, JTWC covers
        WPac + IO + SHem. Pass None to fetch all sources.
    all_episodes : bool, default False
        When True, return every (event_id, episode_id) row rather than
        deduping to the latest episode. Increases row count by ~N×
        (N = avg episodes per event, ~10–50).

    Returns
    -------
    pandas.DataFrame
        One row per event_id (default) or per (event_id, episode_id)
        (``all_episodes=True``), columns per EVENT_SCHEMA.
    """
    all_items: List[Dict[str, Any]] = []
    offset = 0
    params: Dict[str, Any] = {"limit": _PAGE_SIZE}
    if source is not None:
        params["source"] = source

    while True:
        data = _get_json(
            f"{ADAM_BASE}/collections/{COLLECTION}/items",
            params={**params, "offset": offset},
        )
        features = data.get("features", [])
        if not features:
            break
        for f in features:
            all_items.append(_event_feature_to_row(f))
        logger.info(
            "ADAM page offset=%d: %d features (total: %d)",
            offset, len(features), len(all_items),
        )
        if len(features) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    if not all_items:
        return EVENT_SCHEMA.validate(pd.DataFrame(columns=_EVENT_COLS))

    df = pd.DataFrame(all_items)
    if not all_episodes:
        df = _latest_episode_per_event(df)

    # Client-side date filter (ADAM's OGC date-range filter is unreliable).
    # An event "overlaps" the window if it ended on/after window start AND
    # started on/before window end. Strict containment (both endpoints
    # inside the window) would silently drop storms that started just before
    # the window — e.g. KIRK-24 (from_date 2024-09-29) would be filtered out
    # of a 2024-10-01..15 query despite being active for most of the window.
    if from_date is not None:
        df = df[pd.to_datetime(df["to_date"]) >= pd.to_datetime(from_date)]
    if to_date is not None:
        df = df[pd.to_datetime(df["from_date"]) <= pd.to_datetime(to_date)]

    return EVENT_SCHEMA.validate(df.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Cumulative conversion
# ---------------------------------------------------------------------------


def make_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ADAM per-band pop counts to cumulative ≥-threshold.

    ADAM stores POP_60_KMH as "population in the 60-90 km/h band only" —
    *not* "population exposed to ≥ 60 km/h winds". GDACS exposure is
    cumulative. This converter brings ADAM into the same semantic so the two
    sources are directly comparable in downstream queries.

    Operates on per-band columns ``POP_60_KMH``, ``POP_90_KMH``, ``POP_120_KMH``
    (whichever subset is present). Returns a copy with the columns rewritten
    in place. Null/NaN values are preserved as null (not coerced to 0) for
    the column being updated; for the columns being summed *into* an update,
    NaN is treated as 0 (we can't add NaN to a count).
    """
    df = df.copy()
    v60 = df.get("POP_60_KMH")
    v90 = df.get("POP_90_KMH")
    v120 = df.get("POP_120_KMH")
    v90f = v90.fillna(0) if v90 is not None else 0
    v120f = v120.fillna(0) if v120 is not None else 0

    if v60 is not None:
        df["POP_60_KMH"] = v60.where(v60.isna(), v60.fillna(0) + v90f + v120f)
    if v90 is not None:
        df["POP_90_KMH"] = v90.where(v90.isna(), v90.fillna(0) + v120f)
    # POP_120_KMH is the top band, no addition needed.
    return df


# ---------------------------------------------------------------------------
# get_exposure
# ---------------------------------------------------------------------------


def _drop_total_rows(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Drop pre-aggregated subtotal rows the newer (2026+) CSVs embed.

    Those CSVs append rows marked ``<name> - TOT`` (a state subtotal and a
    national total) alongside the ADM2 leaf rows, with the finer admin
    levels left blank. Historical CSVs had no such rows — we always
    aggregated ADM2 leaves up to ADM1/ADM0 ourselves (see emit() below), so
    leaving these in would double-count them into our own totals. Expects
    columns already normalized to ADM0_NAME/ADM1_NAME/ADM2_NAME.
    """
    admin_cols = [
        c for c in ("ADM0_NAME", "ADM1_NAME", "ADM2_NAME") if c in csv_df.columns
    ]
    if not admin_cols:
        return csv_df
    is_total = pd.Series(False, index=csv_df.index)
    for c in admin_cols:
        is_total |= (
            csv_df[c].astype(str).str.strip().str.upper().str.endswith("- TOT")
        )
    return csv_df[~is_total]


def _csv_to_long(csv_df: pd.DataFrame, event_id: int) -> pd.DataFrame:
    """ADM2-granularity CSV → long-form rows for all three admin levels.

    Step order matters: cumulative conversion runs on the ADM2 rows
    *before* aggregation, because cumulative-then-sum and sum-then-cumulative
    yield the same per-band totals (addition commutes) but doing it first
    keeps the semantic invariant local to one function and avoids touching
    the aggregate rows again.
    """
    # Defensive: normalize columns (upper-case + strip), in case ADAM ever
    # publishes a CSV with whitespace or lowercase headers.
    csv_df = csv_df.copy()
    csv_df.columns = csv_df.columns.str.strip().str.upper()
    # Alias the newer LEVEL_n headers onto the canonical ADMn_NAME ones, but
    # only when the canonical column isn't already present — a CSV carrying
    # both would otherwise end up with duplicate ADM0_NAME columns and break
    # the r["ADM0_NAME"] lookups below.
    rename = {
        src: dst
        for src, dst in _ADMIN_COL_ALIASES.items()
        if src in csv_df.columns and dst not in csv_df.columns
    }
    if rename:
        csv_df = csv_df.rename(columns=rename)
    csv_df = _drop_total_rows(csv_df)
    csv_df = make_cumulative(csv_df)

    present_kmh = [c for c in _KMH_COL_TO_KT if c in csv_df.columns]
    if "ADM0_NAME" not in csv_df.columns or not present_kmh:
        raise NoExposureCSVError(
            f"event {event_id}: CSV missing ADM0_NAME or all POP_*_KMH columns "
            f"(columns present: {list(csv_df.columns)})"
        )

    rows: List[Dict[str, Any]] = []

    def emit(level: int, group_cols, parent_col: Optional[str]):
        """Group by the given columns, sum the kt buckets, emit one row per
        (group × wind threshold)."""
        grouped = (
            csv_df.groupby(group_cols, dropna=False)[present_kmh]
            .sum()
            .reset_index()
        )
        for _, r in grouped.iterrows():
            adm0 = r["ADM0_NAME"]
            admin_name = r[group_cols[-1]]
            parent = r[parent_col] if parent_col else None
            iso3 = name_to_iso3(str(adm0)) if adm0 else None
            for kmh_col, kt in _KMH_COL_TO_KT.items():
                if kmh_col not in present_kmh:
                    continue
                pop = r[kmh_col]
                rows.append({
                    "admin_level": level,
                    "iso3": iso3,
                    "admin_name": str(admin_name),
                    "parent_admin_name": str(parent) if parent else None,
                    "wind_speed_kt": kt,
                    "pop_exposed": None if pd.isna(pop) else int(pop),
                })

    # ADM0: group by country only
    emit(0, ["ADM0_NAME"], parent_col=None)
    # ADM1: group by country + state; parent is country
    if "ADM1_NAME" in csv_df.columns:
        emit(1, ["ADM0_NAME", "ADM1_NAME"], parent_col="ADM0_NAME")
    # ADM2: native granularity; parent is state
    if "ADM2_NAME" in csv_df.columns and "ADM1_NAME" in csv_df.columns:
        emit(2, ["ADM0_NAME", "ADM1_NAME", "ADM2_NAME"], parent_col="ADM1_NAME")

    return pd.DataFrame(rows, columns=_EXPOSURE_COLS)


def get_exposure(event_id: int, population_csv_url: str) -> pd.DataFrame:
    """Fetch and shape one ADAM event's exposure data.

    Downloads the per-episode population CSV at the given URL, applies the
    cumulative per-band → ≥-threshold conversion, aggregates ADM2 →
    ADM0/ADM1 (ADM2 also retained), maps ADM0_NAME → ISO3.

    Parameters
    ----------
    event_id : int
        Used only for error messages (the CSV doesn't carry it internally).
    population_csv_url : str
        Per-episode CSV URL from ``get_events()`` output. Must be non-empty;
        callers receiving null from ADAM should not invoke this function
        (raise NoExposureCSVError instead).

    Returns
    -------
    pandas.DataFrame
        Long-form, one row per (admin_level × admin_unit × wind_speed_kt).
        Columns per EXPOSURE_SCHEMA. Caller adds event_id / episode_id /
        valid_time before persisting.

    Raises
    ------
    NoExposureCSVError
        URL is empty/None, or the downloaded CSV is missing the expected
        columns. Caller treats this as "skip this event for this run; retry
        next cycle" the same way the GDACS pipeline treats NoTimelineError.
    """
    if not population_csv_url:
        raise NoExposureCSVError(
            f"event {event_id}: no population_csv_url on latest episode"
        )

    resp = _session.get(population_csv_url, timeout=_TIMEOUT)
    resp.raise_for_status()
    csv_df = pd.read_csv(StringIO(resp.text))

    long_df = _csv_to_long(csv_df, event_id)
    return EXPOSURE_SCHEMA.validate(long_df)
