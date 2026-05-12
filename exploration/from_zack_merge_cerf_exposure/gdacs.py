# ===========================================================================
# Ported verbatim — DO NOT MODIFY (this is the refactor source-of-truth)
# Source repo:   OCHA-DAP/ds-storm-impact-harmonisation
# Source ref:    bf1938fe2ded32cc4371dda56269d4399a8de10c (merge-cerf-exposure HEAD)
# Source path:   src/datasets/gdacs.py
# Pulled:        2026-05-07
# Provenance:    referenced from ocha-lens#40 (https://github.com/OCHA-DAP/ocha-lens/issues/40)
# Refactored functions live in src/ocha_lens/datasources/ — see PR description.
# ===========================================================================

"""
GDACS Tropical Cyclone API client.

Fetches active and historical tropical cyclone events, advisory timelines
with per-advisory population exposure, and country-level impact data from
the GDACS REST API. No authentication required.

API docs: https://www.gdacs.org/gdacsapi/swagger/index.html

Key endpoints
-------------
- Timeline (`/api/export/gettimeline`): per-advisory snapshots with
  pop39/pop74, wind radii, position. Available 2015+.
- Impact (`/api/export/getimpact`): country-level cumulative exposure
  at buffer39/buffer74. Only returns data ~2022+.

See artefacts/01_merge_cerf_exposure/gdacs_endpoint_comparison.md for
a detailed comparison of these two endpoints.
"""

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon

BASE_URL = "https://www.gdacs.org/gdacsapi/api"

IBTRACS_LAST3_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive"
    "-for-climate-stewardship-ibtracs/v04r01/access/csv/"
    "ibtracs.last3years.list.v04r01.csv"
)

# -- Event listing --------------------------------------------------------


def get_active_cyclones(
    alert_levels: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch tropical cyclone events from GDACS.

    Parameters
    ----------
    alert_levels : list of {"green", "orange", "red"}, optional
        Filter by alert level. Default: all levels.
    from_date, to_date : str "YYYY-MM-DD", optional
        Date range filter. Default: GDACS default (recent events).
    limit : int
        Max results (GDACS caps at 100 per page).

    Returns
    -------
    pd.DataFrame with one row per event.
    """
    params = {"eventlist": "TC", "limit": limit}
    if alert_levels:
        params["alertlevel"] = ";".join(alert_levels)
    if from_date:
        params["fromdate"] = from_date
    if to_date:
        params["todate"] = to_date

    r = requests.get(f"{BASE_URL}/events/geteventlist/SEARCH", params=params)
    r.raise_for_status()
    features = r.json().get("features", [])

    rows = []
    for f in features:
        p = f["properties"]
        rows.append(
            {
                "eventid": p["eventid"],
                "name": p["eventname"],
                "alert_level": p["alertlevel"],
                "episode_alert": p["episodealertlevel"],
                "country": p["country"],
                "iso3": p["iso3"],
                "from_date": p["fromdate"],
                "to_date": p["todate"],
                "is_current": p.get("iscurrent") == "true",
                "severity_kmh": p.get("severitydata", {}).get("severity"),
                "severity_text": p.get("severitydata", {}).get(
                    "severitytext"
                ),
                "lon": f["geometry"]["coordinates"][0],
                "lat": f["geometry"]["coordinates"][1],
            }
        )
    return pd.DataFrame(rows)


# -- Event detail ---------------------------------------------------------


def get_event_detail(eventid: int) -> dict:
    """Fetch full event detail including impact URLs and episode list."""
    r = requests.get(
        f"{BASE_URL}/events/geteventdata",
        params={"eventtype": "TC", "eventid": eventid},
    )
    r.raise_for_status()
    return r.json()


def get_episode_detail(eventid: int, episodeid: int) -> dict:
    """Fetch detail for a specific episode of a TC event.

    Each episode represents one model run (issuance), typically
    produced every 6 hours at each new advisory. The returned
    structure mirrors ``get_event_detail()``, but resource URLs
    (impact, timeline, locations) point to episode-specific data.
    """
    r = requests.get(
        f"{BASE_URL}/events/getepisodedata",
        params={
            "eventtype": "TC",
            "eventid": eventid,
            "episodeid": episodeid,
        },
    )
    r.raise_for_status()
    return r.json()


# -- Timeline (per-advisory exposure) ------------------------------------


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


def get_timeline(eventid: int) -> pd.DataFrame:
    """Fetch the advisory timeline for a TC event.

    Each row is one advisory with position, wind speed, population
    exposure (pop39/pop74), and quadrant wind radii.

    pop39/pop74 are instantaneous snapshots: how many people are
    inside the wind polygon at that advisory. Not cumulative.
    """
    detail = get_event_detail(eventid)
    props = detail.get("properties", {})
    impacts = props.get("impacts", [])

    timeline_url = None
    for imp in impacts:
        resource = imp.get("resource", {})
        if "timeline" in resource:
            timeline_url = resource["timeline"]
            break

    if timeline_url is None:
        raise ValueError(f"No timeline found for event {eventid}")

    r = requests.get(timeline_url)
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

    df = pd.DataFrame(rows)
    for col in _TIMELINE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["advisory_datetime"] = pd.to_datetime(df["advisory_datetime"])
    df["advisory_number"] = pd.to_numeric(
        df["advisory_number"], errors="coerce"
    )

    return df.sort_values("advisory_number").reset_index(drop=True)


# -- Impact (country-level cumulative exposure) ---------------------------


def get_impact_by_country(
    eventid: int, aggregate: bool = True
) -> dict[str, pd.DataFrame]:
    """Fetch population exposure for each wind buffer.

    Returns country-level (or admin-level) population exposed within
    the cumulative wind footprint across the full storm track.

    Note: only returns data for events from ~2022 onward. Earlier
    events return zeros. Use get_timeline() for 2015+ coverage.

    The API labels these as buffer39 (39 kt) and buffer74 (74 kt).
    The original CSV mislabeled them as pop_34kt/pop_64kt.

    Parameters
    ----------
    eventid : int
    aggregate : bool
        If True (default), sum to country level.
        If False, return per admin unit.
    """
    detail = get_event_detail(eventid)
    props = detail.get("properties", {})
    impacts = props.get("impacts", [])

    result = {}
    for imp in impacts:
        resource = imp.get("resource", {})
        for key in ["buffer39", "buffer74"]:
            if key not in resource:
                continue
            r = requests.get(resource[key])
            r.raise_for_status()
            data = r.json()
            datums = data.get("datums", [])

            countries = []
            for datum_group in datums:
                if datum_group.get("alias") != "alert":
                    continue
                for d in datum_group.get("datum", []):
                    scalars = d.get("scalars", {}).get("scalar", [])
                    row = {}
                    for s in scalars:
                        row[s["name"]] = s["value"]
                    if "GMI_CNTRY" in row:
                        countries.append(
                            {
                                "iso3": row.get("GMI_CNTRY"),
                                "country": row.get("CNTRY_NAME"),
                                "pop_admin": int(
                                    float(row.get("POP_ADMIN", 0))
                                ),
                                "pop_affected": int(
                                    float(row.get("POP_AFFECTED", 0))
                                ),
                                "distance_km": round(
                                    float(row.get("distance", 0)), 1
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


# -- IBTrACS matching -----------------------------------------------------

_UNNAMED_STORMS = {
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX",
    "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN",
    "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
}


def load_ibtracs_lookup(
    url: str = IBTRACS_LAST3_URL,
) -> pd.DataFrame:
    """Download IBTrACS and return a lookup table for matching.

    Returns DataFrame with one row per storm: SID, NAME, SEASON, BASIN.
    """
    df = pd.read_csv(url, skiprows=[1], low_memory=False)
    lookup = (
        df.groupby("SID")
        .first()[["NAME", "SEASON", "BASIN"]]
        .reset_index()
    )
    lookup["NAME"] = lookup["NAME"].str.upper().str.strip()
    lookup["SEASON"] = lookup["SEASON"].astype(int)
    return lookup


def match_gdacs_to_ibtracs(
    events: pd.DataFrame,
    ibtracs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match GDACS events to IBTrACS SIDs by name + season.

    Parameters
    ----------
    events : DataFrame from get_active_cyclones()
    ibtracs : DataFrame from load_ibtracs_lookup(), or None to download.

    Returns
    -------
    events DataFrame with added 'sid' column (None if no match).
    """
    if ibtracs is None:
        ibtracs = load_ibtracs_lookup()

    result = events.copy()
    sids = []

    for _, ev in result.iterrows():
        parts = ev["name"].rsplit("-", 1)
        name = parts[0].upper()
        try:
            yr_suffix = int(parts[1])
            year = 2000 + yr_suffix if yr_suffix < 100 else yr_suffix
        except (IndexError, ValueError):
            year = pd.to_datetime(ev["from_date"]).year

        base_name = name.split("-")[0] if "-" in ev["name"] else name
        if base_name in _UNNAMED_STORMS:
            sids.append(None)
            continue

        matches = ibtracs[
            (ibtracs["NAME"] == name) & (ibtracs["SEASON"] == year)
        ]

        if len(matches) == 1:
            sids.append(matches.iloc[0]["SID"])
        elif len(matches) > 1:
            sids.append(matches.iloc[0]["SID"])
        else:
            sids.append(None)

    result["sid"] = sids
    return result


# -- NHC matching ---------------------------------------------------------
#
# GDACS scrapes the NHC Forecast/Advisory (TCM), while our DB has the
# ATCF A-deck. Both originate from the same NHC forecast cycle. The TCM
# updates the current position (t=0) to the advisory issue time (+3h),
# but all forecast points (t>=1) are identical to the A-deck.
#
# Therefore: GDACS forecast positions match A-deck forecast positions
# at the same valid_time with 0.000 degree error. No spatial tolerance
# needed. See book chapter 05-gdacs-nhc-matching for full analysis.


def match_gdacs_to_nhc(
    timeline: pd.DataFrame,
    engine,
    max_dist_deg: float = 0.5,
) -> str | None:
    """Match a GDACS timeline to an NHC atcf_id.

    Uses the fact that GDACS (TCM) and A-deck share identical forecast
    positions. Tries two strategies:

    1. Exact valid_time match on GDACS forecast points against A-deck
       forecast rows (leadtime > 0). These should match at 0.000 deg.
    2. Fallback: spatial match of first GDACS actual advisory against
       A-deck leadtime=3 rows (which equal TCM t=0 positions).

    Parameters
    ----------
    timeline : DataFrame from get_timeline()
    engine : SQLAlchemy engine from stratus.get_engine()
    max_dist_deg : float
        Maximum distance for spatial verification (default 0.5 deg).

    Returns
    -------
    atcf_id (e.g., "AL142024") or None if no match.
    """
    actual = timeline[
        timeline["actual"].astype(str).str.lower() == "true"
    ]
    forecast = timeline[
        timeline["actual"].astype(str).str.lower() != "true"
    ]

    with engine.connect() as conn:
        # Strategy 1: exact match on forecast valid_times
        if len(forecast) > 0:
            for _, fc in forecast.iterrows():
                vt = fc["advisory_datetime"]
                match = pd.read_sql(
                    """
                    SELECT atcf_id,
                           ST_Y(geometry) as lat,
                           ST_X(geometry) as lon
                    FROM storms.nhc_tracks_geo
                    WHERE valid_time = %s AND leadtime > 0
                    """,
                    conn,
                    params=(vt,),
                )
                if len(match) == 0:
                    continue

                dists = np.sqrt(
                    (match["lat"] - fc["latitude"]) ** 2
                    + (match["lon"] - fc["longitude"]) ** 2
                )
                best = dists.idxmin()
                if dists[best] < max_dist_deg:
                    return match.loc[best, "atcf_id"]

        # Strategy 2: match first actual advisory via leadtime=3
        if len(actual) > 0:
            first = actual.iloc[0]
            match = pd.read_sql(
                """
                SELECT atcf_id,
                       ST_Distance(
                           geometry,
                           ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                       ) as dist
                FROM storms.nhc_tracks_geo
                WHERE leadtime > 0
                AND valid_time BETWEEN %s AND %s
                ORDER BY dist
                LIMIT 1
                """,
                conn,
                params=(
                    float(first["longitude"]),
                    float(first["latitude"]),
                    first["advisory_datetime"] - pd.Timedelta(days=1),
                    first["advisory_datetime"] + pd.Timedelta(days=1),
                ),
            )
            if len(match) > 0 and match.iloc[0]["dist"] < max_dist_deg:
                return match.iloc[0]["atcf_id"]

    return None


def match_gdacs_events_to_nhc(
    events: pd.DataFrame,
    engine,
) -> pd.DataFrame:
    """Match multiple GDACS events to NHC atcf_ids.

    Parameters
    ----------
    events : DataFrame from get_active_cyclones()
    engine : SQLAlchemy engine from stratus.get_engine()

    Returns
    -------
    events DataFrame with added 'atcf_id' column.
    """
    result = events.copy()
    atcf_ids = []

    for _, ev in result.iterrows():
        try:
            tl = get_timeline(ev["eventid"])
            atcf = match_gdacs_to_nhc(tl, engine)
            atcf_ids.append(atcf)
        except Exception:
            atcf_ids.append(None)

    result["atcf_id"] = atcf_ids
    return result


# -- Batch exposure table ------------------------------------------------


def build_exposure_table(
    from_date: str,
    to_date: str,
    alert_levels: list[str] | None = None,
    use_timeline: bool = True,
) -> pd.DataFrame:
    """Build an exposure table from GDACS for a date range.

    Parameters
    ----------
    from_date, to_date : str "YYYY-MM-DD"
    alert_levels : list, optional (default: all)
    use_timeline : bool
        True = timeline endpoint (per-advisory, 2015+, no country split).
        False = impact endpoint (country-level, ~2022+ only).
    """
    events = get_active_cyclones(
        alert_levels=alert_levels,
        from_date=from_date,
        to_date=to_date,
    )
    print(f"Found {len(events)} TC events in {from_date} to {to_date}")

    all_rows = []
    for _, ev in events.iterrows():
        eid = ev["eventid"]
        print(f"  {ev['name']} (id={eid})...", end=" ", flush=True)

        if use_timeline:
            try:
                tl = get_timeline(eid)
                tl["eventid"] = eid
                tl["event_name"] = ev["name"]
                tl["alert_level"] = ev["alert_level"]
                tl["event_from_date"] = ev["from_date"]
                tl["event_country"] = ev["country"]
                all_rows.append(tl)
                print(f"{len(tl)} advisories")
            except Exception as e:
                print(f"FAILED: {e}")
        else:
            try:
                impact = get_impact_by_country(eid, aggregate=True)
                for buf_name, df_buf in impact.items():
                    if len(df_buf) == 0:
                        continue
                    df_buf = df_buf.copy()
                    df_buf["eventid"] = eid
                    df_buf["event_name"] = ev["name"]
                    df_buf["buffer"] = buf_name
                    df_buf["alert_level"] = ev["alert_level"]
                    df_buf["event_from_date"] = ev["from_date"]
                    all_rows.append(df_buf)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# -- Wind radii polygon helper -------------------------------------------


def wind_radii_polygon(
    lat: float,
    lon: float,
    ne_nm: float,
    se_nm: float,
    sw_nm: float,
    nw_nm: float,
    n_points: int = 120,
    method: str = "asymmetric",
) -> Polygon | None:
    """Build a wind radii polygon from quadrant radii (nm).

    Parameters
    ----------
    lat, lon : float
        Storm center position.
    ne_nm, se_nm, sw_nm, nw_nm : float
        Wind radii in nautical miles per quadrant.
    n_points : int
        Number of points around the polygon.
    method : str
        "asymmetric" -- cosine-interpolated between quadrant values
        (physically realistic, preserves asymmetry).
        "symmetric" -- uses max(quadrant radii) as a uniform circle
        (replicates the GDACS/JRC methodology per their 2026 guide).

    Quadrant convention (standard TC advisory):
    - NE: 0-90 degrees (North to East)
    - SE: 90-180 degrees (East to South)
    - SW: 180-270 degrees (South to West)
    - NW: 270-360 degrees (West to North)
    """
    if ne_nm + se_nm + sw_nm + nw_nm == 0:
        return None

    cos_lat = np.cos(np.radians(lat))

    if method == "symmetric":
        r_deg = max(ne_nm, se_nm, sw_nm, nw_nm) / 60.0
        coords = []
        for az in np.linspace(0, 360, n_points, endpoint=False):
            az_rad = np.radians(az)
            dlat = r_deg * np.cos(az_rad)
            dlon = (
                r_deg * np.sin(az_rad) / cos_lat if cos_lat > 0 else 0
            )
            coords.append((lon + dlon, lat + dlat))
        coords.append(coords[0])
        return Polygon(coords)

    # Asymmetric: cosine-interpolated between quadrant centers
    quad_centers = [45, 135, 225, 315]
    quad_radii = [
        ne_nm / 60.0,
        se_nm / 60.0,
        sw_nm / 60.0,
        nw_nm / 60.0,
    ]

    coords = []
    for az in np.linspace(0, 360, n_points, endpoint=False):
        r_deg = _interp_radius(az, quad_centers, quad_radii)
        if r_deg == 0:
            coords.append((lon, lat))
            continue
        az_rad = np.radians(az)
        dlat = r_deg * np.cos(az_rad)
        dlon = r_deg * np.sin(az_rad) / cos_lat if cos_lat > 0 else 0
        coords.append((lon + dlon, lat + dlat))

    coords.append(coords[0])
    return Polygon(coords)


def _interp_radius(
    azimuth: float,
    centers: list[float],
    radii: list[float],
) -> float:
    """Cosine-interpolate radius between quadrant centers.

    Each quadrant center (45, 135, 225, 315) has a defined radius.
    Between centers, the radius transitions smoothly using cosine
    interpolation, avoiding sharp kinks at quadrant boundaries.
    """
    az = azimuth % 360
    n = len(centers)

    # Find the two bracketing centers
    for i in range(n):
        c0 = centers[i]
        c1 = centers[(i + 1) % n]

        # Handle wrap-around (315 -> 45)
        if c1 < c0:
            if az >= c0 or az < c1:
                span = (c1 + 360 - c0)
                t = ((az if az >= c0 else az + 360) - c0) / span
                # Cosine interpolation for smooth transition
                w = 0.5 * (1 - np.cos(np.pi * t))
                return radii[i] * (1 - w) + radii[(i + 1) % n] * w
        else:
            if c0 <= az < c1:
                t = (az - c0) / (c1 - c0)
                w = 0.5 * (1 - np.cos(np.pi * t))
                return radii[i] * (1 - w) + radii[(i + 1) % n] * w

    return radii[0]
