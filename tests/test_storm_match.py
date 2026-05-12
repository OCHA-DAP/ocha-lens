"""Unit tests for the GDACS↔NHC ATCF matching function in
ocha_lens.utils.storm.

The matching algorithm is a single inner-join on ``valid_time``
(treating GDACS's ``advisory_datetime`` as a valid_time — that field
is overloaded by GDACS to be the issued time on ``actual=True``
rows and the forecast valid_time on ``actual=False`` rows), then a
per-``atcf_id`` mean-distance aggregate. Smallest mean within
``max_avg_dist_deg`` wins.

The tests cover the behaviors that matter in practice:

- A-deck-mode integration: ``leadtime=0`` rows naturally drop because
  A-deck synoptic times don't collide with GDACS issue times; forecast
  rows match at ~0 deg.
- TCM-mode integration: ``leadtime=0`` rows DO match (both at issue
  time) — providing extra zero-distance signal.
- Multi-storm disambiguation via per-storm mean.
- Rejection when no candidate averages within tolerance (e.g.,
  non-NHC basin storms).
"""

import pandas as pd

from ocha_lens.datasources.gdacs import match_to_atcf


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _gdacs_row(time, lat, lon, actual=True):
    """One GDACS timeline row.

    GDACS overloads ``advisory_datetime``: it's the issued time on
    actual rows and the valid_time of the forecast point on
    forecast rows. The match function treats it as valid_time
    uniformly.
    """
    return {
        "actual": "true" if actual else "false",
        "advisory_datetime": pd.Timestamp(time),
        "latitude": lat,
        "longitude": lon,
    }


def _nhc_row(atcf_id, valid_time, leadtime, lat, lon):
    return {
        "atcf_id": atcf_id,
        "valid_time": pd.Timestamp(valid_time),
        "leadtime": leadtime,
        "lat": lat,
        "lon": lon,
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_match_adeck_mode_via_forecast_points():
    """A-deck mode: GDACS issue-time advisories don't match A-deck
    leadtime=0 rows (synoptic time != issue time), but GDACS forecast
    rows match A-deck forecast rows at the same valid_time at ~0 deg.
    """
    gdacs = pd.DataFrame([
        # actual row at issue time 21Z — A-deck LT0 is at synoptic 18Z,
        # so this won't join against any A-deck LT0 row
        _gdacs_row("2024-10-09 21:00", 18.0, -77.0, actual=True),
        # forecast points at future valid_times — these will match
        # A-deck forecast rows at the same valid_time
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
        _gdacs_row("2024-10-10 09:00", 20.0, -80.0, actual=False),
        _gdacs_row("2024-10-10 15:00", 21.0, -81.5, actual=False),
    ])
    # A-deck rows: LT0 at synoptic (18Z) won't match, forecasts will
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 18:00", 0, 17.5, -76.5),
        _nhc_row("AL142024", "2024-10-10 03:00", 9, 19.0, -78.5),
        _nhc_row("AL142024", "2024-10-10 09:00", 15, 20.0, -80.0),
        _nhc_row("AL142024", "2024-10-10 15:00", 21, 21.0, -81.5),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_match_tcm_mode_includes_t0():
    """TCM mode: GDACS actual rows DO match TCM leadtime=0 rows (both
    timestamped at issue time). This adds another ~0 deg signal on
    top of the forecast matches, but the algorithm doesn't need to
    know it's in TCM mode — it just happens because of the join.
    """
    gdacs = pd.DataFrame([
        _gdacs_row("2024-10-09 21:00", 18.0, -77.0, actual=True),
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
    ])
    # TCM rows: LT0 valid_time = issue time = matches GDACS actual
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 0, 18.0, -77.0),
        _nhc_row("AL142024", "2024-10-10 03:00", 6, 19.0, -78.5),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


# ---------------------------------------------------------------------------
# Multi-storm disambiguation
# ---------------------------------------------------------------------------


def test_match_disambiguates_two_storms_at_same_valid_time():
    """Two NHC storms have forecast rows at the same valid_time.
    Aggregate mean-distance picks the right one. The user's specific
    concern — same-issue-time multi-storm collision.
    """
    gdacs = pd.DataFrame([
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
        _gdacs_row("2024-10-10 09:00", 20.0, -80.0, actual=False),
        _gdacs_row("2024-10-10 15:00", 21.0, -81.5, actual=False),
    ])
    # AL14: matches GDACS at 0 deg. AL15: at the same valid_times but
    # in a completely different basin (~50 deg away)
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-10 03:00", 9, 19.0, -78.5),
        _nhc_row("AL142024", "2024-10-10 09:00", 15, 20.0, -80.0),
        _nhc_row("AL142024", "2024-10-10 15:00", 21, 21.0, -81.5),
        _nhc_row("AL152024", "2024-10-10 03:00", 9, 30.0, -50.0),
        _nhc_row("AL152024", "2024-10-10 09:00", 15, 30.5, -49.0),
        _nhc_row("AL152024", "2024-10-10 15:00", 21, 31.0, -48.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_match_disambiguates_when_close_storms_partial_overlap():
    """Two nearby storms with overlapping valid_times — the algorithm
    averages over ALL matching rows, so a storm with 3 zero-distance
    matches beats one with 1 zero-distance match + 2 several-deg
    matches. Robustness check for the averaging behavior.
    """
    gdacs = pd.DataFrame([
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
        _gdacs_row("2024-10-10 09:00", 20.0, -80.0, actual=False),
        _gdacs_row("2024-10-10 15:00", 21.0, -81.5, actual=False),
    ])
    nhc = pd.DataFrame([
        # AL14 matches all 3 perfectly → mean = 0.000
        _nhc_row("AL142024", "2024-10-10 03:00", 9, 19.0, -78.5),
        _nhc_row("AL142024", "2024-10-10 09:00", 15, 20.0, -80.0),
        _nhc_row("AL142024", "2024-10-10 15:00", 21, 21.0, -81.5),
        # AL15 matches 1 perfectly, 2 are off by ~3 deg → mean ~2.0
        _nhc_row("AL152024", "2024-10-10 03:00", 9, 19.0, -78.5),
        _nhc_row("AL152024", "2024-10-10 09:00", 15, 23.0, -80.0),
        _nhc_row("AL152024", "2024-10-10 15:00", 21, 24.0, -81.5),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_match_rejects_when_winner_above_tolerance():
    """Even the best candidate has a large mean distance → no NHC
    counterpart (e.g., JTWC/RSMC basin storm). Returns None.
    """
    gdacs = pd.DataFrame([
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
        _gdacs_row("2024-10-10 09:00", 20.0, -80.0, actual=False),
    ])
    # Only candidate is on the other side of the planet — joins on
    # valid_time but distances are huge
    nhc = pd.DataFrame([
        _nhc_row("WP012024", "2024-10-10 03:00", 9, 15.0, 130.0),
        _nhc_row("WP012024", "2024-10-10 09:00", 15, 16.0, 131.0),
    ])
    assert match_to_atcf(gdacs, nhc) is None


def test_match_returns_none_when_no_valid_times_overlap():
    """GDACS rows and NHC rows have completely disjoint valid_times
    (e.g., different season). Empty join → None.
    """
    gdacs = pd.DataFrame([
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2020-08-15 12:00", 9, 19.0, -78.5),
    ])
    assert match_to_atcf(gdacs, nhc) is None


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_empty_timeline_returns_none():
    empty_tl = pd.DataFrame(
        columns=["actual", "advisory_datetime", "latitude", "longitude"]
    )
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 9, 18.5, -77.5),
    ])
    assert match_to_atcf(empty_tl, nhc) is None


def test_empty_nhc_returns_none():
    gdacs = pd.DataFrame([
        _gdacs_row("2024-10-09 21:00", 18.0, -77.0, actual=True),
        _gdacs_row("2024-10-10 03:00", 19.0, -78.5, actual=False),
    ])
    empty_nhc = pd.DataFrame(
        columns=["atcf_id", "valid_time", "leadtime", "lat", "lon"]
    )
    assert match_to_atcf(gdacs, empty_nhc) is None
