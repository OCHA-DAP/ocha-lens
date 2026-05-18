"""Unit tests for the GDACS↔NHC ATCF matching function in
ocha_lens.datasources.gdacs.

The matcher uses the GDACS event's genesis advisory (smallest
``advisory_number``) and requires an EXACT ``valid_time`` match in
``nhc_tracks``. A-deck OFCL and the TCM advisories GDACS scrapes
are NHC's same forecaster output in different formats — so the
position at a shared valid_time agrees byte-for-byte modulo rounding.

Default spatial tolerance is rounding-only (0.05°). Larger deltas
are either NHC operational corrections or wrong-storm overlap, and
the matcher returns ``None`` rather than accept them.

Multi-basin storms are handled implicitly: the second basin's
``atcf_id`` doesn't yet have a row at the genesis valid_time.
"""

import pandas as pd

from ocha_lens.datasources.gdacs import match_to_atcf


def _gdacs_row(advisory_number, time, lat, lon, actual=True):
    return {
        "advisory_number": advisory_number,
        "actual": "true" if actual else "false",
        "advisory_datetime": pd.Timestamp(time),
        "latitude": lat,
        "longitude": lon,
    }


def _nhc_row(atcf_id, valid_time, lat, lon):
    """One NHC track row (any leadtime). Caller pre-dedups to freshest
    issuance per (atcf_id, valid_time)."""
    return {
        "atcf_id": atcf_id,
        "valid_time": pd.Timestamp(valid_time),
        "lat": lat,
        "lon": lon,
    }


def test_match_exact_at_genesis():
    """Single atcf_id has a row at the exact genesis valid_time and
    is spatially identical → match it."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),
    ])
    # A-deck OFCL TAU=3 forecast from 18Z issuance, valid 21Z
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.0, -77.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_match_tolerates_rounding_artifact():
    """Sub-rounding-tolerance deviations (≤0.05°) are accepted as
    rounding noise between A-deck and TCM formats."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.02, -77.03),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_match_rejects_operational_correction():
    """When NHC operationally updated t=0 between A-deck and TCM,
    positions disagree by more than rounding. The matcher refuses
    to silently accept that deviation — better to return None than
    guess."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),
    ])
    # Off by 0.2° — beyond rounding, indicates real NHC correction
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.2, -77.0),
    ])
    assert match_to_atcf(gdacs, nhc) is None


def test_match_picks_genesis_basin_for_basin_crosser():
    """OTTO-16 archetype: at the GDACS genesis valid_time, only the
    genesis-basin atcf_id has a row. The second basin's atcf_id was
    issued later; it has no row at this time and can't compete."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2016-11-21 09:00", 11.5, -79.4),
        _gdacs_row(20, "2016-11-25 06:00", 10.6, -86.3),
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL162016", "2016-11-21 09:00", 11.5, -79.4),
        _nhc_row("AL162016", "2016-11-25 06:00", 10.6, -86.3),
        # EP22 only exists later — no row at the genesis valid_time
        _nhc_row("EP222016", "2016-11-25 06:00", 10.6, -86.3),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL162016"


def test_match_disambiguates_concurrent_storms_by_position():
    """Two storms have rows at the same genesis valid_time. Pick the
    spatially-exact one; reject the far one."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.0, -77.0),     # exact
        _nhc_row("AL152024", "2024-10-09 21:00", 30.0, -50.0),     # 30° away
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_match_returns_none_when_no_valid_time_overlap():
    """No NHC row shares the genesis valid_time. The closest row is
    3h earlier (a synoptic time) but the match is exact-valid_time
    only — returns None and accepts the DB gap."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-19 15:00", 21.3, -70.2),
    ])
    # NHC has leadtime=0 rows at the prior and next synoptic, but
    # no TAU=3 forecast at the 15:00 TCM cycle (gap)
    nhc = pd.DataFrame([
        _nhc_row("AL162024", "2024-10-19 12:00", 21.3, -70.4),
        _nhc_row("AL162024", "2024-10-19 18:00", 21.4, -70.6),
    ])
    assert match_to_atcf(gdacs, nhc) is None


def test_genesis_picked_by_min_advisory_number():
    """Input order doesn't matter — the matcher sorts on
    advisory_number to identify the genesis row."""
    # Shuffled — latest first, genesis last
    gdacs = pd.DataFrame([
        _gdacs_row(3, "2024-10-10 09:00", 20.0, -80.0),
        _gdacs_row(2, "2024-10-10 03:00", 19.0, -78.5),
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.0, -77.0),
        _nhc_row("AL992024", "2024-10-10 03:00", 19.0, -78.5),
        _nhc_row("AL992024", "2024-10-10 09:00", 20.0, -80.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_empty_timeline_returns_none():
    empty_tl = pd.DataFrame(
        columns=["advisory_number", "actual",
                 "advisory_datetime", "latitude", "longitude"]
    )
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.5, -77.5),
    ])
    assert match_to_atcf(empty_tl, nhc) is None


def test_empty_nhc_returns_none():
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),
    ])
    empty_nhc = pd.DataFrame(
        columns=["atcf_id", "valid_time", "lat", "lon"]
    )
    assert match_to_atcf(gdacs, empty_nhc) is None
