"""Unit tests for the GDACS↔NHC ATCF matching function in
ocha_lens.datasources.gdacs.

The matcher is two-strategy:

1. Forecast cone (primary) — votes the GDACS forecast points
   (``actual="false"``) onto NHC ``atcf_id``s by exact valid_time.
   GDACS (TCM) and our NHC table (A-deck) carry the *same* forecast
   cone, so the correct storm agrees to ~0°; the most-voted
   ``atcf_id`` wins, ties go to the earliest valid_time.
2. Genesis (fallback) — single-point match on the genesis observed
   advisory, used only when the timeline has no forecast cone.

Default spatial tolerance is rounding-only (0.05°); robustness comes
from voting across the cone, not a loose tolerance. Wrong-storm
overlap and NHC operational corrections fall outside it and yield
``None``.
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


def _gdacs_forecast(advisory_number, time, lat, lon):
    """A forecast-cone row (``actual="false"``)."""
    return _gdacs_row(advisory_number, time, lat, lon, actual=False)


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


# --- Strategy 1: forecast-cone matching -----------------------------


def test_forecast_cone_matches_when_genesis_missing():
    """AMANDA-26 archetype: we never captured the storm's genesis
    advisory, so no NHC row exists at the genesis valid_time. The
    shared forecast cone still pins the atcf_id."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2026-06-02 21:00", 9.4, -126.7),  # genesis, unseen
        _gdacs_forecast(5, "2026-06-05 00:00", 12.0, -132.0),
        _gdacs_forecast(5, "2026-06-05 12:00", 12.5, -134.0),
        _gdacs_forecast(5, "2026-06-06 00:00", 13.0, -136.0),
    ])
    nhc = pd.DataFrame([
        # nothing at 2026-06-02 21:00 — the genesis cycle we missed
        _nhc_row("EP012026", "2026-06-05 00:00", 12.0, -132.0),
        _nhc_row("EP012026", "2026-06-05 12:00", 12.5, -134.0),
        _nhc_row("EP012026", "2026-06-06 00:00", 13.0, -136.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "EP012026"


def test_forecast_cone_picks_majority_storm():
    """Most forecast points agree with one atcf_id; a lone near-
    coincidence with a concurrent storm can't outvote it (and the
    nearest-per-point rule rejects it anyway)."""
    gdacs = pd.DataFrame([
        _gdacs_forecast(3, "2024-10-05 00:00", 20.0, -80.0),
        _gdacs_forecast(3, "2024-10-05 12:00", 21.0, -81.0),
        _gdacs_forecast(3, "2024-10-06 00:00", 22.0, -82.0),
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-05 00:00", 20.0, -80.0),
        _nhc_row("AL142024", "2024-10-05 12:00", 21.0, -81.0),
        _nhc_row("AL142024", "2024-10-06 00:00", 22.0, -82.0),
        # concurrent storm, only near the first point and still
        # farther than AL14's exact hit
        _nhc_row("AL152024", "2024-10-05 00:00", 20.04, -80.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"


def test_forecast_cone_tie_breaks_to_genesis_basin():
    """A basin-crossing storm's cone touches two atcf_ids equally;
    the earliest valid_time (genesis basin) wins the tie."""
    gdacs = pd.DataFrame([
        _gdacs_forecast(10, "2016-11-23 00:00", 12.0, -82.0),  # Atlantic
        _gdacs_forecast(10, "2016-11-25 00:00", 11.0, -88.0),  # E Pacific
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL162016", "2016-11-23 00:00", 12.0, -82.0),
        _nhc_row("EP222016", "2016-11-25 00:00", 11.0, -88.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL162016"


def test_falls_back_to_genesis_when_cone_has_no_overlap():
    """Forecast points exist but none land on an NHC row (e.g. our
    NHC horizon is shorter). The genesis observed advisory still
    resolves it."""
    gdacs = pd.DataFrame([
        _gdacs_row(1, "2024-10-09 21:00", 18.0, -77.0),       # genesis
        _gdacs_forecast(1, "2024-10-20 00:00", 40.0, -50.0),  # no NHC row
    ])
    nhc = pd.DataFrame([
        _nhc_row("AL142024", "2024-10-09 21:00", 18.0, -77.0),
    ])
    assert match_to_atcf(gdacs, nhc) == "AL142024"
