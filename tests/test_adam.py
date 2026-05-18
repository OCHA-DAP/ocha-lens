"""Unit tests for ocha_lens.datasources.adam.

Coverage focuses on the behaviors that are easy to silently break:

  - The per-band → cumulative-≥-threshold conversion (cross-source
    semantic parity with GDACS; wrong conversion gives wrong numbers
    but doesn't crash).
  - The name → ISO3 mapping (override path vs. pycountry fallback vs.
    unresolvable).
  - get_events pagination + dedup-to-latest-episode + overlap-style
    date filter (the date filter was originally written as strict
    containment, which silently dropped storms that started before
    the window — these tests pin the overlap semantics).
  - get_exposure raises loudly when the source CSV is missing or
    malformed (NoExposureCSVError), and produces the correct row
    counts + iso3 attribution + aggregation invariant
    (sum(adm1) == adm0).

HTTP is mocked at two seams:
  - adam._get_json  : the OGC /items endpoint (base64-wrapped JSON)
  - adam._session.get : the per-event CSV downloads
"""

import base64
import json
from typing import Optional

import pandas as pd
import pytest
import requests

from ocha_lens.datasources import adam


# ---------------------------------------------------------------------------
# Mocking helpers
# ---------------------------------------------------------------------------


def _install_fake_get_json(monkeypatch, dispatch):
    """Replace adam._get_json with a closure returning canned dicts.

    `dispatch(url, params)` returns the parsed JSON dict for that call.
    Tests that need to vary responses across pages or assert seen params
    capture state via the closure.
    """

    def fake(url, params=None, **kwargs):
        return dispatch(url, params)

    monkeypatch.setattr(adam, "_get_json", fake)


class _StubResponse:
    """Minimal stand-in for requests.Response — enough to satisfy
    .raise_for_status() and .text reads inside get_exposure."""

    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(
                f"{self.status_code} stub error"
            )
            err.response = self
            raise err


def _install_fake_csv_get(monkeypatch, csv_text: str, status: int = 200):
    """Replace adam._session.get with a stub returning csv_text. status>=400
    triggers HTTPError via raise_for_status."""

    def fake(url, **kwargs):
        return _StubResponse(csv_text, status)

    monkeypatch.setattr(adam._session, "get", fake)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _event_feature(
    event_id=1001111,
    episode_id=23,
    name="MILTON-24",
    source="NOAA",
    from_date="2024-10-05T15:00:00",
    to_date="2024-10-10T21:00:00",
    alert_level="Orange",
    population_csv_url: Optional[str] = "https://example/csv",
    uid=None,
):
    """Build an ADAM /items feature with overridable fields."""
    return {
        "properties": {
            "event_id": event_id,
            "episode_id": episode_id,
            "uid": uid if uid is not None else f"{event_id}_{episode_id}",
            "name": name,
            "source": source,
            "from_date": from_date,
            "to_date": to_date,
            "alert_level": alert_level,
            "population_csv_url": population_csv_url,
        },
        "geometry": {"type": "Point", "coordinates": [-83, 27]},
    }


def _page(features):
    """Wrap features in the OGC FeatureCollection envelope ADAM uses, then
    base64-encode it (because adam._get_json is what unwraps; we sidestep
    that by installing a fake that returns the dict directly — but tests
    can still construct the wire format if needed)."""
    return {"features": features}


def _make_csv(rows):
    """rows: list of (ADM0_NAME, ADM1_NAME, ADM2_NAME, POP_60, POP_90, POP_120)."""
    header = "ADM0_NAME,ADM1_NAME,ADM2_NAME,POP_60_KMH,POP_90_KMH,POP_120_KMH\n"
    body = "\n".join(",".join(str(x) for x in r) for r in rows)
    return header + body


# ---------------------------------------------------------------------------
# make_cumulative — the critical semantic test
# ---------------------------------------------------------------------------


def test_make_cumulative_per_band_becomes_geq_threshold():
    """ADAM stores POP_60_KMH as "people in the 60-90 band only", not
    "people ≥ 60 km/h". This converter must produce cumulative ≥-threshold
    so the numbers are comparable to GDACS (which is cumulative).
    Wrong logic here gives wrong numbers but doesn't crash — silent
    semantic bug — so the explicit arithmetic test matters."""
    df = pd.DataFrame({
        "POP_60_KMH": [100, 50],
        "POP_90_KMH": [30, 20],
        "POP_120_KMH": [10, 5],
    })
    out = adam.make_cumulative(df)
    # Row 0: ≥60 = 100+30+10=140; ≥90 = 30+10=40; ≥120 = 10
    assert list(out["POP_60_KMH"]) == [140, 75]
    assert list(out["POP_90_KMH"]) == [40, 25]
    assert list(out["POP_120_KMH"]) == [10, 5]


def test_make_cumulative_preserves_nan():
    """Null in the lower band stays null (we don't know the count, so we
    don't fabricate a value by adding higher bands to 0). Null in higher
    bands is treated as 0 for the addition (you can't add NaN to a count;
    the lower band's known value should still increase by the known
    higher-band amount)."""
    df = pd.DataFrame({
        "POP_60_KMH": [None, 100],
        "POP_90_KMH": [50, None],
        "POP_120_KMH": [10, 5],
    })
    out = adam.make_cumulative(df)
    assert pd.isna(out["POP_60_KMH"].iloc[0])
    assert out["POP_60_KMH"].iloc[1] == 105  # 100 + 0 (NaN) + 5
    assert out["POP_90_KMH"].iloc[0] == 60   # 50 + 10
    assert pd.isna(out["POP_90_KMH"].iloc[1])


def test_make_cumulative_handles_subset_of_columns():
    """If only POP_60_KMH is present (the other bands genuinely missing),
    nothing to add — the column should pass through unchanged rather than
    crash."""
    df = pd.DataFrame({"POP_60_KMH": [100, 50]})
    out = adam.make_cumulative(df)
    assert list(out["POP_60_KMH"]) == [100, 50]


# ---------------------------------------------------------------------------
# name_to_iso3
# ---------------------------------------------------------------------------


def test_name_to_iso3_pycountry_happy_path():
    assert adam.name_to_iso3("Bahamas") == "BHS"
    assert adam.name_to_iso3("Mexico") == "MEX"
    assert adam.name_to_iso3("United States of America") == "USA"


def test_name_to_iso3_override_wins_over_pycountry():
    """Territories whose ADAM name embeds the parent country in parens
    confuse pycountry (it may match either the territory or the parent or
    fail entirely). The hard-coded override is intentional."""
    assert adam.name_to_iso3("Puerto Rico (USA)") == "PRI"
    assert adam.name_to_iso3("United States Virgin Islands (USA)") == "VIR"


def test_name_to_iso3_unresolvable_returns_none():
    """Callers (the pipeline) store iso3=None when ADAM gives a name that
    can't be resolved — exit-loud-by-NULL rather than crash. Schema is
    nullable on iso3 to honor this."""
    assert adam.name_to_iso3("Definitely-Not-A-Country-XYZ-99") is None


def test_name_to_iso3_empty_string_returns_none():
    assert adam.name_to_iso3("") is None


# ---------------------------------------------------------------------------
# _event_feature_to_row — required vs optional fields
# ---------------------------------------------------------------------------


def test_event_feature_to_row_minimal_happy_path():
    row = adam._event_feature_to_row(_event_feature())
    assert row["event_id"] == 1001111
    assert row["episode_id"] == 23
    assert row["name"] == "MILTON-24"
    assert row["population_csv_url"] == "https://example/csv"


def test_event_feature_to_row_missing_required_field_raises():
    """Required fields use direct subscripts so contract breaks surface
    immediately rather than silently producing None and bypassing pandera
    coerce. event_id is the most load-bearing — that's the join key."""
    feat = _event_feature()
    del feat["properties"]["event_id"]
    with pytest.raises(KeyError, match="event_id"):
        adam._event_feature_to_row(feat)


def test_event_feature_to_row_missing_csv_url_is_none():
    """population_csv_url is genuinely absent on some legacy events —
    optional, returns None. Callers raise NoExposureCSVError downstream."""
    feat = _event_feature(population_csv_url=None)
    # _event_feature builds with population_csv_url=None → still has the
    # key but value None. Also test absence-of-key path.
    del feat["properties"]["population_csv_url"]
    row = adam._event_feature_to_row(feat)
    assert row["population_csv_url"] is None


# ---------------------------------------------------------------------------
# get_events — pagination, dedup, source filter, date overlap
# ---------------------------------------------------------------------------


def test_get_events_pagination_terminates_on_partial_page(monkeypatch):
    """Loop must stop when a page returns fewer than _PAGE_SIZE results.
    Without this, we'd hit infinite loop or eat the cost of an extra
    round-trip."""
    pages = [
        _page([_event_feature(event_id=i, episode_id=1) for i in range(adam._PAGE_SIZE)]),
        _page([_event_feature(event_id=999, episode_id=1)]),  # partial page → terminates
    ]
    call_count = {"n": 0}

    def dispatch(url, params):
        i = call_count["n"]
        call_count["n"] += 1
        return pages[i] if i < len(pages) else _page([])

    _install_fake_get_json(monkeypatch, dispatch)
    df = adam.get_events()
    # _PAGE_SIZE events from page 1 + 1 from page 2 = _PAGE_SIZE + 1 unique
    assert len(df) == adam._PAGE_SIZE + 1
    # Verify it stopped after page 2 (didn't try page 3)
    assert call_count["n"] == 2


def test_get_events_dedupes_to_latest_episode(monkeypatch):
    """ADAM returns one feature per (event_id, episode_id). We want one row
    per event_id with the highest episode_id (latest cumulative snapshot)."""
    features = [
        _event_feature(event_id=1001111, episode_id=5),
        _event_feature(event_id=1001111, episode_id=23),  # latest
        _event_feature(event_id=1001111, episode_id=10),
        _event_feature(event_id=1001110, episode_id=2),
    ]
    _install_fake_get_json(monkeypatch, lambda u, p: _page(features))
    df = adam.get_events()
    assert len(df) == 2  # two unique event_ids
    milton = df[df["event_id"] == 1001111].iloc[0]
    assert milton["episode_id"] == 23


def test_get_events_passes_source_filter_to_api(monkeypatch):
    """`source` is a server-side param — verify it gets sent. Saves us
    pagination through events from other basins (~50% of total)."""
    seen_params = []

    def dispatch(url, params):
        seen_params.append(dict(params or {}))
        return _page([])

    _install_fake_get_json(monkeypatch, dispatch)
    adam.get_events(source="NOAA")
    assert seen_params[0].get("source") == "NOAA"


def test_get_events_date_filter_uses_overlap_not_containment(monkeypatch):
    """Pin the overlap semantic. An event that starts before the window
    but ends inside the window MUST be included (e.g., KIRK-24 starting
    Sep 29 for a 2024-10-01..15 query). Earlier implementation used
    strict containment, which silently dropped these — regression
    target."""
    features = [
        # Starts before window, ends inside → OVERLAPS, include.
        _event_feature(
            event_id=1, from_date="2024-09-29T00:00:00",
            to_date="2024-10-07T00:00:00",
        ),
        # Fully inside window → include.
        _event_feature(
            event_id=2, from_date="2024-10-05T00:00:00",
            to_date="2024-10-10T00:00:00",
        ),
        # Starts inside, ends after window → OVERLAPS, include.
        _event_feature(
            event_id=3, from_date="2024-10-12T00:00:00",
            to_date="2024-10-20T00:00:00",
        ),
        # Fully before window → exclude.
        _event_feature(
            event_id=4, from_date="2024-09-01T00:00:00",
            to_date="2024-09-25T00:00:00",
        ),
        # Fully after window → exclude.
        _event_feature(
            event_id=5, from_date="2024-10-20T00:00:00",
            to_date="2024-10-25T00:00:00",
        ),
    ]
    _install_fake_get_json(monkeypatch, lambda u, p: _page(features))
    df = adam.get_events(
        from_date="2024-10-01", to_date="2024-10-15",
    )
    assert set(df["event_id"]) == {1, 2, 3}


def test_get_events_empty_returns_typed_empty_df(monkeypatch):
    """Empty result must still validate against EVENT_SCHEMA. Callers can
    iterate rows or check len() without special-casing."""
    _install_fake_get_json(monkeypatch, lambda u, p: _page([]))
    df = adam.get_events()
    assert len(df) == 0
    assert "event_id" in df.columns


# ---------------------------------------------------------------------------
# get_exposure — error paths
# ---------------------------------------------------------------------------


def test_get_exposure_empty_url_raises():
    with pytest.raises(adam.NoExposureCSVError):
        adam.get_exposure(event_id=1001111, population_csv_url="")


def test_get_exposure_missing_adm0_column_raises(monkeypatch):
    """If WFP changes the CSV schema and drops ADM0_NAME, we can't
    aggregate. Raise NoExposureCSVError (loud) rather than emit empty
    rows."""
    csv = "ADM1_NAME,POP_60_KMH\nFlorida,1000\n"
    _install_fake_csv_get(monkeypatch, csv)
    with pytest.raises(adam.NoExposureCSVError, match="missing ADM0_NAME"):
        adam.get_exposure(event_id=1, population_csv_url="https://example/csv")


def test_get_exposure_no_pop_columns_raises(monkeypatch):
    """CSV with admin columns but no POP_*_KMH columns — no exposure
    data to ingest. Raise rather than emit zero rows that would mask the
    contract break."""
    csv = "ADM0_NAME,ADM1_NAME,ADM2_NAME\nUSA,Florida,Bay\n"
    _install_fake_csv_get(monkeypatch, csv)
    with pytest.raises(adam.NoExposureCSVError, match="missing ADM0_NAME or all"):
        adam.get_exposure(event_id=1, population_csv_url="https://example/csv")


def test_get_exposure_http_403_propagates(monkeypatch):
    """WFP-restricted CSVs return 403. The library should propagate
    HTTPError so the caller (pipeline) can distinguish 403 from 5xx
    and log appropriately. We do NOT want to swallow this here."""
    _install_fake_csv_get(monkeypatch, "", status=403)
    with pytest.raises(requests.exceptions.HTTPError):
        adam.get_exposure(event_id=1, population_csv_url="https://example/csv")


# ---------------------------------------------------------------------------
# get_exposure — happy path: row counts, aggregation, iso3 attribution
# ---------------------------------------------------------------------------


def test_get_exposure_produces_correct_long_form_shape(monkeypatch):
    """One ADAM CSV with 3 ADM2 rows across 1 country, 2 states should
    produce: 1 adm0 row × 3 thresholds + 2 adm1 rows × 3 thresholds + 3
    adm2 rows × 3 thresholds = 18 long-form rows total."""
    csv = _make_csv([
        ("United States of America", "Florida", "Hillsborough", 100, 60, 20),
        ("United States of America", "Florida", "Pinellas",     200, 80, 30),
        ("United States of America", "Georgia", "Chatham",       50, 10, 0),
    ])
    _install_fake_csv_get(monkeypatch, csv)

    out = adam.get_exposure(event_id=1, population_csv_url="https://example/csv")

    assert (out["admin_level"] == 0).sum() == 3   # 1 country × 3 kt
    assert (out["admin_level"] == 1).sum() == 6   # 2 states × 3 kt
    assert (out["admin_level"] == 2).sum() == 9   # 3 districts × 3 kt
    assert len(out) == 18


def test_get_exposure_aggregation_invariant_adm0_equals_sum_adm1(monkeypatch):
    """The strongest internal-consistency check: at each wind threshold,
    SUM of adm1 pop_exposed must equal adm0 pop_exposed. Catches both
    aggregation bugs and out-of-order cumulative conversion."""
    csv = _make_csv([
        ("United States of America", "Florida", "Hillsborough", 100, 60, 20),
        ("United States of America", "Florida", "Pinellas",     200, 80, 30),
        ("United States of America", "Georgia", "Chatham",       50, 10, 0),
    ])
    _install_fake_csv_get(monkeypatch, csv)

    out = adam.get_exposure(event_id=1, population_csv_url="https://example/csv")
    for kt in [34, 50, 64]:
        adm0_total = out[
            (out["admin_level"] == 0) & (out["wind_speed_kt"] == kt)
        ]["pop_exposed"].sum()
        sum_adm1 = out[
            (out["admin_level"] == 1) & (out["wind_speed_kt"] == kt)
        ]["pop_exposed"].sum()
        assert adm0_total == sum_adm1, (
            f"invariant break at {kt} kt: adm0={adm0_total} vs sum(adm1)={sum_adm1}"
        )


def test_get_exposure_applies_cumulative_conversion(monkeypatch):
    """End-to-end: input per-band CSV → cumulative ≥-threshold output.
    Florida row alone, post-cumulative pop at 34 kt should equal raw
    POP_60 + POP_90 + POP_120 from both districts.

    Verify Florida adm1 at 34 kt = (100+60+20) + (200+80+30) = 490.
    """
    csv = _make_csv([
        ("United States of America", "Florida", "Hillsborough", 100, 60, 20),
        ("United States of America", "Florida", "Pinellas",     200, 80, 30),
    ])
    _install_fake_csv_get(monkeypatch, csv)

    out = adam.get_exposure(event_id=1, population_csv_url="https://example/csv")
    florida_34 = out[
        (out["admin_level"] == 1)
        & (out["admin_name"] == "Florida")
        & (out["wind_speed_kt"] == 34)
    ]["pop_exposed"].iloc[0]
    assert florida_34 == 490  # 180 + 310 cumulative


def test_get_exposure_iso3_attribution(monkeypatch):
    """ADM0_NAME gets mapped to ISO3 via name_to_iso3 for every row of
    that country — at all three admin levels. Verify the override path
    fires (territory in parens)."""
    csv = _make_csv([
        ("Puerto Rico (USA)", "SomeState", "SomeDistrict", 10, 5, 1),
    ])
    _install_fake_csv_get(monkeypatch, csv)

    out = adam.get_exposure(event_id=1, population_csv_url="https://example/csv")
    assert (out["iso3"] == "PRI").all()


def test_get_exposure_parent_admin_name_attached(monkeypatch):
    """adm1 rows carry the country in parent_admin_name; adm2 rows carry
    the state. adm0 rows have null parent. Lets consumers navigate the
    hierarchy without joins."""
    csv = _make_csv([
        ("United States of America", "Florida", "Hillsborough", 100, 0, 0),
    ])
    _install_fake_csv_get(monkeypatch, csv)

    out = adam.get_exposure(event_id=1, population_csv_url="https://example/csv")
    adm0 = out[out["admin_level"] == 0].iloc[0]
    adm1 = out[out["admin_level"] == 1].iloc[0]
    adm2 = out[out["admin_level"] == 2].iloc[0]
    # Nullable string column comes back as NaN, not None, after pandera
    # coerce — pandas object-dtype convention.
    assert pd.isna(adm0["parent_admin_name"])
    assert adm1["parent_admin_name"] == "United States of America"
    assert adm2["parent_admin_name"] == "Florida"
