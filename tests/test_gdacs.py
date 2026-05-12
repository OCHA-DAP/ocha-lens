"""Unit tests for ocha_lens.datasources.gdacs.

Tests focus on real behaviors that the source authors (hannah, zack)
wrestled with: name/severity field fallbacks, is_current truthiness,
per-alert-level dedup, client-side source filter, response-shape
parsing. Pure helpers are tested with inline dicts; the three
orchestration functions (get_events, get_timeline,
get_impact_by_country) replace requests.get with a dispatch callable
that returns canned JSON keyed off URL + params.

Mocking at gdacs._get_json is the canonical HTTP boundary inside
the module — every public function routes its requests through it.
"""

import pandas as pd
import pytest
from shapely.geometry import Point

from ocha_lens.datasources import gdacs


def _install_fake_get(monkeypatch, dispatch):
    """Replace gdacs._get_json with a dispatcher returning canned JSON.

    `dispatch(url, params)` should return the parsed-JSON dict the
    test wants to surface for that call. Tests that need to vary
    responses between calls keep state inside the dispatch closure.
    """

    def fake_get_json(url, params=None, **kwargs):
        return dispatch(url, params)

    monkeypatch.setattr(gdacs, "_get_json", fake_get_json)


# ---------------------------------------------------------------------------
# Inline real-shape fixture builders
# ---------------------------------------------------------------------------


def _sample_feature(
    eventid=1234567,
    name="TEST",
    source="NOAA",
    lon=-79.5,
    lat=23.1,
    iscurrent="false",
    severity_kmh=100,
    severity_text="Hurricane Cat 1",
    alertlevel="Red",
):
    """Build a GDACS event feature dict with overridable fields."""
    return {
        "properties": {
            "eventid": eventid,
            "eventname": name,
            "alertlevel": alertlevel,
            "episodealertlevel": alertlevel,
            "iso3": "USA",
            "country": "United States",
            "fromdate": "2024-09-01",
            "todate": "2024-09-05",
            "iscurrent": iscurrent,
            "source": source,
            "severitydata": (
                {"severity": severity_kmh, "severitytext": severity_text}
                if severity_kmh is not None
                else None
            ),
        },
        "geometry": {"coordinates": [lon, lat]},
    }


def _search_response(features):
    return {"features": features}


def _event_detail(timeline_url=None, buffer_urls=None):
    resource = {}
    if timeline_url:
        resource["timeline"] = timeline_url
    if buffer_urls:
        resource.update(buffer_urls)
    return {"properties": {"impacts": [{"resource": resource}]}}


def _datum_group(alias, rows):
    """One datum_group with the given alias and scalar rows."""
    return {
        "alias": alias,
        "datum": [
            {
                "scalars": {
                    "scalar": [{"name": k, "value": v} for k, v in row.items()]
                }
            }
            for row in rows
        ],
    }


def _buffer_response(*datum_groups):
    """GDACS impact-buffer response with one or more datum groups."""
    return {"datums": list(datum_groups)}


def _timeline_response(items):
    return {"channel": {"item": items}}


# ---------------------------------------------------------------------------
# _event_feature_to_row
# ---------------------------------------------------------------------------


def test_event_feature_to_row_with_all_required_fields():
    """A feature with all required fields populates the row dict
    correctly. eventid is coerced from string to int."""
    feat = _sample_feature(eventid="999")
    row = gdacs._event_feature_to_row(feat)
    assert row["eventid"] == 999
    assert isinstance(row["eventid"], int)
    assert row["is_current"] is False
    assert row["severity_kmh"] == 100  # _sample_feature default


def test_event_feature_to_row_missing_eventname_raises():
    """Missing `eventname` is a GDACS contract break — raise KeyError
    rather than silently using the empty string. If GDACS ever
    renames or drops this field we want to find out, not ingest
    nameless records."""
    feat = _sample_feature()
    del feat["properties"]["eventname"]
    with pytest.raises(KeyError):
        gdacs._event_feature_to_row(feat)


def test_event_feature_to_row_missing_geometry_raises():
    """Missing geometry coordinates raise rather than emit a row
    with (lon=None, lat=None) — silent NaN geometries are bad data
    we don't want to ingest."""
    feat = _sample_feature()
    del feat["geometry"]
    with pytest.raises(KeyError):
        gdacs._event_feature_to_row(feat)


def test_event_feature_to_row_severitydata_none_does_not_crash():
    """severitydata can be None in the API response.

    Both hannah's and zack's source code defensively coerce to {} via
    `or {}` — without that, .get("severity") would raise AttributeError.
    """
    feat = _sample_feature()
    feat["properties"]["severitydata"] = None
    row = gdacs._event_feature_to_row(feat)
    assert row["severity_kmh"] is None
    assert row["severity_text"] is None


@pytest.mark.parametrize(
    "iscurrent_value,expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("", False),
        ("anything-else", False),
    ],
)
def test_event_feature_to_row_iscurrent_string_to_bool(
    iscurrent_value, expected
):
    """is_current arrives as a string; conversion must be case-insensitive
    and only "true" (any case) is truthy.
    """
    feat = _sample_feature()
    feat["properties"]["iscurrent"] = iscurrent_value
    assert gdacs._event_feature_to_row(feat)["is_current"] is expected


def test_event_feature_to_row_optional_iso3_country_can_be_null():
    """iso3 and country can legitimately be missing on some GDACS
    events (e.g., very small or open-ocean storms). These are not
    required — caller can handle None."""
    feat = _sample_feature()
    del feat["properties"]["iso3"]
    del feat["properties"]["country"]
    row = gdacs._event_feature_to_row(feat)
    assert row["iso3"] is None
    assert row["country"] is None


def test_event_feature_to_row_geometry_coords_preserved():
    """Coordinates from feature.geometry survive into the row dict
    under the keys consumed by `_to_gdf` (longitude/latitude)."""
    feat = _sample_feature(lon=-79.5, lat=23.1)
    row = gdacs._event_feature_to_row(feat)
    assert row["longitude"] == -79.5
    assert row["latitude"] == 23.1


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------


def test_get_events_pagination_terminates_on_partial_page(monkeypatch):
    """When a page returns fewer than page_size features, stop paging.
    Without the early-break, the loop would do one extra empty fetch
    per alert level on storm-rare date ranges.
    """
    feat = _sample_feature(eventid=1)
    pointers = {
        "Green": iter([_search_response([feat])]),  # 1 < page_size=10
        "Orange": iter([_search_response([])]),
        "Red": iter([_search_response([])]),
    }

    def fake(url, params=None, **kwargs):
        return next(pointers[params["alertlevel"]], _search_response([]))

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_events(page_size=10)
    assert len(result) == 1


def test_get_events_dedup_across_alert_levels(monkeypatch):
    """A storm's alert level can change during its lifetime, causing
    the same eventid to appear under multiple per-level queries.
    Hannah's deliberate per-level-then-dedup design prevents duplicates;
    the first appearance wins.
    """
    shared = _sample_feature(eventid=42, alertlevel="Green")
    only_green = _sample_feature(eventid=43, alertlevel="Green")
    only_orange = _sample_feature(eventid=44, alertlevel="Orange")

    pointers = {
        "Green": iter([_search_response([shared, only_green])]),
        "Orange": iter([_search_response([shared, only_orange])]),
        "Red": iter([_search_response([])]),
    }

    def fake(url, params=None, **kwargs):
        return next(pointers[params["alertlevel"]], _search_response([]))

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_events()
    assert sorted(result["eventid"].tolist()) == [42, 43, 44]


def test_get_events_source_filter_is_clientside(monkeypatch):
    """Hannah found the API's source param unreliable — events tagged
    with an unwanted source still come back. The library applies the
    filter client-side after parsing each feature.
    """
    noaa = _sample_feature(eventid=1, source="NOAA")
    jtwc = _sample_feature(eventid=2, source="JTWC")

    def fake(url, params=None, **kwargs):
        if params["alertlevel"] == "Red" and params["pageNumber"] == 1:
            return _search_response([noaa, jtwc])
        return _search_response([])

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_events(source="NOAA")
    assert result["eventid"].tolist() == [1]


def test_get_events_empty_returns_typed_geodataframe(monkeypatch):
    """Empty result must still return a GeoDataFrame with the expected
    columns + EPSG:4326 — so callers can chain operations without
    a special case for "no events".
    """

    def fake(url, params=None, **kwargs):
        return _search_response([])

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_events()
    assert len(result) == 0
    assert "geometry" in result.columns
    assert str(result.crs) == "EPSG:4326"
    expected = {
        "eventid",
        "name",
        "alert_level",
        "iso3",
        "country",
        "from_date",
        "to_date",
        "is_current",
        "severity_kmh",
        "severity_text",
        "source",
        "geometry",
    }
    assert expected.issubset(set(result.columns))


def test_get_events_geometry_built_from_coords(monkeypatch):
    """Resulting GeoDataFrame has Point geometries with x,y matching
    feature coordinates — invariant whether built via points_from_xy
    or _to_gdf.
    """
    feat = _sample_feature(eventid=1, lon=-79.5, lat=23.1)

    def fake(url, params=None, **kwargs):
        if params["alertlevel"] == "Red" and params["pageNumber"] == 1:
            return _search_response([feat])
        return _search_response([])

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_events()
    geom = result.iloc[0].geometry
    assert isinstance(geom, Point)
    assert geom.x == pytest.approx(-79.5)
    assert geom.y == pytest.approx(23.1)


# ---------------------------------------------------------------------------
# get_exposure_adm0 and get_exposure_adm1
# ---------------------------------------------------------------------------


_BUFFER_URLS = {
    "buffer39": "https://gdacs/buffer39.json",
    "buffer74": "https://gdacs/buffer74.json",
}


_ADM0_ROW = {
    "ISO_3DIGIT": "USA",
    "ISO_2DIGIT": "US",
    "CNTRY_NAME": "United States",
    "GMI_CNTRY": "USA",
    "POP_AFFECTED": 12059178,
    "distance": 12.345,
}

_ADM1_ROW = {
    "GMI_CNTRY": "USA",
    "CNTRY_NAME": "United States",
    "FIPS_ADMIN": "US05",
    "GMI_ADMIN": "USA-ARK",
    "ADMIN_NAME": "Arkansas",
    "TYPE_ENG": "State",
    "POP_ADMIN": 3000000,
    "POP_AFFECTED": 161050,
    "distance": 50.789,
}


def _install_buffer_fixture(monkeypatch, *datum_groups):
    """Common fake: event detail + a single buffer response."""
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = _buffer_response(*datum_groups)

    def fake(url, params=None, **kwargs):
        if "geteventdata" in url or "getepisodedata" in url:
            return detail
        return buf

    _install_fake_get(monkeypatch, fake)


# ----- get_exposure_adm0 -----


def test_get_exposure_adm0_reads_country_alias(monkeypatch):
    """ADM0 builder reads alias='country' directly (no aggregation).
    The iso3 column holds GDACS' native GMI_CNTRY — which is the
    valid ISO3 for sovereign states but X-prefixed for territories
    (caller applies to_iso3 for standardization).
    """
    _install_buffer_fixture(monkeypatch, _datum_group("country", [_ADM0_ROW]))
    result = gdacs.get_exposure_adm0(eventid=1)
    assert set(result.keys()) == {"buffer39", "buffer74"}

    df = result["buffer39"]
    assert len(df) == 1
    assert df.iloc[0]["iso3"] == "USA"
    assert df.iloc[0]["country"] == "United States"
    assert df.iloc[0]["pop_affected"] == 12059178


def test_get_exposure_adm0_ignores_alert_alias(monkeypatch):
    """When only alias='alert' is present (no country rollup), ADM0
    builder returns no rows — it does NOT fall back to aggregating
    the ADM1 grain, which was the old behavior we're moving away from.
    """
    _install_buffer_fixture(monkeypatch, _datum_group("alert", [_ADM1_ROW]))
    result = gdacs.get_exposure_adm0(eventid=1)
    assert len(result["buffer39"]) == 0


def test_get_exposure_adm0_distance_rounded_to_one_decimal(monkeypatch):
    _install_buffer_fixture(monkeypatch, _datum_group("country", [_ADM0_ROW]))
    result = gdacs.get_exposure_adm0(eventid=1)
    assert result["buffer39"].iloc[0]["distance_km"] == 12.3


def test_get_exposure_adm0_empty_returns_typed_dataframe(monkeypatch):
    """No country rows → empty DataFrame with expected columns so
    callers can chain df['iso3'] without KeyError."""
    _install_buffer_fixture(monkeypatch, _datum_group("country", []))
    result = gdacs.get_exposure_adm0(eventid=1)
    df = result["buffer39"]
    assert len(df) == 0
    assert list(df.columns) == [
        "iso3",
        "country",
        "pop_affected",
        "distance_km",
    ]


# ----- get_exposure_adm1 -----


def test_get_exposure_adm1_reads_alert_alias(monkeypatch):
    """ADM1 builder reads alias='alert' and extracts the full admin
    field set (fips/gmi/name/type) plus the parent country attachment.
    """
    _install_buffer_fixture(monkeypatch, _datum_group("alert", [_ADM1_ROW]))
    result = gdacs.get_exposure_adm1(eventid=1)
    df = result["buffer39"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["iso3"] == "USA"
    assert row["country"] == "United States"
    assert row["fips_admin"] == "US05"
    assert row["gmi_admin"] == "USA-ARK"
    assert row["admin_name"] == "Arkansas"
    assert row["admin_type"] == "State"
    assert row["pop_admin"] == 3000000
    assert row["pop_affected"] == 161050
    assert row["distance_km"] == 50.8


def test_get_exposure_adm1_ignores_country_alias(monkeypatch):
    """ADM1 builder must NOT read alias='country' — that's the ADM0
    rollup, different grain entirely."""
    _install_buffer_fixture(monkeypatch, _datum_group("country", [_ADM0_ROW]))
    result = gdacs.get_exposure_adm1(eventid=1)
    assert len(result["buffer39"]) == 0


def test_get_exposure_adm1_missing_gmi_admin_raises(monkeypatch):
    """Missing GMI_ADMIN under alias='alert' is a GDACS contract
    violation — raise KeyError loudly rather than silently skip the
    row. If GDACS ever changes the alert-alias scalar set, we want
    to find out, not lose rows quietly."""
    bad_row = {k: v for k, v in _ADM1_ROW.items() if k != "GMI_ADMIN"}
    _install_buffer_fixture(
        monkeypatch, _datum_group("alert", [_ADM1_ROW, bad_row])
    )
    with pytest.raises(KeyError):
        gdacs.get_exposure_adm1(eventid=1)


def test_get_exposure_adm1_empty_returns_typed_dataframe(monkeypatch):
    _install_buffer_fixture(monkeypatch, _datum_group("alert", []))
    result = gdacs.get_exposure_adm1(eventid=1)
    df = result["buffer39"]
    assert len(df) == 0
    assert "fips_admin" in df.columns
    assert "admin_name" in df.columns
    assert "pop_admin" in df.columns


# ----- episodeid routing (shared between adm0 and adm1) -----


def test_exposure_with_episodeid_routes_via_episode_detail(monkeypatch):
    """Passing episodeid must route detail-fetch to getepisodedata,
    not geteventdata — historical episode snapshots can differ from
    the current/latest data.
    """
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = _buffer_response(_datum_group("country", [_ADM0_ROW]))
    seen_endpoints = []

    def fake(url, params=None, **kwargs):
        seen_endpoints.append(url)
        if "getepisodedata" in url or "geteventdata" in url:
            return detail
        return buf

    _install_fake_get(monkeypatch, fake)
    gdacs.get_exposure_adm0(eventid=1, episodeid=42)
    assert any("getepisodedata" in u for u in seen_endpoints)
    assert not any("geteventdata" in u for u in seen_endpoints)


# ---------------------------------------------------------------------------
# get_timeline
# ---------------------------------------------------------------------------


def test_get_timeline_parses_advisories_with_coercion_and_sort(monkeypatch):
    """Timeline parsing must:
    - coerce string numerics to numeric dtype
    - parse advisory_datetime to Timestamp
    - sort by advisory_number ascending
    - flatten storm_status from list to scalar (first element)
    """
    detail = _event_detail(timeline_url="https://gdacs/timeline.json")
    items = [
        {
            "advisory_number": "3",
            "name": "TEST",
            "advisory_datetime": "2024-09-01T18:00:00Z",
            "actual": "true",
            "current": "true",
            "alertcolor": "Red",
            "country": "USA",
            "latitude": "23.5",
            "longitude": "-79.0",
            "wind_speed": "85",
            "pop": "1000",
            "pop39": "500",
            "pop74": "100",
            "storm_status": ["Hurricane"],
        },
        {
            "advisory_number": "1",  # earlier; should sort to top
            "name": "TEST",
            "advisory_datetime": "2024-09-01T06:00:00Z",
            "actual": "true",
            "current": "false",
            "alertcolor": "Orange",
            "country": "USA",
            "latitude": "22.0",
            "longitude": "-80.0",
            "wind_speed": "55",
            "pop": "0",
            "pop39": "0",
            "pop74": "0",
            "storm_status": ["Tropical Storm"],
        },
    ]

    def fake(url, params=None, **kwargs):
        return detail if "geteventdata" in url else _timeline_response(items)

    _install_fake_get(monkeypatch, fake)
    df = gdacs.get_timeline(eventid=1)
    assert list(df["advisory_number"]) == [1, 3]
    assert df.iloc[0]["wind_speed"] == 55.0
    assert pd.api.types.is_numeric_dtype(df["wind_speed"])
    assert pd.api.types.is_datetime64_any_dtype(df["advisory_datetime"])
    assert df.iloc[0]["storm_status"] == "Tropical Storm"


def test_get_timeline_no_url_raises_no_timeline_error(monkeypatch):
    """Missing timeline resource raises NoTimelineError loudly so
    the caller can decide whether to skip or abort. Previously this
    returned an empty DataFrame which was indistinguishable from
    'event has timeline but zero advisories' — a meaningful
    distinction the caller now gets."""
    detail = _event_detail()  # empty resource, no timeline URL

    def fake(url, params=None, **kwargs):
        return detail

    _install_fake_get(monkeypatch, fake)
    with pytest.raises(gdacs.NoTimelineError):
        gdacs.get_timeline(eventid=1)


# ---------------------------------------------------------------------------
# latest_episode_id
# ---------------------------------------------------------------------------


def test_latest_episode_id_extracts_from_url():
    """GDACS embeds the episode id in the `details` URL query param.
    The helper pulls it out reliably for the last entry in
    `properties.episodes`."""
    detail = {
        "properties": {
            "episodes": [
                {"details": "https://gdacs/api/foo?eventid=1&episodeid=5&x=1"},
                {"details": "https://gdacs/api/foo?eventid=1&episodeid=22&x=1"},
            ]
        }
    }
    assert gdacs.latest_episode_id(detail) == 22


def test_latest_episode_id_no_episodes_raises():
    """Event with empty episodes list raises NoEpisodesError —
    legitimately new event the caller may want to handle, but
    distinct from a malformed payload."""
    with pytest.raises(gdacs.NoEpisodesError):
        gdacs.latest_episode_id({"properties": {"episodes": []}})


def test_latest_episode_id_missing_properties_raises():
    """Event detail without the expected `properties` or
    `episodes` keys raises KeyError — that's a GDACS contract
    break, not a normal 'no episodes yet' case."""
    with pytest.raises(KeyError):
        gdacs.latest_episode_id({"properties": {}})
    with pytest.raises(KeyError):
        gdacs.latest_episode_id({})


def test_to_iso3_passes_through_standard_codes():
    """Most GMI_CNTRY values are already valid ISO 3166-1 alpha-3 —
    to_iso3 must not transform them."""
    assert gdacs.to_iso3("USA") == "USA"
    assert gdacs.to_iso3("MEX") == "MEX"
    assert gdacs.to_iso3("BHS") == "BHS"


def test_to_iso3_remaps_proprietary_codes():
    """X-prefixed GDACS codes for territories/dependencies get
    remapped to the standard ISO 3166-1 code. New mappings get
    added as we encounter them in real data."""
    assert gdacs.to_iso3("XJE") == "JEY"   # Jersey


def test_latest_episode_id_malformed_url_raises():
    """If GDACS's details URL doesn't contain `episodeid=`, raise
    EpisodeUrlFormatError — likely a GDACS API contract change."""
    detail = {
        "properties": {
            "episodes": [{"details": "https://gdacs/api/foo?eventid=1"}]
        }
    }
    with pytest.raises(gdacs.EpisodeUrlFormatError):
        gdacs.latest_episode_id(detail)
