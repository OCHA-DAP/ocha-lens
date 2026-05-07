"""Unit tests for ocha_lens.datasources.gdacs.

Tests focus on real behaviors that the source authors (hannah, zack)
wrestled with: name/severity field fallbacks, is_current truthiness,
per-alert-level dedup, client-side source filter, response-shape
parsing. Pure helpers are tested with inline dicts; the three
orchestration functions (get_events, get_timeline,
get_impact_by_country) replace requests.get with a dispatch callable
that returns canned JSON keyed off URL + params.

Mocking at requests.get is intentional: it remains the actual HTTP
boundary in both the pre- and post-simplify versions of gdacs.py,
so these tests survive the upcoming refactor.
"""

import pandas as pd
import pytest
from shapely.geometry import Point

from ocha_lens.datasources import gdacs


class _FakeResponse:
    """Minimal stand-in for requests.Response used across tests."""

    def __init__(self, json_data):
        self._json = json_data
        self.text = "{}" if json_data is not None else ""

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def _install_fake_get(monkeypatch, dispatch):
    """Replace requests.get in the gdacs module with a dispatcher.

    `dispatch(url, params)` should return the JSON-decoded body the
    fake response will surface. Tests that need to vary response
    bodies between calls keep state inside the dispatch closure.
    """

    def fake_get(url, params=None, timeout=None, **kwargs):
        return _FakeResponse(dispatch(url, params))

    monkeypatch.setattr(gdacs.requests, "get", fake_get)


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


def _buffer_response(country_rows):
    """GDACS impact-buffer response with the given scalar rows."""
    return {
        "datums": [
            {
                "alias": "alert",
                "datum": [
                    {
                        "scalars": {
                            "scalar": [
                                {"name": k, "value": v} for k, v in row.items()
                            ]
                        }
                    }
                    for row in country_rows
                ],
            }
        ]
    }


def _timeline_response(items):
    return {"channel": {"item": items}}


# ---------------------------------------------------------------------------
# _event_feature_to_row
# ---------------------------------------------------------------------------


def test_event_feature_to_row_minimal():
    """Minimal feature: eventid coerced to int, missing fields default cleanly."""
    feat = {
        "properties": {"eventid": "999"},  # API often returns string ids
        "geometry": {"coordinates": [50.0, 10.0]},
    }
    row = gdacs._event_feature_to_row(feat)
    assert row["eventid"] == 999
    assert isinstance(row["eventid"], int)
    assert row["name"] == ""
    assert row["is_current"] is False
    assert row["severity_kmh"] is None
    assert row["severity_text"] is None


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


def test_event_feature_to_row_name_fallback_chain():
    """`eventname` wins; if missing, falls back to `name`; else empty string."""
    feat = _sample_feature()
    feat["properties"]["eventname"] = None
    feat["properties"]["name"] = "fallback-name"
    assert gdacs._event_feature_to_row(feat)["name"] == "fallback-name"

    feat["properties"]["name"] = None
    assert gdacs._event_feature_to_row(feat)["name"] == ""


def test_event_feature_to_row_geometry_coords_preserved():
    """Coordinates from feature.geometry survive into the row dict."""
    feat = _sample_feature(lon=-79.5, lat=23.1)
    row = gdacs._event_feature_to_row(feat)
    # Current implementation uses lon/lat keys; if simplify renames to
    # longitude/latitude, this test will fail and that test commit
    # should update the assertion.
    assert row["lon"] == -79.5
    assert row["lat"] == 23.1


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
# get_impact_by_country
# ---------------------------------------------------------------------------


_BUFFER_URLS = {
    "buffer39": "https://gdacs/buffer39.json",
    "buffer74": "https://gdacs/buffer74.json",
}


def test_get_impact_by_country_aggregates_by_iso3(monkeypatch):
    """aggregate=True: multiple admin rows for one country collapse into
    a single row with summed pop_affected.
    """
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = _buffer_response(
        [
            {
                "GMI_CNTRY": "USA",
                "CNTRY_NAME": "United States",
                "POP_ADMIN": 1000,
                "POP_AFFECTED": 100,
                "distance": 50.0,
            },
            {
                "GMI_CNTRY": "USA",
                "CNTRY_NAME": "United States",
                "POP_ADMIN": 1000,
                "POP_AFFECTED": 200,
                "distance": 75.0,
            },
            {
                "GMI_CNTRY": "MEX",
                "CNTRY_NAME": "Mexico",
                "POP_ADMIN": 500,
                "POP_AFFECTED": 50,
                "distance": 100.0,
            },
        ]
    )

    def fake(url, params=None, **kwargs):
        return detail if "geteventdata" in url else buf

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_impact_by_country(eventid=1, aggregate=True)
    assert set(result.keys()) == {"buffer39", "buffer74"}

    df = result["buffer39"]
    usa = df[df["iso3"] == "USA"]
    assert len(usa) == 1
    assert usa.iloc[0]["pop_affected"] == 300
    # MEX preserved as separate row
    mex = df[df["iso3"] == "MEX"]
    assert len(mex) == 1
    assert mex.iloc[0]["pop_affected"] == 50


def test_get_impact_by_country_skips_rows_without_gmi_cntry(monkeypatch):
    """The "alert" alias mixes country-level and sub-country rows. The
    parser keeps only those with GMI_CNTRY (country-level).
    """
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = {
        "datums": [
            {
                "alias": "alert",
                "datum": [
                    # country-level, kept
                    {
                        "scalars": {
                            "scalar": [
                                {"name": "GMI_CNTRY", "value": "USA"},
                                {
                                    "name": "CNTRY_NAME",
                                    "value": "United States",
                                },
                                {"name": "POP_ADMIN", "value": 1000},
                                {"name": "POP_AFFECTED", "value": 100},
                                {"name": "distance", "value": 0},
                            ]
                        }
                    },
                    # sub-country (FIPS_ADMIN, no GMI_CNTRY) — skipped
                    {
                        "scalars": {
                            "scalar": [
                                {"name": "FIPS_ADMIN", "value": "USA-NY"},
                                {"name": "POP_AFFECTED", "value": 50},
                            ]
                        }
                    },
                ],
            }
        ]
    }

    def fake(url, params=None, **kwargs):
        return detail if "geteventdata" in url else buf

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_impact_by_country(eventid=1)
    assert len(result["buffer39"]) == 1
    assert result["buffer39"].iloc[0]["iso3"] == "USA"


def test_get_impact_by_country_skips_non_alert_alias(monkeypatch):
    """Only datum_groups with alias=='alert' are parsed. Pins the current
    interpretation (zack's). NOTE: hannah's pipeline reads alias=='country'
    for ADM0 — this discrepancy is open work for a follow-up commit.
    """
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = {
        "datums": [
            {
                "alias": "country",  # not "alert" — should be skipped
                "datum": [
                    {
                        "scalars": {
                            "scalar": [
                                {"name": "GMI_CNTRY", "value": "USA"},
                                {
                                    "name": "CNTRY_NAME",
                                    "value": "United States",
                                },
                                {"name": "POP_ADMIN", "value": 1},
                                {"name": "POP_AFFECTED", "value": 1},
                                {"name": "distance", "value": 0},
                            ]
                        }
                    }
                ],
            }
        ]
    }

    def fake(url, params=None, **kwargs):
        return detail if "geteventdata" in url else buf

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_impact_by_country(eventid=1)
    assert len(result["buffer39"]) == 0


def test_get_impact_by_country_aggregate_false_keeps_all_rows(monkeypatch):
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = _buffer_response(
        [
            {
                "GMI_CNTRY": "USA",
                "CNTRY_NAME": "USA",
                "POP_ADMIN": 1,
                "POP_AFFECTED": 100,
                "distance": 0,
            },
            {
                "GMI_CNTRY": "USA",
                "CNTRY_NAME": "USA",
                "POP_ADMIN": 1,
                "POP_AFFECTED": 200,
                "distance": 0,
            },
        ]
    )

    def fake(url, params=None, **kwargs):
        return detail if "geteventdata" in url else buf

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_impact_by_country(eventid=1, aggregate=False)
    assert len(result["buffer39"]) == 2


def test_get_impact_by_country_distance_rounded_to_one_decimal(monkeypatch):
    detail = _event_detail(buffer_urls=_BUFFER_URLS)
    buf = _buffer_response(
        [
            {
                "GMI_CNTRY": "USA",
                "CNTRY_NAME": "USA",
                "POP_ADMIN": 1,
                "POP_AFFECTED": 1,
                "distance": 123.4567,
            }
        ]
    )

    def fake(url, params=None, **kwargs):
        return detail if "geteventdata" in url else buf

    _install_fake_get(monkeypatch, fake)
    result = gdacs.get_impact_by_country(eventid=1, aggregate=False)
    assert result["buffer39"].iloc[0]["distance_km"] == 123.5


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


@pytest.mark.xfail(
    reason=(
        "Current code returns bare pd.DataFrame() when no timeline URL is "
        "found; simplify will fix this to return a DataFrame with the "
        "expected columns so callers can chain df['advisory_number'] "
        "without KeyError. This test pins the desired post-simplify "
        "behavior."
    ),
    strict=True,
)
def test_get_timeline_no_url_returns_dataframe_with_columns(monkeypatch):
    detail = _event_detail()  # empty resource, no timeline URL

    def fake(url, params=None, **kwargs):
        return detail

    _install_fake_get(monkeypatch, fake)
    df = gdacs.get_timeline(eventid=1)
    assert "advisory_number" in df.columns
    assert "wind_speed" in df.columns
    assert len(df) == 0
