"""Unit tests for ocha_lens.utils.storm.match_wsp_to_tracks.

Focus: the three matching passes, and in particular the cross-threshold
containment fallback (pass 3) added after Dolly/AL042026 2026-08-28 06:00,
where a trailing 50 kt band behind the weakening storm — no longer touched
by the departing forecast track line, and with no matched 50 kt sibling to
contain it — went unattributed and silently dropped ~679k pop of exposure.
A storm's higher-threshold WSP bands nest inside its own lower-threshold
bands, so the matched 34 kt footprint is a valid attribution donor.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from ocha_lens.utils.storm import match_wsp_to_tracks

IT = pd.Timestamp("2026-08-28 06:00:00")


def _wsp(rows):
    return gpd.GeoDataFrame(
        [
            {
                "issued_time": IT,
                "wind_threshold_kt": kt,
                "percentage": pct,
                "geometry": geom,
            }
            for kt, pct, geom in rows
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _tracks(points, atcf_id="AL042026"):
    return gpd.GeoDataFrame(
        [
            {
                "atcf_id": atcf_id,
                "issued_time": IT,
                "valid_time": IT + pd.Timedelta(hours=6 * i),
                "geometry": Point(x, y),
            }
            for i, (x, y) in enumerate(points)
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_line_intersection_matches_band_on_track():
    # Track runs east through the 34 kt band.
    tracks = _tracks([(-62, 17), (-58, 19), (-54, 21)])
    wsp = _wsp([(34, 0, box(-64, 15, -56, 20))])
    out = match_wsp_to_tracks(wsp, tracks)
    assert list(out["atcf_id"]) == ["AL042026"]


def test_cross_threshold_fallback_attributes_trailing_band():
    # The Dolly case: 34 kt band still touches the departing track line;
    # the trailing 50 kt band sits fully inside it but behind the storm,
    # away from the line. No 50 kt sibling matched, so before pass 3 this
    # band came back atcf_id=None.
    tracks = _tracks([(-58, 19), (-54, 21), (-50, 24)])
    band_34 = box(-64, 15, -56, 20)  # track enters at (-58, 19)
    band_50 = box(-63, 16, -60.5, 18.5)  # inside 34 kt, off the line
    wsp = _wsp([(34, 0, band_34), (50, 0, band_50)])
    out = match_wsp_to_tracks(wsp, tracks)
    by_kt = dict(zip(out["wind_threshold_kt"], out["atcf_id"], strict=True))
    assert by_kt[34] == "AL042026"
    assert by_kt[50] == "AL042026"


def test_unrelated_far_band_stays_unmatched():
    # A 50 kt band nowhere near the storm (not inside any matched band)
    # must NOT be attributed by the cross-threshold fallback.
    tracks = _tracks([(-58, 19), (-54, 21), (-50, 24)])
    wsp = _wsp(
        [
            (34, 0, box(-64, 15, -56, 20)),
            (50, 0, box(-120, 10, -117, 13)),  # mid-Pacific, unrelated
        ]
    )
    out = match_wsp_to_tracks(wsp, tracks)
    by_kt = dict(zip(out["wind_threshold_kt"], out["atcf_id"], strict=True))
    assert by_kt[34] == "AL042026"
    assert pd.isna(by_kt[50])


def test_same_threshold_containment_still_wins_first():
    # Pass 2 regression guard: an inner high-percentage 34 kt lobe inside
    # the matched outer 34 kt band matches via same-threshold containment.
    tracks = _tracks([(-62, 17), (-58, 19), (-54, 21)])
    outer = box(-64, 15, -56, 20)
    inner = box(-61, 16, -59, 17.5)  # inside outer, off the track line
    wsp = _wsp([(34, 5, outer), (34, 50, inner)])
    out = match_wsp_to_tracks(wsp, tracks)
    assert list(out["atcf_id"]) == ["AL042026", "AL042026"]


def test_ambiguous_cross_threshold_containment_stays_unmatched():
    # Two storms: the orphan part sits inside BOTH storms' 34 kt bands.
    # The single-storm nesting rationale can't name an owner, so pass 3
    # must leave it unmatched rather than guess — a misattribution would
    # be invisible downstream, unlike an atcf_id=None row.
    tracks = pd.concat(
        [
            _tracks([(-62, 17), (-58, 19)], atcf_id="AL012026"),
            _tracks([(-75, 6), (-45, 9)], atcf_id="AL022026"),
        ],
        ignore_index=True,
    )
    small_a = box(-64, 15, -56, 20)  # crossed by AL012026's line only
    big_b = box(-80, 5, -40, 30)  # crossed by AL022026's line
    # Inside BOTH 34 kt bands, touched by NEITHER track line (west of
    # AL012026's segment, north of AL022026's).
    orphan_50 = box(-63.8, 15.2, -62.5, 16.2)
    wsp = _wsp([(34, 0, small_a), (34, 0, big_b), (50, 0, orphan_50)])
    out = match_wsp_to_tracks(wsp, tracks)
    row_50 = out[out["wind_threshold_kt"] == 50]
    assert [pd.isna(a) for a in row_50["atcf_id"]] == [True]


def test_cross_threshold_attributes_when_only_one_storm_contains():
    # Two storms present, but only AL012026's 34 kt band contains the
    # orphan — unambiguous, so pass 3 attributes it.
    tracks = pd.concat(
        [
            _tracks([(-62, 17), (-58, 19)], atcf_id="AL012026"),
            _tracks([(-75, 6), (-45, 9)], atcf_id="AL022026"),
        ],
        ignore_index=True,
    )
    band_a = box(-64, 15, -56, 20)
    band_b = box(-80, 5, -70, 10)  # far southwest, does not reach the orphan
    orphan_50 = box(-63.8, 15.2, -62.5, 16.2)
    wsp = _wsp([(34, 0, band_a), (34, 0, band_b), (50, 0, orphan_50)])
    out = match_wsp_to_tracks(wsp, tracks)
    row_50 = out[out["wind_threshold_kt"] == 50]
    assert list(row_50["atcf_id"]) == ["AL012026"]


def test_pass2_tiebreak_uses_filled_area():
    # Same-threshold containment ranks candidate containers by FILLED
    # extent — the same geometry the containment test runs against. Storm
    # B's container is a wide thin annulus whose donut hole makes its
    # unfilled .area smaller than storm A's solid band, but its filled
    # extent is far larger. The inner lobe sits inside both (filled)
    # containers and must attribute to A. Containers arrive via
    # extra_containers so track lines play no part.
    tracks = _tracks([(-10, 40), (-8, 42)], atcf_id="AL092026")  # far away
    solid_a = box(-64, 15, -56, 20)  # unfilled area 40
    ring_b = box(-90, 0, -30, 35).difference(box(-89.8, 0.2, -30.2, 34.8))
    assert ring_b.area < solid_a.area  # the trap the fix closes
    containers = gpd.GeoDataFrame(
        [
            {
                "issued_time": IT,
                "wind_threshold_kt": 34,
                "atcf_id": "AL012026",
                "geometry": solid_a,
            },
            {
                "issued_time": IT,
                "wind_threshold_kt": 34,
                "atcf_id": "AL022026",
                "geometry": ring_b,
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    inner = box(-61, 16, -59, 17.5)
    wsp = _wsp([(34, 50, inner)])
    out = match_wsp_to_tracks(wsp, tracks, extra_containers=containers)
    assert list(out["atcf_id"]) == ["AL012026"]


def test_cross_threshold_works_without_percentage_column():
    # Pass 3 load-bears on ascending-kt processing order; that ordering
    # must hold even when the optional percentage column is absent and the
    # input lists the 50 kt part FIRST.
    tracks = _tracks([(-58, 19), (-54, 21), (-50, 24)])
    wsp = gpd.GeoDataFrame(
        [
            {
                "issued_time": IT,
                "wind_threshold_kt": 50,
                "geometry": box(-63, 16, -60.5, 18.5),
            },
            {
                "issued_time": IT,
                "wind_threshold_kt": 34,
                "geometry": box(-64, 15, -56, 20),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    out = match_wsp_to_tracks(wsp, tracks)
    by_kt = dict(zip(out["wind_threshold_kt"], out["atcf_id"], strict=True))
    assert by_kt[34] == "AL042026"
    assert by_kt[50] == "AL042026"
