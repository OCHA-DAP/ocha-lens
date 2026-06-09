from pathlib import Path

from ocha_lens.datasources.nhc import (
    _parse_atcf_adeck,
    _parse_current_center_radii,
    _parse_forecast_advisory,
    get_storms,
    get_tracks,
    load_nhc,
)

# Path to test fixtures
FIXTURES_PATH = Path(__file__).parent / "fixtures"


def test_json_sample_observations():
    """Test parsing observational data from NHC JSON sample."""
    sample_json_path = FIXTURES_PATH / "NHC_JSON_Sample.json"
    df_raw = load_nhc(file_path=sample_json_path, use_cache=False)
    df_storms = get_storms(df_raw)
    df_tracks = get_tracks(df_raw)

    assert len(df_tracks) == 4
    assert len(df_storms) == 4
    assert len(df_storms.storm_id.unique()) == 4


def test_json_empty():
    """Test parsing empty JSON data returns empty DataFrames."""
    sample_json_path = FIXTURES_PATH / "NHC_JSON_Empty.json"
    df_raw = load_nhc(file_path=sample_json_path, use_cache=False)
    df_storms = get_storms(df_raw)
    df_tracks = get_tracks(df_raw)

    assert len(df_tracks) == 0
    assert len(df_storms) == 0


def test_adeck_no_ofcl():
    """Test archival A-Deck parsing: al012023 had no OFCL entries."""
    df_raw = _parse_atcf_adeck(FIXTURES_PATH / "aal012023.dat")
    assert len(df_raw) == 0


def test_adeck_margot():
    """Test archival A-Deck parsing: al142023 is storm MARGOT."""
    df_raw = _parse_atcf_adeck(FIXTURES_PATH / "aal142023.dat")
    df_storms = get_storms(df_raw)
    df_tracks = get_tracks(df_raw)

    assert len(df_storms) == 1
    assert df_storms.name[0] == "MARGOT"
    assert len(df_tracks) == 479
    assert len(df_tracks.storm_id.unique()) == 1
    assert len(df_tracks[df_tracks.leadtime == 0]) == 41


def test_adeck_hilary():
    """Test archival A-Deck parsing: ep092023 is storm HILARY."""
    df_raw = _parse_atcf_adeck(FIXTURES_PATH / "aep092023.dat")
    df_storms = get_storms(df_raw)
    df_tracks = get_tracks(df_raw)

    assert len(df_storms) == 1
    assert df_storms.name[0] == "HILARY"
    assert len(df_tracks) == 210
    assert len(df_tracks.storm_id.unique()) == 1
    assert len(df_tracks[df_tracks.leadtime == 0]) == 26


def test_adeck_incorrect_filename():
    """Test that incorrectly named storm_id returns None."""
    df_raw = _parse_atcf_adeck(FIXTURES_PATH / "no_id.dat")
    assert df_raw is None


def test_forecast_advisory_parsing():
    """Test reading and parsing forecast advisory."""
    with open(FIXTURES_PATH / "TCM_example.txt", "r") as f:
        advisory_text = f.read()

    forecast_points = _parse_forecast_advisory(
        advisory_text=advisory_text,
        storm_id="AL132023",
        storm_name="Lee",
        issuance="2011-06-10T15:00:00Z",
        basin="XX",  # Arbitrary basin for testing
    )

    assert len(forecast_points) == 8

    # Check first point has required fields
    point = forecast_points[0]
    assert "atcf_id" in point
    assert "valid_time" in point
    assert "latitude" in point
    assert "longitude" in point
    assert "wind_speed" in point
    assert point["basin"] == "XX"
    assert point["quadrant_radius_34"] == [150, 140, 100, 140]


def test_forecast_advisory_wrong_id():
    """Test that wrong storm_id returns empty list."""
    with open(FIXTURES_PATH / "TCM_example.txt", "r") as f:
        advisory_text = f.read()

    forecast_points = _parse_forecast_advisory(
        advisory_text=advisory_text,
        storm_id="WRONG_ID",
        storm_name="Lee",
        issuance="2011-06-10T15:00:00Z",
        basin="XX",
    )

    assert len(forecast_points) == 0


def test_current_center_radii_parsing():
    """Current-center (analysis-time) radii are read from the advisory's
    section preceding the first FORECAST VALID line. CurrentStorms.json omits
    these, so this is the only realtime source of leadtime=0 wind radii."""
    with open(FIXTURES_PATH / "TCM_example.txt", "r") as f:
        advisory_text = f.read()

    radii = _parse_current_center_radii(advisory_text)

    # From the analysis block of the Lee advisory (lines "64/50/34 KT...").
    assert radii["quadrant_radius_64"] == [40, 35, 30, 40]
    assert radii["quadrant_radius_50"] == [90, 70, 50, 80]
    assert radii["quadrant_radius_34"] == [150, 140, 100, 140]


def test_current_center_radii_stops_before_forecast():
    """Radii from the first FORECAST VALID hour must not leak into the
    current-center result (the 34 KT forecast[0] line is 150/140/100/140 too,
    so assert the 64 KT value, which differs: analysis 40NE vs forecast 50NE)."""
    with open(FIXTURES_PATH / "TCM_example.txt", "r") as f:
        advisory_text = f.read()

    radii = _parse_current_center_radii(advisory_text)
    # Analysis 64 KT NE radius is 40; the first forecast hour's is 50.
    assert radii["quadrant_radius_64"][0] == 40


def test_current_center_radii_absent():
    """A system with no 34/50/64 KT analysis lines yields all-None."""
    text_no_radii = (
        "NWS NATIONAL HURRICANE CENTER MIAMI FL       EP992026\n"
        "TROPICAL DEPRESSION CENTER LOCATED NEAR 12.0N 100.0W AT 09/1500Z\n"
        "MAX SUSTAINED WINDS 25 KT WITH GUSTS TO 35 KT.\n"
        "FORECAST VALID 10/0000Z 13.0N 101.0W\n"
        "MAX WIND 30 KT...GUSTS 40 KT.\n"
    )
    radii = _parse_current_center_radii(text_no_radii)
    assert radii == {
        "quadrant_radius_34": None,
        "quadrant_radius_50": None,
        "quadrant_radius_64": None,
    }


def test_current_center_radii_stops_at_block_end_without_marker():
    """Backstop: even if the FORECAST VALID/OUTLOOK VALID header were reworded,
    the parser stops at the end of the first contiguous KT block, so a later
    forecast-hour KT line can't overwrite the analysis radii."""
    text = (
        "NWS NATIONAL HURRICANE CENTER MIAMI FL       AL012026\n"
        "MAX SUSTAINED WINDS 105 KT WITH GUSTS TO 120 KT.\n"
        "64 KT....... 40NE  35SE  30SW  40NW.\n"
        "50 KT....... 90NE  70SE  50SW  80NW.\n"
        "34 KT.......150NE 140SE 100SW 140NW.\n"
        "12 FT SEAS..120NE 150SE 120SW  90NW.\n"
        # No "FORECAST VALID" marker here — simulate a header reword.
        "SOMETHING ELSE 11/0600Z 25.0N 70.0W\n"
        "MAX WIND 100 KT...GUSTS 120 KT.\n"
        "64 KT....... 50NE  45SE  40SW  50NW.\n"
    )
    radii = _parse_current_center_radii(text)
    # Analysis value kept; the later 50NE forecast-hour radius did not leak in.
    assert radii["quadrant_radius_64"] == [40, 35, 30, 40]


def test_current_center_radii_all_zero_kept():
    """An all-zero analysis line is kept as [0, 0, 0, 0], not collapsed to
    None, so it remains a usable numeric anchor for downstream interpolation."""
    text = (
        "NWS NATIONAL HURRICANE CENTER MIAMI FL       AL012026\n"
        "MAX SUSTAINED WINDS 65 KT WITH GUSTS TO 80 KT.\n"
        "64 KT.......  0NE   0SE   0SW   0NW.\n"
        "50 KT.......  0NE   0SE   0SW   0NW.\n"
        "34 KT....... 90NE  90SE  60SW  70NW.\n"
        "FORECAST VALID 11/0600Z 25.0N 70.0W\n"
    )
    radii = _parse_current_center_radii(text)
    assert radii["quadrant_radius_64"] == [0, 0, 0, 0]
    assert radii["quadrant_radius_50"] == [0, 0, 0, 0]
    assert radii["quadrant_radius_34"] == [90, 90, 60, 70]


def test_observation_backfills_radii_from_advisory(monkeypatch):
    """End-to-end: the leadtime=0 observation (from CurrentStorms.json, which
    omits radii) is enriched with the advisory's analysis-time wind radii."""
    import ocha_lens.datasources.nhc as nhc

    with open(FIXTURES_PATH / "TCM_example.txt", "r") as f:
        advisory_text = f.read()
    monkeypatch.setattr(
        nhc, "_fetch_forecast_advisory", lambda url: advisory_text
    )

    raw = {
        "activeStorms": [
            {
                "id": "al132023",
                "name": "Lee",
                "lastUpdate": "2023-09-10T21:00:00Z",
                "intensity": "105",
                "pressure": "954",
                "latitudeNumeric": 22.1,
                "longitudeNumeric": -61.7,
                "forecastAdvisory": {
                    "url": "http://example/advisory",
                    "issuance": "2023-09-10T21:00:00Z",
                },
            }
        ]
    }

    df = nhc._process_nhc_to_df(
        raw, include_observations=True, include_forecasts=True
    )
    obs = df[df.leadtime == 0].iloc[0]
    # Backfilled from the advisory analysis block, NOT the first forecast hour
    # (whose 64 KT NE radius is 50, vs the analysis 40).
    assert obs["quadrant_radius_34"] == [150, 140, 100, 140]
    assert obs["quadrant_radius_50"] == [90, 70, 50, 80]
    assert obs["quadrant_radius_64"] == [40, 35, 30, 40]
