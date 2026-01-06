from pathlib import Path

from ocha_lens.datasources.nhc import (
    _parse_atcf_adeck,
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
    assert len(df_tracks[df_tracks.forecast_type == "observation"]) == 41


def test_adeck_hilary():
    """Test archival A-Deck parsing: ep092023 is storm HILARY."""
    df_raw = _parse_atcf_adeck(FIXTURES_PATH / "aep092023.dat")
    df_storms = get_storms(df_raw)
    df_tracks = get_tracks(df_raw)

    assert len(df_storms) == 1
    assert df_storms.name[0] == "HILARY"
    assert len(df_tracks) == 210
    assert len(df_tracks.storm_id.unique()) == 1
    assert len(df_tracks[df_tracks.forecast_type == "observation"]) == 26


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
