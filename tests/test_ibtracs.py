from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from ocha_lens.datasources.ibtracs import (
    IBTRACS_CONFIG,
    get_best_tracks,
    get_provisional_tracks,
    get_storms,
)

# Path to test data
TEST_DATA_PATH = Path(__file__).parent / "fixtures" / "sample_small_ibtracs.nc"


@pytest.fixture(scope="session")
def sample_ibtracs_dataset():
    """
    Load a small sample IBTrACS dataset for testing.

    Uses session scope to load data only once for all tests.
    """
    if not TEST_DATA_PATH.exists():
        pytest.skip(f"Test data file not found: {TEST_DATA_PATH}")

    return xr.open_dataset(TEST_DATA_PATH)


@pytest.fixture(scope="session")
def processed_ibtracs_data(sample_ibtracs_dataset):
    """
    Process IBTrACS data once and cache results for all tests.

    Returns a dictionary with all processed dataframes.
    """
    return {
        "storms": get_storms(sample_ibtracs_dataset),
        "provisional": get_provisional_tracks(sample_ibtracs_dataset),
        "best": get_best_tracks(sample_ibtracs_dataset),
    }


# Tests for get_provisional_tracks function
def test_get_provisional_tracks_returns_dataframe(processed_ibtracs_data):
    """Test that get_provisional_tracks returns a pandas DataFrame"""
    result = processed_ibtracs_data["provisional"]
    assert isinstance(result, pd.DataFrame)
    expected_output = 294
    assert len(result) == expected_output, (
        f"Output data has incorrect number of rows. Expected {expected_output} and got {len(result)}"
    )


def test_get_provisional_tracks_has_expected_columns(processed_ibtracs_data):
    """Test that get_provisional_tracks result has all expected columns"""
    result = processed_ibtracs_data["provisional"]
    expected_columns = IBTRACS_CONFIG["tracks"].keys()
    for col in expected_columns:
        assert col in result.columns, f"Missing expected column: {col}"
    # Result has expected data types
    result_types = result.dtypes.astype(str).to_dict()
    result_types = dict(sorted(result_types.items()))
    expected_types = IBTRACS_CONFIG["tracks"]
    expected_types = dict(sorted(expected_types.items()))
    assert result_types == expected_types, (
        "Output data types don't match expected"
    )


def test_get_provisional_tracks_has_coordinates(processed_ibtracs_data):
    """Test that get_best_tracks includes valid coordinates"""
    result = processed_ibtracs_data["provisional"]

    if len(result) > 0:
        # No missing coordinates should remain
        assert not result["latitude"].isna().any()
        assert not result["longitude"].isna().any()

        # Coordinates should be reasonable
        assert result["latitude"].between(-90, 90).all()
        assert result["longitude"].between(-180, 180).all()


# Tests for get_best_tracks function
def test_get_best_tracks_returns_dataframe(processed_ibtracs_data):
    """Test that get_best_tracks returns a pandas DataFrame"""
    result = processed_ibtracs_data["best"]
    assert isinstance(result, pd.DataFrame)
    expected_output = 1091
    assert len(result) == expected_output, (
        f"Output data has incorrect number of rows. Expected {expected_output} and got {len(result)}"
    )


def test_get_best_tracks_has_expected_columns(processed_ibtracs_data):
    """Test that get_best_tracks result has all expected columns"""
    result = processed_ibtracs_data["best"]
    expected_columns = IBTRACS_CONFIG["tracks"].keys()
    for col in expected_columns:
        assert col in result.columns, f"Missing expected column: {col}"
    # Result has expected data types
    result_types = result.dtypes.astype(str).to_dict()
    result_types = dict(sorted(result_types.items()))
    expected_types = IBTRACS_CONFIG["tracks"]
    expected_types = dict(sorted(expected_types.items()))
    assert result_types == expected_types, (
        "Output data types don't match expected"
    )


def test_get_best_tracks_has_coordinates(processed_ibtracs_data):
    """Test that get_best_tracks includes valid coordinates"""
    result = processed_ibtracs_data["best"]

    if len(result) > 0:
        # No missing coordinates should remain
        assert not result["latitude"].isna().any()
        assert not result["longitude"].isna().any()

        # Coordinates should be reasonable
        assert result["latitude"].between(-90, 90).all()
        assert result["longitude"].between(-180, 180).all()


# Tests for get_storms function
def test_get_storms_returns_dataframe(processed_ibtracs_data):
    """Test that get_storms returns a pandas DataFrame"""
    result = processed_ibtracs_data["storms"]
    assert isinstance(result, pd.DataFrame)
    expected_output = 50
    assert len(result) == expected_output, (
        f"Output data has incorrect number of rows. Expected {expected_output} and got {len(result)}"
    )


def test_get_storms_has_expected_columns(processed_ibtracs_data):
    """Test that get_storms result has all expected columns"""
    result = processed_ibtracs_data["storms"]
    expected_columns = IBTRACS_CONFIG["storms"].keys()
    for col in expected_columns:
        assert col in result.columns, f"Missing expected column: {col}"
    # Result has expected data types
    result_types = result.dtypes.astype(str).to_dict()
    result_types = dict(sorted(result_types.items()))
    expected_types = IBTRACS_CONFIG["storms"]
    expected_types = dict(sorted(expected_types.items()))
    assert result_types == expected_types, (
        "Output data types don't match expected"
    )


def test_get_storms_one_row_per_storm(
    processed_ibtracs_data, sample_ibtracs_dataset
):
    """Test that get_storms returns exactly one row per storm"""
    result = processed_ibtracs_data["storms"]
    # Should have same number of storms as in dataset
    expected_storms = len(sample_ibtracs_dataset.storm)
    assert len(result) == expected_storms
    # All storm IDs should be unique
    assert len(result["sid"].unique()) == len(result)


def test_get_storms_storm_id_is_unique(processed_ibtracs_data):
    """Test that get_storms assigns unique storm_id to each storm"""
    result = processed_ibtracs_data["storms"]
    # All storm_id values should be unique
    assert len(result["storm_id"].unique()) == len(result)
    # All storm_id values should be non-empty strings
    assert all(
        isinstance(sid, str) and len(sid) > 0 for sid in result["storm_id"]
    )
