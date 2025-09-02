from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from ocha_lens.datasources.ibtracs import (
    get_storms,
    get_tracks,
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
        "tracks": get_tracks(sample_ibtracs_dataset),
    }


# Tests for get_provisional_tracks function
def test_get_tracks_returns_dataframe(processed_ibtracs_data):
    """Test that get_provisional_tracks returns a pandas DataFrame"""
    result = processed_ibtracs_data["tracks"]
    assert isinstance(result, pd.DataFrame)
    expected_output = 1242
    assert len(result) == expected_output, (
        f"Output data has incorrect number of rows. Expected {expected_output} and got {len(result)}"
    )


# Tests for get_storms function
def test_get_storms_returns_dataframe(processed_ibtracs_data):
    """Test that get_storms returns a pandas DataFrame"""
    result = processed_ibtracs_data["storms"]
    assert isinstance(result, pd.DataFrame)
    expected_output = 50
    assert len(result) == expected_output, (
        f"Output data has incorrect number of rows. Expected {expected_output} and got {len(result)}"
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
    """Test that get_storms assigns unique storm_id to each named storm"""
    result = processed_ibtracs_data["storms"]
    assert result["storm_id"].nunique() == 39
