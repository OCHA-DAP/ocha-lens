from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocha_lens.datasources.ibtracs import (
    _mask_invalid_gusts,
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


def test_mask_invalid_gusts_only_touches_out_of_range_gusts():
    """Gust values outside the declared valid range become NaN; nothing else
    changes"""
    dims = ("storm", "date_time")
    ds = xr.Dataset(
        {
            "foo_gust": (
                dims,
                [[100.0, 999.0, np.nan]],
                {"valid_min": 1, "valid_max": 350},
            ),
            "bar_gust": (dims, [[100.0, 999.0, np.nan]]),
            "foo_wind": (
                dims,
                [[100.0, 999.0, -1.0]],
                {"valid_min": 1, "valid_max": 250},
            ),
        }
    )
    result = _mask_invalid_gusts(ds)
    np.testing.assert_array_equal(
        result["foo_gust"].values, [[100.0, np.nan, np.nan]]
    )
    # No declared range: left as is
    np.testing.assert_array_equal(
        result["bar_gust"].values, [[100.0, 999.0, np.nan]]
    )
    # Not a gust variable: left as is
    np.testing.assert_array_equal(
        result["foo_wind"].values, [[100.0, 999.0, -1.0]]
    )
    # The input dataset is not modified
    assert ds["foo_gust"].values[0, 1] == 999.0


def test_get_tracks_masks_gust_sentinel(sample_ibtracs_dataset):
    """A 999 gust sentinel from the WMO agency comes out as a missing
    gust_speed instead of failing schema validation"""
    ds = sample_ibtracs_dataset.copy(deep=True)
    is_main = (ds["track_type"] == b"main").broadcast_like(ds["wmo_agency"])
    candidates = np.argwhere(
        (
            (ds["wmo_agency"] == b"bom") & is_main & ds["bom_gust"].notnull()
        ).values
    )
    assert len(candidates), "fixture has no BoM main-track point with a gust"
    storm_idx, time_idx = candidates[0]
    ds["bom_gust"][storm_idx, time_idx] = 999

    result = get_tracks(ds, track_type="best")

    sid = ds["sid"].values[storm_idx].decode("utf-8")
    valid_time = pd.Timestamp(ds["time"].values[storm_idx, time_idx]).round(
        "min"
    )
    row = result[(result["sid"] == sid) & (result["valid_time"] == valid_time)]
    assert len(row) == 1
    assert pd.isna(row["gust_speed"].iloc[0])
    # Only that one point lost its gust; no rows were dropped
    baseline = get_tracks(sample_ibtracs_dataset, track_type="best")
    assert len(result) == len(baseline)
    assert (
        result["gust_speed"].notna().sum()
        == baseline["gust_speed"].notna().sum() - 1
    )
