import os
from pathlib import Path
from typing import Literal, Optional

import cdsapi
import pandas as pd
import xarray as xr

DATASET_NAMES = {
    "reanalysis": "cems-glofas-reanalysis",
    "forecast": "cems-glofas-forecast",
    "reforecast": "cems-glofas-reforecast",
}


def download_glofas(
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    area,  # coords in list like [N, W, S, E]
    years,  # eg. [2020, 2021, 2022]
    months,  # eg. [1, 2, 3]
    days,  # eg. [1, 2, 3, 4, 5]
    leadtimes,  # eg. [24, 48, 72, 96] (must be hours in day increments)
    output_dir: Optional[str],  # May also be an Azure blob client
    clobber: bool = True,
) -> Path:  # Or Azure blob client?
    """
    Downloads glofas data from cds api.
    1. clobber == True, overwrite if existing file is present
    2. clobber == False, skip download and don't overwrite file
    """
    # Configure where output file should be
    # TODO: Need to also handle if output_dir is an AzureBlobClient
    f = _get_file_name(dataset, area, years, months, days, leadtimes)
    output_dir = Path(output_dir)
    output_f = output_dir / f

    # Check if the file exists
    if not clobber:
        if output_f.exists():
            return output_f
    # Otherwise go ahead and download
    else:
        if dataset == "reanalysis":
            request = _reanalysis_request(area, years, months, days, leadtimes)
        elif dataset == "reforecast":
            request = _reforecast_request(area, years, months, days, leadtimes)
        elif dataset == "forecast":
            request = _forecast_request(area, years, months, days, leadtimes)

        c = cdsapi.Client()
        response = c.retrieve(DATASET_NAMES["dataset"], request)
        # TODO: Also handle if output is to Azure
        os.makedirs(output_dir, exist_ok=True)
        response.download(output_f)
        return output_f


def load_glofas(
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    area,  # [N, W, S, E]
    years,
    months,
    days,
    leadtimes,
    output_dir: Optional[str] = None,
    use_cache: bool = False,
) -> xr.Dataset:
    """
    Loads glofas data into memory. May download from cds api or get from existing cache.
    1. output_dir == None, use_cache == False: Download data to a temporary directory
    2. output_dir != None, use_cache == False: Download data to specific directory (can be Blob). Generate grib
        filename from input params.
    3. output_dir != None, use_cache == True: Load cached data from specific directory (can be Azure Blob).
        Generate grib filename from input params. Go to 2. if the file doesn't exist.
    4. output_dir == None, use_cache == True: Invalid. Return error telling user to specify an output_dir.
    """
    if use_cache:
        f = _get_file_name(dataset, area, years, months, days, leadtimes)
        dir = Path(output_dir) / f
        return xr.open_dataset(
            dir,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs={"indexpath": ""},
        )
    else:
        print("!")

    return


def get_discharge(ds: xr.Dataset) -> pd.DataFrame:
    # TODO: Transform xarray into pandas dataframe
    return


# ---------


def _reanalysis_request():
    return


def _reforecast_request():
    return


def _forecast_request():
    return


def _get_file_name(
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    area,
    years,
    months,
    days,
    leadtimes,
):
    return f"glofas-{dataset}-somenametbd.grib"
