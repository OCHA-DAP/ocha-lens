import os
import tempfile
from pathlib import Path
from typing import List, Literal, Optional, Union

import cdsapi
import pandas as pd
import xarray as xr

GLOFAS_DATASET_NAMES = {
    "reanalysis": "cems-glofas-historical",
    "forecast": "cems-glofas-forecast",
    "reforecast": "cems-glofas-reforecast",
}
GLOFAS_HYDROLOGICAL_MODEL = "lisflood"

EWDSAPI_URL = os.getenv("EWDSAPI_URL")
EWDSAPI_KEY = os.getenv("EWDSAPI_KEY")


def download_glofas(
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    area: List[float],
    years: List[int],
    months: List[int],
    days: List[int],
    leadtimes: List[int] = None,
    output_dir: str = None,
    clobber: bool = True,
    consolidated=True,
) -> Union[Path, str]:
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
            request = _reanalysis_request(
                area, years, months, days, consolidated
            )
        elif dataset == "reforecast":
            request = _reforecast_request(area, years, months, days, leadtimes)
        elif dataset == "forecast":
            request = _forecast_request(area, years, months, days, leadtimes)

        c = cdsapi.Client(url=EWDSAPI_URL, key=EWDSAPI_KEY)
        response = c.retrieve(GLOFAS_DATASET_NAMES[dataset], request)
        # TODO: Also handle if output is to Azure
        os.makedirs(output_dir, exist_ok=True)
        response.download(output_f)
        return output_f


def load_glofas(
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    area: List[float],  # [N, W, S, E]
    years: List[int],
    months: List[int],
    days: List[int],
    leadtimes: List[int] = None,
    output_dir: Optional[str] = None,
    use_cache: bool = False,
    consolidated=False,
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
    if use_cache and output_dir is None:
        raise ValueError("Must specify output_dir when use_cache=True")

    f = _get_file_name(dataset, area, years, months, days, leadtimes)

    if use_cache:
        file_path = Path(output_dir) / f
        if file_path.exists():
            return xr.open_dataset(
                file_path,
                engine="cfgrib",
                decode_timedelta=True,
                backend_kwargs={"indexpath": ""},
            )

    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    file_path = download_glofas(
        dataset,
        area,
        years,
        months,
        days,
        leadtimes,
        output_dir,
        clobber=not use_cache,
        consolidated=consolidated,
    )
    return xr.open_dataset(
        file_path,
        engine="cfgrib",
        decode_timedelta=True,
        backend_kwargs={"indexpath": ""},
    )


def get_discharge(ds: xr.Dataset) -> pd.DataFrame:
    return ds.to_dataframe().reset_index()


def _reanalysis_request(area, years, months, days, consolidated=True):
    product_type = "consolidated" if consolidated else "intermediate"
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": [GLOFAS_HYDROLOGICAL_MODEL],
        "product_type": [product_type],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "area": area,
        "hyear": [str(y) for y in years],
        "hmonth": [str(m).zfill(2) for m in months],
        "hday": [str(d).zfill(2) for d in days],
        "format": "grib2",
        "download_format": "unarchived",
    }


def _reforecast_request(area, years, months, days, leadtimes=None):
    leadtimes = [i for i in range(24, 744, 24)] if not leadtimes else leadtimes
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": [GLOFAS_HYDROLOGICAL_MODEL],
        "product_type": ["ensemble_perturbed_reforecast"],
        "variable": "river_discharge_in_the_last_24_hours",
        "area": area,
        "hyear": [str(y) for y in years],
        "hmonth": [str(m).zfill(2) for m in months],
        "hday": [str(d).zfill(2) for d in days],
        "leadtime_hour": [str(lt) for lt in leadtimes],
        "format": "grib2",
        "download_format": "unarchived",
    }


def _forecast_request(area, years, months, days, leadtimes=None):
    leadtimes = [i for i in range(24, 744, 24)] if not leadtimes else leadtimes
    return {
        "system_version": ["operational"],
        "hydrological_model": [GLOFAS_HYDROLOGICAL_MODEL],
        "product_type": ["ensemble_perturbed_forecasts"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "area": area,
        "year": [str(y) for y in years],
        "month": [str(m).zfill(2) for m in months],
        "day": [str(d).zfill(2) for d in days],
        "leadtime_hour": [str(lt) for lt in leadtimes],
        "format": "grib2",
        "download_format": "unarchived",
    }


# TODO
def _get_file_name(
    dataset: Literal["reanalysis", "reforecast", "forecast"],
    area,
    years,
    months,
    days,
    leadtimes,
):
    return f"glofas-{dataset}-somenametbd.grib"
