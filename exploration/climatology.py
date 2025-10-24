import marimo

__generated_with = "0.15.2"
app = marimo.App(app_title="SEAS5 Tercile Probabilities")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Computing Tercile Probabilities from ECMWF SEAS5 forecasts

    This notebook is adapted from this [ECMWF documentation](https://ecmwf-projects.github.io/c3s-seasonal-forecasts/workflows/graphical_product_example_tmin.html#produce-tercile-summary). We walk though how one might calculate tercile probabilities from SEAS5 forecasts and calculate summary stats across administrative boundaries.
    """
    )
    return


@app.cell
def _():
    from datetime import datetime

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import earthkit.data
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import rioxarray  # noqa
    import xarray as xr
    from dateutil.relativedelta import relativedelta
    from exactextract import exact_extract
    from fsspec.implementations.http import HTTPFileSystem

    return (
        HTTPFileSystem,
        ccrs,
        cfeature,
        datetime,
        earthkit,
        exact_extract,
        gpd,
        np,
        pd,
        plt,
        relativedelta,
        xr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Calculate climatology

    **Question**: What time period should we use to calculate this climatology? Should it be static or should it be rolling? We've pulled in data from 1993-2016, following [this example](https://ecmwf-projects.github.io/c3s-seasonal-forecasts/workflows/graphical_product_example_tmin.html#load-forecast-and-hindcast-data)."
    """
    )
    return


@app.cell
def _():
    variable = ["total_precipitation"]
    product = ["monthly_mean"]
    dataset = "seasonal-monthly-single-levels"
    use_cache = True
    cache_path = "/Users/hannahker/Downloads/hindcast.grib"
    return cache_path, dataset, product, use_cache, variable


@app.cell
def _(cache_path, dataset, earthkit, product, use_cache, variable, xr):
    if use_cache:
        hcst = xr.open_dataset(
            cache_path,
            engine="cfgrib",
            chunks="auto",
            backend_kwargs=dict(
                time_dims=("time", "forecastMonth"), indexpath=""
            ),
        )
    else:
        _request = {
            "originating_centre": "ecmwf",
            "system": "51",
            "variable": variable,
            "product_type": product,
            "year": [
                "1993",
                "1994",
                "1995",
                "1996",
                "1997",
                "1998",
                "1999",
                "2000",
                "2001",
                "2002",
                "2003",
                "2004",
                "2005",
                "2006",
                "2007",
                "2008",
                "2009",
                "2010",
                "2011",
                "2012",
                "2013",
                "2014",
                "2015",
                "2016",
            ],
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "leadtime_month": ["1", "2", "3", "4", "5", "6"],
            "data_format": "grib",
        }
        hcst = earthkit.data.from_source("cds", dataset, _request).to_xarray(
            engine="cfgrib",
            backend_kwargs=dict(
                time_dims=("time", "forecastMonth"), indexpath=""
            ),
        )
    return (hcst,)


@app.cell
def _(hcst):
    # Rename and convert longitude to [-180, 180]
    hcst_ = hcst.rename(
        {"time": "start_date", "longitude": "lon", "latitude": "lat"}
    )
    hcst_ = hcst_.assign_coords(lon=(((hcst_.lon + 180) % 360) - 180)).sortby(
        "lon"
    )
    hcst_ = hcst_.tprate
    return (hcst_,)


@app.cell
def _(hcst_):
    # Compute quantiles
    quantiles = [1.0 / 3.0, 2.0 / 3.0]
    hcst_qbnds = hcst_.quantile(quantiles, ["start_date", "number"])
    return (hcst_qbnds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 2. Download recent forecast""")
    return


@app.cell
def _(dataset, datetime, earthkit, product, variable, xr):
    use_cache_1 = True
    cache_path_1 = "/Users/hannahker/Downloads/forecast.grib"
    now = datetime.now()
    current_month = now.month
    current_year = now.year
    if use_cache_1:
        fcst = xr.open_dataset(
            cache_path_1,
            engine="cfgrib",
            chunks="auto",
            backend_kwargs=dict(
                time_dims=("time", "forecastMonth"),
                indexpath="",
                filter_by_keys={"dataType": "fcmean"},
            ),
        )
    else:
        _request = {
            "originating_centre": "ecmwf",
            "system": "51",
            "variable": [variable],
            "product_type": [product],
            "year": [str(current_year)],
            "month": [str(current_month).zfill(0)],
            "leadtime_month": ["1", "2", "3", "4", "5", "6"],
            "data_format": "grib",
        }
        fcst = earthkit.data.from_source("cds", dataset, _request).to_xarray(
            engine="cfgrib",
            backend_kwargs=dict(
                time_dims=("time", "forecastMonth"), indexpath=""
            ),
        )
    return (fcst,)


@app.cell
def _(fcst, pd, relativedelta):
    fcst_1 = fcst.rename(
        {"time": "start_date", "longitude": "lon", "latitude": "lat"}
    )
    fcst_1 = fcst_1.assign_coords(lon=(fcst_1.lon + 180) % 360 - 180).sortby(
        "lon"
    )
    fcst_1 = fcst_1.tprate
    _valid_time = [
        pd.to_datetime(fcst_1.start_date.values)
        + relativedelta(months=fcmonth - 1)
        for fcmonth in fcst_1.forecastMonth
    ]
    fcst_1 = fcst_1.assign_coords(valid_time=("forecastMonth", _valid_time))
    return (fcst_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Calculate tercile summary from hindcasts

    We calculate the tercile summary by taking the percentage of ensemble members in each tercile range.
    """
    )
    return


@app.cell
def _(fcst_1, hcst_qbnds, np):
    fc_ens = fcst_1.sizes["number"]
    above = fcst_1.where(fcst_1 > hcst_qbnds.sel(quantile=2.0 / 3.0)).count(
        "number"
    )
    below = fcst_1.where(fcst_1 < hcst_qbnds.sel(quantile=1.0 / 3.0)).count(
        "number"
    )
    P_above = 100.0 * (above / float(fc_ens))
    P_below = 100.0 * (below / float(fc_ens))
    P_normal = 100.0 - (P_above + P_below)
    a = P_above.where(P_above > np.maximum(40.0 + 0 * P_above, P_below), 0.0)
    b = P_below.where(P_below > np.maximum(40.0 + 0 * P_below, P_above), 0.0)
    P_summary = a - b
    return P_above, P_below, P_normal, P_summary


@app.cell
def _(P_summary, ccrs, cfeature, fcst_1, pd, plt):
    contour_levels = [
        -100.0,
        -70.0,
        -60.0,
        -50.0,
        -40.0,
        40.0,
        50.0,
        60.0,
        70.0,
        100.0,
    ]
    contour_colours = [
        "navy",
        "blue",
        "deepskyblue",
        "cyan",
        "white",
        "yellow",
        "orange",
        "orangered",
        "tab:red",
    ]
    center = "ecmwf"
    version = "51"
    var_str = "tprate"
    start_date = pd.Timestamp(fcst_1.start_date.values)
    lt_mons = [1]
    for ltm in lt_mons:
        lt_str = str(ltm).zfill(2)
        plot_data = P_summary.sel(forecastMonth=ltm)
        plot_data = plot_data.clip(min=-99, max=99)
        _valid_time = pd.to_datetime(plot_data.valid_time.values)
        vm_str = _valid_time.strftime("%b")
        title_txt1 = f"{center} system={version}, Probability most likely category ({var_str})"
        title_txt2 = f"start date = {start_date.year}/{start_date.month}, valid month: {vm_str} (leadtime_month = {lt_str})"
        _fig = plt.figure(figsize=(16, 8))
        _ax = plt.axes(projection=ccrs.PlateCarree())
        _ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
        _ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=2.0)
        plot_data.plot(
            levels=contour_levels,
            colors=contour_colours,
            cbar_kwargs={"fraction": 0.033, "extend": "neither"},
        )
        plt.title(title_txt1 + "\n" + title_txt2)
        plt.tight_layout()

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Take raster stats per admin level

    We can summarize the tercile probability per administrative level as shown below. For each tercile, we calculate the average probability across a given admin boundary. We can highlight the "dominant" tercile per admin boundary by identifying the one that has the most likely probability (given that it is above a 40% threshold).
    """
    )
    return


@app.cell
def _(HTTPFileSystem, gpd):
    GLOBAL_ADM1 = "https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm1_polygons.parquet"
    ISO3 = "NGA"

    filesystem = HTTPFileSystem()
    filters = [("iso_3", "=", ISO3)]
    gdf = gpd.read_parquet(GLOBAL_ADM1, filesystem=filesystem, filters=filters)
    return (gdf,)


@app.cell
def _(P_above, P_below, P_normal, gdf):
    # Clip all rasters to the geodataframe and select a single forecast month
    above_clipped = (
        P_above.rio.write_crs("EPSG:4326")
        .rio.set_spatial_dims("lon", "lat")
        .rio.clip(gdf.geometry)
    )
    above_clipped = above_clipped.sel(forecastMonth=1).compute()

    below_clipped = (
        P_below.rio.write_crs("EPSG:4326")
        .rio.set_spatial_dims("lon", "lat")
        .rio.clip(gdf.geometry)
    )
    below_clipped = below_clipped.sel(forecastMonth=1).compute()

    norm_clipped = (
        P_normal.rio.write_crs("EPSG:4326")
        .rio.set_spatial_dims("lon", "lat")
        .rio.clip(gdf.geometry)
    )
    norm_clipped = norm_clipped.sel(forecastMonth=1).compute()
    return above_clipped, below_clipped, norm_clipped


@app.cell
def _(above_clipped, below_clipped, gdf, norm_clipped, plt):
    _fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for _ax, (data, title) in zip(
        axes,
        [
            (below_clipped, "Below Normal"),
            (norm_clipped, "Near Normal"),
            (above_clipped, "Above Normal"),
        ],
    ):
        data.plot(ax=_ax, cmap="Reds", vmin=0, vmax=100)
        gdf.boundary.plot(ax=_ax, color="black", linewidth=1.5)
        _ax.set_title(f"Probability: {title} (%)")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(above_clipped, below_clipped, exact_extract, gdf, norm_clipped):
    gdf_output = gdf[["adm1_src", "adm1_name", "geometry"]]

    df_b = exact_extract(
        below_clipped,
        gdf_output,
        "mean",
        output="pandas",
        include_cols=["adm1_src", "adm1_name"],
    )
    df_b.rename(columns={"mean": "mean_prob_below"}, inplace=True)
    gdf_output = gdf_output.merge(df_b)

    df_a = exact_extract(
        above_clipped,
        gdf_output,
        "mean",
        output="pandas",
        include_cols=["adm1_src", "adm1_name"],
    )
    df_a.rename(columns={"mean": "mean_prob_above"}, inplace=True)
    gdf_output = gdf_output.merge(df_a)

    df_n = exact_extract(
        norm_clipped,
        gdf_output,
        "mean",
        output="pandas",
        include_cols=["adm1_src", "adm1_name"],
    )
    df_n.rename(columns={"mean": "mean_prob_norm"}, inplace=True)
    gdf_output = gdf_output.merge(df_n)
    return (gdf_output,)


@app.cell
def _(gdf_output):
    # Add dominant category column
    gdf_output["dominant_category"] = (
        gdf_output[["mean_prob_below", "mean_prob_norm", "mean_prob_above"]]
        .idxmax(axis=1)
        .str.replace("mean_prob_", "")
    )

    # Add probability of dominant category
    gdf_output["dominant_probability"] = gdf_output[
        ["mean_prob_below", "mean_prob_norm", "mean_prob_above"]
    ].max(axis=1)
    return


@app.cell
def _(gdf_output, np):
    # Calculate max probability and dominant category for all rows
    max_prob = gdf_output[
        ["mean_prob_below", "mean_prob_norm", "mean_prob_above"]
    ].max(axis=1)
    max_category = gdf_output[
        ["mean_prob_below", "mean_prob_norm", "mean_prob_above"]
    ].idxmax(axis=1)

    # Apply 40% threshold
    gdf_output["dominant_category"] = np.where(
        max_prob >= 40, max_category.str.replace("mean_prob_", ""), np.nan
    )

    gdf_output["dominant_probability"] = np.where(
        max_prob >= 40, max_prob.round(1), np.nan
    )
    return


@app.cell
def _(gdf_output):
    gdf_output.drop("geometry", axis=1)
    return


if __name__ == "__main__":
    app.run()
