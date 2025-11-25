import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Calculating daily flood exposure from Floodscan and WorldPop

    The notebook demonstrates a basic flood exposure calculation method, such as is implemented in [this flood exposure application](https://chd-ds-floodexposure-monitoring.azurewebsites.net/).

    We:

    1. Stack Floodscan COGs across our date range of interest
    2. Filter out flooding that is <5% of a pixel
    3. Load WorldPop's 2025 UN-adjusted 1km population estimates
    4. Multiply the Floodscan flooded fraction by WorldPop's population
    5. Aggregate across admin units by taking the sum
    """
    )
    return


@app.cell
def _():
    # Installed from 0.1.7.dev3
    import datetime

    import exactextract as ee
    import matplotlib.pyplot as plt
    import ocha_stratus as stratus
    import pandas as pd
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    ISO3 = "SSD"
    ADM_LEVEL = 1
    DATES = [
        datetime.date(2025, 10, day).strftime("%Y-%m-%d")
        for day in range(1, 32)
    ]
    return ADM_LEVEL, DATES, ISO3, ee, pd, plt, stratus


@app.cell
def _(ADM_LEVEL, DATES, ISO3, stratus):
    # 1. Load data
    gdf = stratus.codab.load_codab_from_fieldmaps(ISO3, ADM_LEVEL)
    da_pop = stratus.datasources.worldpop.load_worldpop_from_stac(
        iso3=ISO3, year=2025, resolution="1km"
    )
    da_flood = stratus.stack_cogs("floodscan", dates=DATES, clip_gdf=gdf)

    gdf_fixed = gdf.set_crs("EPSG:4326", allow_override=True)
    return da_flood, da_pop, gdf, gdf_fixed


@app.cell
def _(ADM_LEVEL, da_flood, da_pop, ee, gdf_fixed):
    # 2. Calculate exposure
    da_flood_sfed = da_flood.sel(band=1)
    da_flood_sfed = da_flood_sfed.where(da_flood_sfed >= 0.05)
    da_flood_sfed_matched = da_flood_sfed.rio.reproject_match(da_pop)
    da_exposure = da_flood_sfed_matched * da_pop

    # 3. Calculate raster stats
    df_stats = ee.exact_extract(
        da_exposure,
        gdf_fixed,
        ["sum"],
        include_cols=[f"adm{ADM_LEVEL}_src"],
        output="pandas",
    )
    return da_exposure, df_stats


@app.cell
def _(ADM_LEVEL, da_exposure, df_stats, pd):
    # 4. Clean dataframe
    dates = da_exposure.date.values
    df_long = df_stats.melt(
        id_vars=[f"adm{ADM_LEVEL}_src"], var_name="band", value_name="sum"
    )
    df_long["band_num"] = (
        df_long["band"].str.extract(r"band_(\d+)_sum").astype(int)
    )
    df_long["date"] = df_long["band_num"].map(lambda i: dates[i - 1])
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long[[f"adm{ADM_LEVEL}_src", "date", "sum"]].sort_values(
        [f"adm{ADM_LEVEL}_src", "date"]
    )
    return (df_long,)


@app.cell
def _(da_exposure, gdf, plt):
    fig, ax = plt.subplots()
    da_exposure.isel(date=1).plot(ax=ax)
    gdf.plot(ax=ax, facecolor="none", edgecolor="black")
    fig
    return


@app.cell
def _(df_long):
    df_long
    return


if __name__ == "__main__":
    app.run()
