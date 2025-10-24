import marimo

__generated_with = "0.15.2"
app = marimo.App(app_title="ERA5 Rainy Season")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Rainy Season Definition from ERA5 Rainfall Patterns

    This is a POC analysis to identify distinct rainfall seasonality patterns across a selected country using gridded monthly precipitation data from ECMWF's ERA5 reanalysis dataset (1993-2016).

    ## 1. K-means clustering

    For each grid cell, we compute a climatological seasonal cycle by averaging monthly rainfall across all years, then normalize these patterns to focus on the timing rather than magnitude of rainfall peaks. This normalization ensures that clustering is driven by seasonal patterns rather than absolute precipitation amounts. We then apply k-means clustering to group grid cells with similar seasonal cycles. Use the dropdown below to define the target number of clusters.
    """
    )
    return


@app.cell
def _():
    from datetime import date

    import numpy as np
    import ocha_stratus as stratus
    import plotly.express as px
    import plotly.graph_objects as go
    import xarray as xr
    from dotenv import find_dotenv, load_dotenv
    from exactextract import exact_extract
    from plotly.subplots import make_subplots
    from scipy.fft import fft
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    _ = load_dotenv(find_dotenv(usecwd=True))
    return (
        KMeans,
        MinMaxScaler,
        date,
        exact_extract,
        fft,
        go,
        make_subplots,
        np,
        px,
        stratus,
        xr,
    )


@app.cell
def _(mo):
    iso3_dropdown = mo.ui.dropdown(
        options=["eth", "som", "bfa", "nga", "ssd"], label="ISO3", value="eth"
    )
    zones_dropdown = mo.ui.dropdown(
        options=[1, 2, 3, 4, 5], label="Target number of zones", value=4
    )

    mo.hstack([iso3_dropdown, zones_dropdown], justify="start")
    return iso3_dropdown, zones_dropdown


@app.cell
def _(date, get_codab, iso3_dropdown, zones_dropdown):
    CLIM_START = 1993
    CLIM_END = 2016
    ISO3 = iso3_dropdown.value
    N_ZONES = zones_dropdown.value

    DATES = [
        date(year, month, 1).strftime("%Y-%m-%d")
        for year in range(CLIM_START, CLIM_END + 1)
        for month in range(1, 13)
    ]

    # Define your color mapping
    cluster_colors = {
        0: "#007ce0",
        1: "#f2645a",
        2: "#1ebfb3",
        3: "#888888",
        4: "#9467bd",
    }

    gdf_0 = get_codab(ISO3, 0)
    return DATES, ISO3, N_ZONES, cluster_colors, gdf_0


@app.cell
def _(mo, stratus):
    @mo.persistent_cache
    def get_cogs(dates, gdf):
        return stratus.stack_cogs("era5", dates, clip_gdf=gdf)

    @mo.persistent_cache
    def get_codab(iso3, adm_level):
        return stratus.codab.load_codab_from_blob(iso3, adm_level)

    return get_codab, get_cogs


@app.cell
def _(DATES, gdf_0, get_cogs):
    da = get_cogs(DATES, gdf_0)
    return (da,)


@app.cell
def _(KMeans, MinMaxScaler, N_ZONES, da, np):
    # 1. Calculate monthly climatology (average across years for each month)
    monthly_clim = da.groupby(da.date.str[5:7].astype(int)).mean()

    # 2. Reshape for clustering: (pixels, months)
    clim_2d = monthly_clim.stack(pixel=("y", "x")).T  # (pixels, 12)
    clim_vals = clim_2d.values

    # 3. Remove NaN pixels (ocean/missing data)
    valid = ~np.isnan(clim_vals).any(axis=1)
    clim_clean = clim_vals[valid]

    # 4. Normalize (0-1) per pixel to focus on timing not magnitude
    clim_norm = MinMaxScaler().fit_transform(clim_clean.T).T

    # 5. Cluster similar patterns
    kmeans = KMeans(n_clusters=N_ZONES, random_state=42)
    labels = kmeans.fit_predict(clim_norm)

    # 6. Visualize clusters spatially
    cluster_map = np.full(clim_2d.shape[0], np.nan)
    cluster_map[valid] = labels
    cluster_map = cluster_map.reshape(len(da.y), len(da.x))
    return clim_norm, cluster_map, kmeans, labels, monthly_clim


@app.cell
def _(N_ZONES, clim_norm, cluster_colors, go, labels, np):
    # Calculate standard deviation or percentiles for each cluster
    cluster_stats = {}

    for i in range(N_ZONES):
        # Get all pixels belonging to this cluster
        mask = labels == i
        cluster_data = clim_norm[mask]  # shape: (n_pixels_in_cluster, 12)

        cluster_stats[i] = {
            "mean": cluster_data.mean(axis=0),
            "p25": np.percentile(cluster_data, 25, axis=0),
            "p75": np.percentile(cluster_data, 75, axis=0),
        }

    # Plot with optional uncertainty bands
    fig_lines = go.Figure()

    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for i in range(N_ZONES):
        stats = cluster_stats[i]

        fig_lines.add_trace(
            go.Scatter(
                x=months + months[::-1],  # x, then x reversed
                y=list(stats["p75"])
                + list(stats["p25"][::-1]),  # upper, then lower reversed
                fill="toself",
                fillcolor=cluster_colors[i],
                opacity=0.1,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add mean line
        fig_lines.add_trace(
            go.Scatter(
                x=months,
                y=stats["mean"],
                mode="lines",
                name=f"Cluster {i}",
                line=dict(color=cluster_colors[i], width=2),
            )
        )

    fig_lines.update_layout(
        title="Monthly Rainfall Patterns by Cluster<br><sup>Cluster mean, with 25th-75th percentile band</sup>",
        xaxis_title="",
        yaxis_title="Normalized Rainfall",
        template="simple_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        margin=dict(l=0, r=0, t=40, b=50),
        hovermode="x unified",
        height=300,
    )
    return (months,)


@app.cell
def _(mo):
    mo.md(
        r"""Each resulting cluster represents a distinct rainfall regime, characterized by its mean monthly pattern and interquartile range (25th-75th percentiles) to capture within-cluster variability. We can see on the map below how these clusters may be distributed spatially across the given country."""
    )
    return


@app.cell
def _(N_ZONES, cluster_colors, cluster_map, go, np):
    # Create colorscale based on N_ZONES
    if N_ZONES == 1:
        colorscale = [[0, cluster_colors[0]], [1, cluster_colors[0]]]
    else:
        colorscale = [
            [i / (N_ZONES - 1), cluster_colors[i]] for i in range(N_ZONES)
        ]

    # Update the map with custom colors
    fig_map = go.Figure(
        data=go.Heatmap(
            z=np.flipud(cluster_map),  # Flip vertically
            colorscale=colorscale,
            showscale=False,
            hovertemplate="Cluster: %{z}<extra></extra>",
        )
    )

    fig_map.update_layout(
        title="Rainfall Pattern Clusters",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        template="simple_white",
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
    )

    fig_map.update_traces(hovertemplate="Cluster: %{z}<extra></extra>")
    return


@app.cell
def _(mo):
    mo.md(
        r"""From here we can transform the data to operationally-suited outputs. We assign a cluster to an administrative boundary based on the majority cluster of overlapping grid cells."""
    )
    return


@app.cell
def _(mo):
    admin_level = mo.ui.dropdown(options=[1, 2], label="Admin level", value=2)

    mo.hstack([admin_level], justify="start")
    return (admin_level,)


@app.cell
def _(ISO3, admin_level, get_codab):
    ADM_LEVEL = admin_level.value

    gdf = get_codab(ISO3, ADM_LEVEL)
    return ADM_LEVEL, gdf


@app.cell
def _(ADM_LEVEL, cluster_map, da, exact_extract, gdf):
    # Convert cluster_map to a proper rioxarray DataArray with spatial info
    cluster_da = da.isel(date=0).copy()  # Use first timestep as template
    cluster_da.values = cluster_map  # Replace with cluster assignments

    gdf_ = exact_extract(
        cluster_da,
        gdf,
        "majority",
        include_cols=[f"ADM{ADM_LEVEL}_EN", f"ADM{ADM_LEVEL}_PCODE"],
        output="pandas",
        include_geom=True,
    )
    gdf_ = gdf_.rename(columns={"majority": "cluster"})
    return (gdf_,)


@app.cell
def _(ADM_LEVEL, cluster_colors, gdf_, px):
    # Convert cluster to string for discrete color mapping
    gdf_["cluster"] = gdf_["cluster"].astype("Int64")
    gdf_["cluster_str"] = gdf_["cluster"].astype(str)
    gdf_["geometry"] = gdf_["geometry"].simplify(tolerance=0.01)

    # Create choropleth map
    fig_admin = px.choropleth_map(
        gdf_,
        geojson=gdf_.geometry,
        locations=gdf_.index,
        color="cluster_str",
        color_discrete_map={str(k): v for k, v in cluster_colors.items()},
        hover_data=[f"ADM{ADM_LEVEL}_PCODE"],
        labels={"cluster_str": "Cluster"},
        center={
            "lat": gdf_.geometry.centroid.y.mean(),
            "lon": gdf_.geometry.centroid.x.mean(),
        },
        zoom=4.5,
    )

    fig_admin.update_layout(
        title="Clusters by Administrative Boundary: K-means",
        template="simple_white",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        height=400,
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""We can also identify a boolean in/out of season classification based on a simple threshold of the normalized rainfall values."""
    )
    return


@app.cell
def _(mo):
    threshold = mo.ui.number(
        start=0, stop=1, step=0.05, value=0.25, label="In-season threshold"
    )
    threshold
    return (threshold,)


@app.cell
def _(
    ADM_LEVEL,
    N_ZONES,
    cluster_colors,
    gdf_,
    go,
    kmeans,
    months,
    np,
    threshold,
):
    heatmap_values = []
    admin_names = []

    # Sort gdf by cluster
    gdf_sorted = gdf_.sort_values("cluster")

    for _idx, _row in gdf_sorted.iterrows():
        cluster_id = int(_row["cluster"])
        rainy_pattern = kmeans.cluster_centers_[cluster_id] > threshold.value
        # Use cluster_id+1 where rainy (shift clusters to 1,2,3...), 0 where not rainy
        heatmap_values.append(np.where(rainy_pattern, cluster_id + 1, 0))
        admin_names.append(_row[f"ADM{ADM_LEVEL}_PCODE"])

    # Create colorscale: 0=white, 1 to N_ZONES map to cluster 0 to N_ZONES-1
    _colorscale = [[0, "white"]]  # Non-rainy

    for _i in range(N_ZONES):
        position = (_i + 1) / N_ZONES
        _colorscale.append([position, cluster_colors[_i]])

    fig_rainy = go.Figure(
        data=go.Heatmap(
            z=heatmap_values,
            x=months,
            y=admin_names,
            colorscale=_colorscale,
            showscale=False,
        )
    )

    fig_rainy.update_layout(
        title="Rainy Season by Admin Unit",
        template="simple_white",
        height=len(admin_names) * 20,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    x = 10  # noqa
    return fig_rainy, gdf_sorted


@app.cell
def _(fig_rainy, mo):
    mo.accordion({"### See plot": fig_rainy})
    return


@app.cell
def _(gdf_sorted, mo):
    mo.accordion(
        {
            "### Download cluster assignments per admin": gdf_sorted.drop(
                ["geometry", "cluster_str"], axis=1
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Index of seasonality

    We can also identify locations with annual/biannual rainfall regimes using an index of seasonality as defined by [Dunning et al., 2016](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JD025428). This approach is appropriate for differentiating between areas with either one or two wet seasons per year.

    This method works by first computing the amplitude of the first and second harmonics, per pixel. Then we calculate the ratio of the amplitude of the second harmonic to the first harmonic. If the ratio is greater than 1, then we classify the pixel as having a biannual rainfall regime. Less than 1, and the pixel is considered to have an annual regime.

    Note that this method should be applied with caution in Saharan regions where low rainfalls can produce misleading harminic ratios. This method may also have a hard time distinguishing seasonality patterns in humid equatorial regions.
    """
    )
    return


@app.cell
def _(fft, monthly_clim, np, xr):
    fft_result = fft(monthly_clim, axis=0)

    first_harmonic = 2 * np.abs(fft_result[1]) / 12
    second_harmonic = 2 * np.abs(fft_result[2]) / 12

    ratio = second_harmonic / first_harmonic

    valid_mask = ~np.isnan(ratio) & (first_harmonic > 0)
    regime = np.full_like(ratio, np.nan)
    regime[valid_mask] = (ratio[valid_mask] > 1.0).astype(float)

    coords = {"y": monthly_clim.coords["y"], "x": monthly_clim.coords["x"]}

    da_ratio = xr.DataArray(ratio, coords=coords)
    da_regime = xr.DataArray(regime, coords=coords)
    return da_ratio, da_regime


@app.cell
def _(da_ratio, da_regime, go, make_subplots, np):
    # Create subplot figure
    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Harmonic Ratio", "Regime (0=Annual, 1=Biannual)"),
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    # Calculate percentiles to handle outliers
    ratio_values = da_ratio.values[~np.isnan(da_ratio.values)]
    ratio_5th = np.percentile(ratio_values, 5)
    ratio_95th = np.percentile(ratio_values, 95)

    # Ensure the color range is symmetric around 1.0
    max_deviation = max(abs(1.0 - ratio_5th), abs(ratio_95th - 1.0))
    color_min = 1.0 - max_deviation
    color_max = 1.0 + max_deviation

    # Plot 1: Harmonic ratio
    _fig.add_trace(
        go.Heatmap(
            z=da_ratio.values,
            x=da_ratio.x.values,
            y=da_ratio.y.values,
            colorscale="RdBu_r",
            zmin=color_min,
            zmax=color_max,
            name="Ratio",
            showscale=True,
            colorbar=dict(x=0.45),
        ),
        row=1,
        col=1,
    )

    # Add contour for ratio=1.0 threshold
    _fig.add_trace(
        go.Contour(
            z=da_ratio.values,
            x=da_ratio.x.values,
            y=da_ratio.y.values,
            contours_coloring="lines",
            contours=dict(start=1.0, end=1.0, size=0.1),
            line=dict(color="red", width=3),
            showscale=False,
            name="Threshold",
        ),
        row=1,
        col=1,
    )

    # Plot 2: Regime classification
    _fig.add_trace(
        go.Heatmap(
            z=da_regime.values,
            x=da_regime.x.values,
            y=da_regime.y.values,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            name="Regime",
            showscale=True,
            colorbar=dict(x=1.02),
        ),
        row=1,
        col=2,
    )

    # Update layout
    _fig.update_layout(
        title="Ethiopian Rainfall Regime Classification",
        width=800,
        height=400,
        template="simple_white",
    )

    _fig.update_xaxes(visible=False)
    _fig.update_yaxes(visible=False)
    return


@app.cell
def _(mo):
    mo.md(r"""Now summarize per admin region, as we did above.""")
    return


@app.cell
def _(ADM_LEVEL, cluster_colors, da_regime, exact_extract, gdf, gdf_, px):
    # Get the cluster with the majority of pixels in each polygon
    gdf_harmonic = exact_extract(
        da_regime,
        gdf,
        "majority",
        include_cols=[f"ADM{ADM_LEVEL}_EN", f"ADM{ADM_LEVEL}_PCODE"],
        output="pandas",
        include_geom=True,
    )
    gdf_harmonic = gdf_harmonic.rename(columns={"majority": "cluster"})

    # Convert cluster to string for discrete color mapping
    gdf_harmonic["cluster"] = gdf_harmonic["cluster"].astype("Int64")
    gdf_harmonic["cluster_str"] = gdf_harmonic["cluster"].astype(str)
    gdf_harmonic["geometry"] = gdf_harmonic["geometry"].simplify(
        tolerance=0.01
    )

    # Create choropleth map
    _fig_admin = px.choropleth_map(
        gdf_harmonic,
        geojson=gdf_.geometry,
        locations=gdf_harmonic.index,
        color="cluster_str",
        color_discrete_map={str(k): v for k, v in cluster_colors.items()},
        hover_data=[f"ADM{ADM_LEVEL}_PCODE"],
        labels={"cluster_str": "Cluster"},
        center={
            "lat": gdf_harmonic.geometry.centroid.y.mean(),
            "lon": gdf_harmonic.geometry.centroid.x.mean(),
        },
        zoom=4.5,
    )

    _fig_admin.update_layout(
        title="Clusters by Administrative Boundary: Harmonic Analysis",
        template="simple_white",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        height=400,
    )
    return (gdf_harmonic,)


@app.cell
def _(gdf_harmonic, mo):
    mo.accordion(
        {
            "### Download cluster assignments per admin": gdf_harmonic.drop(
                ["geometry", "cluster_str"], axis=1
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
