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

    This is a POC analysis to identify distinct rainfall seasonality patterns across a selected country using gridded monthly precipitation data from ECMWF's ERA5 reanalysis dataset (1993-2016). For each grid cell, we compute a climatological seasonal cycle by averaging monthly rainfall across all years, then normalize these patterns to focus on the timing rather than magnitude of rainfall peaks. This normalization ensures that clustering is driven by seasonal patterns rather than absolute precipitation amounts.

    We then apply k-means clustering to group grid cells with similar seasonal cycles. Use the dropdown below to define the target number of clusters.
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
    from dotenv import find_dotenv, load_dotenv
    from exactextract import exact_extract
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    _ = load_dotenv(find_dotenv(usecwd=True))
    return KMeans, MinMaxScaler, date, exact_extract, go, np, px, stratus


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
    return clim_norm, cluster_map, kmeans, labels


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
        title="Clusters by Administrative Boundary",
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
def _(threshold):
    THRESHOLD = threshold.value
    return (THRESHOLD,)


@app.cell
def _(
    ADM_LEVEL,
    N_ZONES,
    THRESHOLD,
    cluster_colors,
    gdf_,
    go,
    kmeans,
    months,
    np,
):
    heatmap_values = []
    admin_names = []

    # Sort gdf by cluster
    gdf_sorted = gdf_.sort_values("cluster")

    for _idx, _row in gdf_sorted.iterrows():
        cluster_id = int(_row["cluster"])
        rainy_pattern = kmeans.cluster_centers_[cluster_id] > THRESHOLD
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
    return (gdf_sorted,)


@app.cell
def _(gdf_sorted):
    gdf_sorted
    return


if __name__ == "__main__":
    app.run()
