import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # WSP Polygon → Track Matching

        `match_wsp_to_tracks` assigns an `atcf_id` to each row in
        `storms.nhc_wsp_polygon` by matching against `storms.nhc_tracks_geo`.

        The matching logic has three cases:

        | Scenario | Method |
        |---|---|
        | One active storm at `issued_time` | Direct merge — no geometry needed |
        | Multiple active storms | Explode MultiPolygon components, pick the track with the most forecast points inside each component |
        | No tracks at `issued_time` | Returned with `atcf_id = None` |

        **Does the whole track need to be inside the polygon?**
        No — we count how many forecast *points* from each track fall inside the
        *filled* polygon (outer boundary only; donut holes are ignored).
        The storm with the highest count wins.  The centroid fallback only fires
        if literally no track has any points inside.
        """
    )
    return


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from shapely.geometry import MultiPolygon, Point, Polygon

    from ocha_lens.utils.storm import match_wsp_to_tracks

    return (
        MultiPolygon,
        Point,
        Polygon,
        gpd,
        match_wsp_to_tracks,
        mpatches,
        np,
        pd,
        plt,
    )


@app.cell
def _(Point, Polygon):
    def _make_donut(cx, cy, r_outer, r_inner, n=64):
        """Create a ring polygon (outer circle minus inner circle) in degrees."""
        outer = Point(cx, cy).buffer(r_outer, resolution=n)
        inner = Point(cx, cy).buffer(r_inner, resolution=n)
        return outer.difference(inner)

    def _make_track_points(cx, cy, n=8, spread=1.5):
        """Create a cluster of Points simulating forecast positions near (cx, cy)."""
        import numpy as np

        rng = np.random.default_rng(42)
        lons = cx + rng.uniform(-spread, spread, n)
        lats = cy + rng.uniform(-spread, spread, n)
        return [Point(lon, lat) for lon, lat in zip(lons, lats)]

    return


@app.cell
def _(mo):
    mo.md("## Case 1 — single active storm")
    return


@app.cell
def _(MultiPolygon, Point, Polygon, gpd, match_wsp_to_tracks, pd):
    def _make_donut(cx, cy, r_outer, r_inner, n=64):
        outer = Point(cx, cy).buffer(r_outer, resolution=n)
        inner = Point(cx, cy).buffer(r_inner, resolution=n)
        return outer.difference(inner)

    _t1 = pd.Timestamp("2024-09-10 12:00")

    # One donut for storm AL05 centred at (-70, 25)
    _donut_a = _make_donut(-70, 25, r_outer=4, r_inner=1.5)

    _gdf_wsp_1 = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "issued_time": [_t1, _t1, _t1],
            "wind_threshold_kt": [34, 50, 64],
            "percentage": [50, 50, 50],
        },
        geometry=[MultiPolygon([_donut_a])] * 3,
        crs="EPSG:4326",
    )

    # Eight track forecast points scattered around the storm centre
    import numpy as np

    _rng = np.random.default_rng(0)
    _track_pts = [
        Point(-70 + _rng.uniform(-1.2, 1.2), 25 + _rng.uniform(-1.2, 1.2))
        for _ in range(8)
    ]
    _gdf_tracks_1 = gpd.GeoDataFrame(
        {
            "atcf_id": ["AL052024"] * 8,
            "issued_time": [_t1] * 8,
            "valid_time": pd.date_range(_t1, periods=8, freq="6h"),
        },
        geometry=_track_pts,
        crs="EPSG:4326",
    )

    _result_1 = match_wsp_to_tracks(_gdf_wsp_1, _gdf_tracks_1)
    _result_1[
        ["id", "issued_time", "wind_threshold_kt", "percentage", "atcf_id"]
    ]
    return


@app.cell
def _(mo):
    mo.md("## Case 2 — two active storms at the same issued_time")
    return


@app.cell
def _(MultiPolygon, Point, Polygon, gpd, match_wsp_to_tracks, pd):
    def _make_donut(cx, cy, r_outer, r_inner, n=64):
        outer = Point(cx, cy).buffer(r_outer, resolution=n)
        inner = Point(cx, cy).buffer(r_inner, resolution=n)
        return outer.difference(inner)

    _t2 = pd.Timestamp("2024-09-14 00:00")
    import numpy as np

    _rng = np.random.default_rng(1)

    # Storm A: Atlantic, centred at (-70, 25)
    # Storm B: Eastern Pacific, centred at (-110, 18)
    _donut_a = _make_donut(-70, 25, r_outer=4, r_inner=1.5)
    _donut_b = _make_donut(-110, 18, r_outer=3, r_inner=1.0)

    # NHC publishes one MultiPolygon per (issued_time, threshold, percentage)
    # containing one component per active storm
    _gdf_wsp_2 = gpd.GeoDataFrame(
        {
            "id": [10, 11],
            "issued_time": [_t2, _t2],
            "wind_threshold_kt": [34, 34],
            "percentage": [50, 70],
        },
        geometry=[MultiPolygon([_donut_a, _donut_b])] * 2,
        crs="EPSG:4326",
    )

    _pts_a = [
        Point(-70 + _rng.uniform(-1.2, 1.2), 25 + _rng.uniform(-1.2, 1.2))
        for _ in range(8)
    ]
    _pts_b = [
        Point(-110 + _rng.uniform(-1.0, 1.0), 18 + _rng.uniform(-1.0, 1.0))
        for _ in range(8)
    ]
    _gdf_tracks_2 = gpd.GeoDataFrame(
        {
            "atcf_id": ["AL052024"] * 8 + ["EP092024"] * 8,
            "issued_time": [_t2] * 16,
            "valid_time": list(pd.date_range(_t2, periods=8, freq="6h")) * 2,
        },
        geometry=_pts_a + _pts_b,
        crs="EPSG:4326",
    )

    _result_2 = match_wsp_to_tracks(_gdf_wsp_2, _gdf_tracks_2)
    # Each row is exploded — expect 4 rows (2 wsp rows × 2 storm components each)
    _result_2[
        ["id", "issued_time", "wind_threshold_kt", "percentage", "atcf_id"]
    ]
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Edge case — track entirely inside the hole (centroid fallback)

        If every forecast point lands in the donut's hole (e.g. the inner radius
        is large relative to the forecast scatter), `within()` returns zero for
        all tracks.  The fallback computes the centroid of each track cluster and
        assigns the polygon to the nearest one.
        """
    )
    return


@app.cell
def _(MultiPolygon, Point, Polygon, gpd, match_wsp_to_tracks, pd):
    def _make_donut(cx, cy, r_outer, r_inner, n=64):
        outer = Point(cx, cy).buffer(r_outer, resolution=n)
        inner = Point(cx, cy).buffer(r_inner, resolution=n)
        return outer.difference(inner)

    _t3 = pd.Timestamp("2024-09-18 06:00")
    import numpy as np

    _rng = np.random.default_rng(2)

    # Very large inner radius — track points land in the hole
    _donut_tight = _make_donut(-70, 25, r_outer=5, r_inner=4)
    _donut_far = _make_donut(-110, 18, r_outer=5, r_inner=4)

    _gdf_wsp_3 = gpd.GeoDataFrame(
        {
            "id": [20],
            "issued_time": [_t3],
            "wind_threshold_kt": [34],
            "percentage": [90],
        },
        geometry=[MultiPolygon([_donut_tight, _donut_far])],
        crs="EPSG:4326",
    )

    # Track points clustered tightly at the storm centres (inside both holes)
    _pts_a = [
        Point(-70 + _rng.uniform(-0.3, 0.3), 25 + _rng.uniform(-0.3, 0.3))
        for _ in range(6)
    ]
    _pts_b = [
        Point(-110 + _rng.uniform(-0.3, 0.3), 18 + _rng.uniform(-0.3, 0.3))
        for _ in range(6)
    ]
    _gdf_tracks_3 = gpd.GeoDataFrame(
        {
            "atcf_id": ["AL052024"] * 6 + ["EP092024"] * 6,
            "issued_time": [_t3] * 12,
            "valid_time": list(pd.date_range(_t3, periods=6, freq="6h")) * 2,
        },
        geometry=_pts_a + _pts_b,
        crs="EPSG:4326",
    )

    _result_3 = match_wsp_to_tracks(_gdf_wsp_3, _gdf_tracks_3)
    _result_3[
        ["id", "issued_time", "wind_threshold_kt", "percentage", "atcf_id"]
    ]
    return


@app.cell
def _(mo):
    mo.md("## Edge case — no tracks for an issued_time")
    return


@app.cell
def _(MultiPolygon, Point, Polygon, gpd, match_wsp_to_tracks, pd):
    def _make_donut(cx, cy, r_outer, r_inner, n=64):
        outer = Point(cx, cy).buffer(r_outer, resolution=n)
        inner = Point(cx, cy).buffer(r_inner, resolution=n)
        return outer.difference(inner)

    _t4 = pd.Timestamp("2024-10-01 00:00")
    _t5 = pd.Timestamp("2024-10-02 00:00")

    _gdf_wsp_4 = gpd.GeoDataFrame(
        {
            "id": [30, 31],
            "issued_time": [_t4, _t5],  # t5 has no tracks
            "wind_threshold_kt": [34, 34],
            "percentage": [50, 50],
        },
        geometry=[MultiPolygon([_make_donut(-70, 25, 4, 1.5)])] * 2,
        crs="EPSG:4326",
    )

    _gdf_tracks_4 = gpd.GeoDataFrame(
        {
            "atcf_id": ["AL052024"],
            "issued_time": [_t4],  # no entry for t5
            "valid_time": [_t4],
        },
        geometry=[Point(-70, 25)],
        crs="EPSG:4326",
    )

    _result_4 = match_wsp_to_tracks(_gdf_wsp_4, _gdf_tracks_4)
    _result_4[
        ["id", "issued_time", "wind_threshold_kt", "percentage", "atcf_id"]
    ]
    return


@app.cell
def _(mo):
    mo.md("## Visual overview — two-storm case")
    return


@app.cell
def _(
    MultiPolygon, Point, Polygon, gpd, match_wsp_to_tracks, mpatches, pd, plt
):
    def _make_donut(cx, cy, r_outer, r_inner, n=64):
        outer = Point(cx, cy).buffer(r_outer, resolution=n)
        inner = Point(cx, cy).buffer(r_inner, resolution=n)
        return outer.difference(inner)

    import numpy as np

    _rng = np.random.default_rng(1)
    _t = pd.Timestamp("2024-09-14 00:00")

    _donut_a = _make_donut(-70, 25, r_outer=4, r_inner=1.5)
    _donut_b = _make_donut(-110, 18, r_outer=3, r_inner=1.0)

    _gdf_wsp = gpd.GeoDataFrame(
        {
            "id": [10],
            "issued_time": [_t],
            "wind_threshold_kt": [34],
            "percentage": [50],
        },
        geometry=[MultiPolygon([_donut_a, _donut_b])],
        crs="EPSG:4326",
    )
    _pts_a = [
        Point(-70 + _rng.uniform(-1.2, 1.2), 25 + _rng.uniform(-1.2, 1.2))
        for _ in range(8)
    ]
    _pts_b = [
        Point(-110 + _rng.uniform(-1.0, 1.0), 18 + _rng.uniform(-1.0, 1.0))
        for _ in range(8)
    ]
    _gdf_tracks = gpd.GeoDataFrame(
        {
            "atcf_id": ["AL052024"] * 8 + ["EP092024"] * 8,
            "issued_time": [_t] * 16,
            "valid_time": list(pd.date_range(_t, periods=8, freq="6h")) * 2,
        },
        geometry=_pts_a + _pts_b,
        crs="EPSG:4326",
    )

    _result = match_wsp_to_tracks(_gdf_wsp, _gdf_tracks)

    _colors = {"AL052024": "#2196F3", "EP092024": "#FF5722"}
    _fig, _ax = plt.subplots(figsize=(10, 5))
    for _, _row in _result.iterrows():
        _c = _colors.get(_row["atcf_id"], "grey")
        _x, _y = _row.geometry.exterior.xy
        _ax.fill(_x, _y, alpha=0.25, color=_c)
        _ax.plot(_x, _y, color=_c, linewidth=1.5)
        # draw the hole(s)
        for _interior in _row.geometry.interiors:
            _hx, _hy = _interior.xy
            _ax.fill(_hx, _hy, color="white")
            _ax.plot(_hx, _hy, color=_c, linewidth=1, linestyle="--")

    for _atcf, _grp in _gdf_tracks.groupby("atcf_id"):
        _c = _colors[_atcf]
        _xs = [p.x for p in _grp.geometry]
        _ys = [p.y for p in _grp.geometry]
        _ax.scatter(_xs, _ys, color=_c, s=40, zorder=5, label=_atcf)

    _ax.set_title(
        "WSP donuts (filled = matched storm) + forecast track points"
    )
    _ax.legend(
        handles=[mpatches.Patch(color=v, label=k) for k, v in _colors.items()]
    )
    _ax.set_xlabel("Longitude")
    _ax.set_ylabel("Latitude")
    _fig
