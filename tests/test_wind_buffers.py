import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from shapely import segmentize
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from ocha_lens.utils.storm import (
    _radius_from_quadrants,
    calculate_wind_buffers_gdf,
    fix_antimeridian,
    make_geodesic_quadrant_disk,
)

GEOD = Geod(ellps="WGS84")
NM = 1852.0


def _track(lons, lats, r34=100.0, r50=50.0, r64=20.0, basin="SP"):
    n = len(lons)
    df = pd.DataFrame(
        {
            "valid_time": pd.date_range("2026-01-01", periods=n, freq="6h"),
            "basin": basin,
        }
    )
    for speed, r in ((34, r34), (50, r50), (64, r64)):
        for q in ("ne", "se", "sw", "nw"):
            df[f"usa_quadrant_radius_{speed}_{q}"] = r
    return gpd.GeoDataFrame(
        df, geometry=[Point(x, y) for x, y in zip(lons, lats)], crs=4326
    )


def _geodesic_area_km2(geom):
    # Orient each part (exterior CCW, holes CW): pyproj sums signed ring
    # areas, so a hole wound like its shell would be added, not removed.
    parts = [orient(p, sign=1.0) for p in getattr(geom, "geoms", [geom])]
    return sum(GEOD.geometry_area_perimeter(p)[0] for p in parts) / 1e6


@pytest.mark.parametrize("lat", [-35.0, -18.0, 0.0, 25.0])
def test_geodesic_disk_reaches_stated_radius(lat):
    disk = make_geodesic_quadrant_disk((170.0, lat), *(100 * NM,) * 4)
    xs, ys = disk.exterior.xy
    _, _, d = GEOD.inv(np.full(len(xs), 170.0), np.full(len(xs), lat), xs, ys)
    assert np.allclose(d, 100 * NM, rtol=1e-6)


def test_geodesic_disk_quadrants_point_the_right_way():
    # Only the NE quadrant has wind: the disk's far point must be to the NE.
    disk = make_geodesic_quadrant_disk((170.0, -18.0), 100 * NM, 0.0, 0.0, 0.0)
    xs, ys = np.asarray(disk.exterior.xy)
    i = np.argmax((xs - 170.0) ** 2 + (ys + 18.0) ** 2)
    assert xs[i] > 170.0 and ys[i] > -18.0


def test_geodesic_buffer_area_matches_a_circle():
    # A stationary storm: one 100 nm circle, area pi r^2 within 0.5%.
    gdf = _track([170.0, 170.0], [-18.0, -18.0])
    out = calculate_wind_buffers_gdf(gdf, method="geodesic")
    g34 = out.loc[out.wind_speed_kt == 34, "geometry"].iloc[0]
    expected = np.pi * (100 * 1.852) ** 2
    assert _geodesic_area_km2(g34) == pytest.approx(expected, rel=5e-3)


def test_mercator_buffers_are_cos_lat_too_small():
    # Documents why the geodesic method exists.
    gdf = _track([170.0, 170.0], [-35.0, -35.0])
    merc = calculate_wind_buffers_gdf(gdf, method="mercator")
    geo = calculate_wind_buffers_gdf(gdf, method="geodesic")
    a_m = _geodesic_area_km2(merc.geometry.iloc[0])
    a_g = _geodesic_area_km2(geo.geometry.iloc[0])
    assert a_m / a_g == pytest.approx(np.cos(np.radians(35)) ** 2, rel=0.02)


def test_dateline_crossing_is_split_and_symmetric():
    # The same track mirrored across the antimeridian gives the same area,
    # and no stored part spans more than 180 degrees of longitude.
    east = _track([178.0, 179.0, -179.0, -177.0], [-17.0, -18.0, -19.0, -20.0])
    west = _track([-178.0, -179.0, 179.0, 177.0], [-17.0, -18.0, -19.0, -20.0])
    a = calculate_wind_buffers_gdf(east, method="geodesic")
    b = calculate_wind_buffers_gdf(west, method="geodesic")
    for ga, gb in zip(a.geometry, b.geometry):
        assert _geodesic_area_km2(ga) == pytest.approx(
            _geodesic_area_km2(gb), rel=1e-3
        )
        for g in (ga, gb):
            assert g.geom_type == "MultiPolygon"
            for p in g.geoms:
                minx, _, maxx, _ = p.bounds
                assert -180 <= minx and maxx <= 180
                assert maxx - minx < 180


def _wrapped(geom):
    from shapely.ops import transform

    return transform(lambda x, y: ((np.asarray(x) + 180) % 360 - 180, y), geom)


def test_fix_antimeridian_splits_a_wrapped_disk_and_keeps_area():
    disk = make_geodesic_quadrant_disk((179.5, -18.0), *(200 * NM,) * 4)
    split = fix_antimeridian(_wrapped(disk))
    assert split.geom_type == "MultiPolygon"
    assert _geodesic_area_km2(split) == pytest.approx(
        _geodesic_area_km2(disk), rel=1e-6
    )


def test_missing_radii_contribute_nothing():
    gdf = _track([170.0, 171.0], [-18.0, -18.5], r64=np.nan)
    out = calculate_wind_buffers_gdf(gdf, method="geodesic")
    assert out.loc[out.wind_speed_kt == 64, "geometry"].iloc[0] is None


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        calculate_wind_buffers_gdf(_track([170.0], [-18.0]), method="flat")


def test_geodesic_disk_uses_the_mercator_quadrant_interpolation():
    radii = (100 * NM, 60 * NM, 30 * NM, 80 * NM)
    disk = make_geodesic_quadrant_disk((0.0, 0.0), *radii, n_points=72)
    xs, ys = disk.exterior.xy
    az, _, d = GEOD.inv(np.zeros(len(xs)), np.zeros(len(xs)), xs, ys)
    expected = _radius_from_quadrants(
        np.mod(90.0 - np.asarray(az), 360.0), *radii
    )
    assert np.allclose(d, expected, rtol=1e-6)


@pytest.mark.parametrize("method", ["mercator", "geodesic"])
def test_zero_and_missing_radii_same_for_both_methods(method):
    gdf = _track([170.0, 171.0], [-18.0, -18.5], r50=0.0, r64=np.nan)
    out = calculate_wind_buffers_gdf(gdf, method=method).set_index(
        "wind_speed_kt"
    )
    assert out.loc[50, "geometry"].is_empty  # all-zero radii: empty polygon
    assert out.loc[64, "geometry"] is None  # no radii anywhere: None


def test_speeds_outside_the_schema_rejected():
    with pytest.raises(ValueError):
        calculate_wind_buffers_gdf(_track([170.0], [-18.0]), speeds=[34, 100])
    out = calculate_wind_buffers_gdf(
        _track([170.0, 170.0], [-18.0, -18.0]), method="geodesic", speeds=[64]
    )
    assert out.wind_speed_kt.tolist() == [64]


def test_fix_antimeridian_keeps_a_hole_next_to_the_dateline():
    # Ring crossing 180 with a hole just west of it (Winston-like loop),
    # densified so geodesic areas compare like with like.
    shell = [(170, -25), (190, -25), (190, -15), (170, -15), (170, -25)]
    hole = [(175, -22), (179.9, -22), (179.9, -19), (175, -19), (175, -22)]
    geom = segmentize(Polygon(shell, [hole]), 0.05)
    split = fix_antimeridian(_wrapped(geom))
    assert split.geom_type == "MultiPolygon"
    assert sum(len(p.interiors) for p in split.geoms) == 1
    assert _geodesic_area_km2(split) == pytest.approx(
        _geodesic_area_km2(geom), rel=1e-4
    )
