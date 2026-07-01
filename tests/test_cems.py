"""Unit tests for ocha_lens.datasources.cems.

The behaviors worth pinning (easy to break silently):

  - get_activations follows the paginated ``next`` chain, flattens both
    countries shapes, and applies the client-side filters.
  - get_activation returns the raw detail dict and raises loudly when the
    code isn't found.
  - The three flatteners (get_products / get_catalog / get_stats) walk the
    AOI→product→layer/stats tree at the right granularity: one row per
    product, one row per layer (incl. layer-less products), one row per
    (category, subcategory) with numeric coercion of "NA"-style totals.
  - The download layer resolves URLs from dicts/rows/strings, writes to
    memory vs. disk correctly, and download_products honours the
    type/aoi/feasible filters and skips products with no zip URL.

HTTP is mocked at two seams:
  - cems._get_json    : the dashboard-api list/detail endpoints
  - cems.download_file: the actual file fetch (so download_* logic is tested
    without network I/O)
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ocha_lens.datasources import cems

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def detail_payload():
    """The full detail response (count/results) for EMSR884."""
    return json.loads((FIXTURES / "cems_emsr884.json").read_text())


@pytest.fixture(scope="session")
def activation(detail_payload):
    """The single activation dict (results[0])."""
    return detail_payload["results"][0]


@pytest.fixture(scope="session")
def list_payload():
    return json.loads((FIXTURES / "cems_activations_list.json").read_text())


# ---------------------------------------------------------------------------
# Discovery: get_activations
# ---------------------------------------------------------------------------


def _install_get_json(monkeypatch, dispatch):
    def fake(url, params=None):
        return dispatch(url, params)

    monkeypatch.setattr(cems, "_get_json", fake)


def test_get_activations_paginates_and_parses(monkeypatch, list_payload):
    # Two pages: first carries a `next`, second ends the chain.
    page1 = {**list_payload, "next": "PAGE2"}
    page2 = {
        **list_payload,
        "results": list_payload["results"][:1],
        "next": None,
    }
    seen = []

    def dispatch(url, params):
        seen.append(url)
        return page1 if url == cems._LIST_URL else page2

    _install_get_json(monkeypatch, dispatch)
    df = cems.get_activations()

    # First page URL then the `next` URL were both fetched.
    assert seen == [cems._LIST_URL, "PAGE2"]
    assert len(df) == len(page1["results"]) + len(page2["results"])
    assert set(df.columns) == set(cems._ACTIVATION_COLS)
    # countries flattened to a joined string (list endpoint = plain strings).
    assert df.loc[df["code"] == "EMSR885", "countries"].iloc[0] == "Spain"
    # newest code first
    assert df["code"].iloc[0] == df["code"].max()


def test_get_activations_filters(monkeypatch, list_payload):
    _install_get_json(monkeypatch, lambda u, p: {**list_payload, "next": None})

    quakes = cems.get_activations(category="earthquake")  # case-insensitive
    assert set(quakes["category"]) == {"Earthquake"}

    ongoing = cems.get_activations(closed=False)
    assert (ongoing["closed"] == False).all()  # noqa: E712

    spain = cems.get_activations(country="spa")  # substring, case-insensitive
    assert (spain["countries"].str.contains("Spain")).all()


def test_get_activations_country_is_literal_not_regex(
    monkeypatch, list_payload
):
    # `country` is a literal substring, not a regex. A name with regex
    # metacharacters (parentheses) must not raise, and a metachar in the query
    # must not match arbitrarily.
    rec = {
        **list_payload["results"][0],
        "code": "EMSR900",
        "countries": ["Congo (the Democratic Republic of the)"],
    }
    payload = {**list_payload, "results": [rec], "next": None}
    _install_get_json(monkeypatch, lambda u, p: payload)

    # Unbalanced paren in the query would be an re.error under regex=True.
    hit = cems.get_activations(country="Congo (the")
    assert len(hit) == 1
    # A regex wildcard must be treated literally: "a.b" doesn't match "...ratic".
    assert cems.get_activations(country="a.b").empty


def test_get_activations_sorts_by_numeric_code(monkeypatch, list_payload):
    # Newest-first must hold once codes reach 4 digits: EMSR1000 > EMSR999.
    base = list_payload["results"][0]
    recs = [
        {**base, "code": "EMSR999"},
        {**base, "code": "EMSR1000"},
        {**base, "code": "EMSR885"},
    ]
    payload = {**list_payload, "results": recs, "next": None}
    _install_get_json(monkeypatch, lambda u, p: payload)

    df = cems.get_activations()
    assert list(df["code"]) == ["EMSR1000", "EMSR999", "EMSR885"]


def test_get_activations_empty(monkeypatch):
    _install_get_json(monkeypatch, lambda u, p: {"results": [], "next": None})
    df = cems.get_activations()
    assert df.empty
    assert set(df.columns) == set(cems._ACTIVATION_COLS)


# ---------------------------------------------------------------------------
# Discovery: get_activation
# ---------------------------------------------------------------------------


def test_get_activation_returns_dict(monkeypatch, detail_payload):
    captured = {}

    def dispatch(url, params):
        captured["params"] = params
        return detail_payload

    _install_get_json(monkeypatch, dispatch)
    act = cems.get_activation("EMSR884")
    assert act["code"] == "EMSR884"
    assert captured["params"] == {"code": "EMSR884"}


def test_get_activation_not_found(monkeypatch):
    _install_get_json(monkeypatch, lambda u, p: {"results": []})
    with pytest.raises(cems.ActivationNotFoundError):
        cems.get_activation("EMSR000")


# ---------------------------------------------------------------------------
# Flatteners
# ---------------------------------------------------------------------------


def test_get_products_one_row_per_product(activation):
    df = cems.get_products(activation)
    n_products = sum(len(a["products"]) for a in activation["aois"])
    assert len(df) == n_products
    assert set(df.columns) == set(cems._PRODUCT_COLS)
    # The feasible product with a zip carries a download_url.
    grm = df[df["product_type"] == "GRM"].iloc[0]
    assert grm["download_url"].endswith(".zip")
    assert grm["n_layers"] == 2
    # The layer-less GRA has no download URL (null), not an empty string.
    no_dl = df[df["download_url"].isna()]
    assert len(no_dl) == 1


def test_get_products_accepts_code(monkeypatch, detail_payload):
    _install_get_json(monkeypatch, lambda u, p: detail_payload)
    df = cems.get_products("EMSR884")
    assert (df["code"] == "EMSR884").all()


def test_get_products_mixed_timestamp_precision():
    # The live service mixes ISO8601 precisions within one activation:
    # deliveryTime carries microseconds, expectedDelivery does not. A naive
    # format-inferring parse picks the first row's format and fails on the
    # rest — pin that both parse to real timestamps (regression guard).
    act = {
        "code": "EMSR999",
        "aois": [
            {
                "number": 0,
                "name": "AOI",
                "products": [
                    {
                        "id": 1,
                        "type": "GRA",
                        "expectedDelivery": "2026-06-28T10:50:00",
                        "version": {
                            "number": 1,
                            "deliveryTime": "2026-06-26T14:50:11.319814",
                        },
                    },
                    {
                        "id": 2,
                        "type": "GRA",
                        "expectedDelivery": "2026-06-27T06:00:00",
                        "version": {
                            "number": 1,
                            "deliveryTime": "2026-06-26T04:01:10.948274",
                        },
                    },
                ],
            }
        ],
    }
    df = cems.get_products(act)
    assert str(df["delivery_time"].dtype).startswith("datetime64")
    assert str(df["expected_delivery"].dtype).startswith("datetime64")
    assert df["delivery_time"].notna().all()


def test_get_catalog_one_row_per_layer(activation):
    df = cems.get_catalog(activation)
    n_layers = sum(
        len(p.get("layers") or [])
        for a in activation["aois"]
        for p in a["products"]
    )
    n_layerless = sum(
        1
        for a in activation["aois"]
        for p in a["products"]
        if not (p.get("layers") or [])
    )
    # Layer rows + one placeholder row per layer-less product.
    assert len(df) == n_layers + n_layerless
    assert set(df.columns) == set(cems._CATALOG_COLS)
    # Layer-less product still surfaces its zip URL.
    placeholder = df[df["layer_name"].isna()]
    assert len(placeholder) == n_layerless
    # GeoJSON URLs are populated for vt/json layers.
    assert df["geojson_url"].notna().any()


def test_get_stats_flattens_and_coerces(activation):
    df = cems.get_stats(activation)
    assert set(df.columns) == set(cems._STATS_COLS)
    # The "NA" placeholder total coerces to null, not the string "NA".
    assert df["total"].dtype == "Float64"
    assert not (df["total"].astype("string") == "NA").any()
    # A real numeric total survives.
    assert df["total"].notna().any()
    # Built-up / Residential Buildings affected figure from the fixture.
    resid = df[df["subcategory"] == "Residential Buildings"]
    assert not resid.empty
    assert resid["affected"].notna().any()


# ---------------------------------------------------------------------------
# Download layer
# ---------------------------------------------------------------------------


def test_product_url_resolution():
    assert cems._product_url("http://x/y.zip") == "http://x/y.zip"
    assert (
        cems._product_url({"downloadPath": "http://x/a.zip"})
        == "http://x/a.zip"
    )
    row = pd.Series({"download_url": "http://x/b.zip"})
    assert cems._product_url(row) == "http://x/b.zip"
    with pytest.raises(ValueError):
        cems._product_url({"id": 1})


def test_download_file_to_memory(monkeypatch):
    class _Resp:
        content = b"payload"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(cems._session, "get", lambda url, timeout: _Resp())
    assert cems.download_file("http://x/f.bin") == b"payload"


def test_download_file_to_disk(monkeypatch, tmp_path):
    class _Resp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"abc"
            yield b"def"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        cems._session, "get", lambda url, timeout, stream: _Resp()
    )
    out = cems.download_file("http://x/EMSR884_AOI00.zip", dest=tmp_path)
    assert out == tmp_path / "EMSR884_AOI00.zip"
    assert out.read_bytes() == b"abcdef"


def test_download_products_filters_and_skips(monkeypatch, activation):
    calls = []

    def fake_download(url, dest=None):
        calls.append(url)
        return b"zip"

    monkeypatch.setattr(cems, "download_file", fake_download)

    # Only GRM: one product, downloaded; keyed by filename.
    out = cems.download_products(activation, product_types=["GRM"])
    assert len(out) == 1
    assert all(k.endswith(".zip") for k in out)
    assert all(v == b"zip" for v in out.values())

    # All feasible products with a URL: the GRM + the GRA-with-zip, but NOT
    # the layer-less GRA (no downloadPath → skipped).
    calls.clear()
    out_all = cems.download_products(activation)
    assert len(out_all) == 2


def test_download_products_logs_summary_not_warnings(
    monkeypatch, activation, caplog
):
    # Skips (products with no published zip) should NOT emit WARNINGs — they
    # are the normal W/N state — and a single INFO summary should report the
    # downloaded/skipped counts.
    import logging

    monkeypatch.setattr(cems, "download_file", lambda url, dest=None: b"z")
    with caplog.at_level(logging.DEBUG, logger=cems.logger.name):
        out = cems.download_products(activation)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    summaries = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "download_products" in r.message
    ]
    assert len(summaries) == 1
    msg = summaries[0].getMessage()
    assert f"{len(out)} downloaded" in msg
    # The fixture has products with no downloadPath → reported as skipped.
    assert "skipped" in msg


def test_download_products_feasible_filter(monkeypatch):
    # Synthetic activation: one feasible, one not — both with a URL.
    act = {
        "code": "EMSR999",
        "aois": [
            {
                "number": 0,
                "name": "AOI",
                "products": [
                    {
                        "id": 1,
                        "type": "GRA",
                        "feasible": True,
                        "downloadPath": "http://x/a.zip",
                    },
                    {
                        "id": 2,
                        "type": "GRA",
                        "feasible": False,
                        "downloadPath": "http://x/b.zip",
                    },
                ],
            }
        ],
    }
    monkeypatch.setattr(cems, "download_file", lambda url, dest=None: b"z")

    assert len(cems.download_products(act)) == 1  # default feasible_only
    assert len(cems.download_products(act, feasible_only=False)) == 2


def test_download_activation_bundle(monkeypatch, activation):
    monkeypatch.setattr(
        cems,
        "download_file",
        lambda url, dest=None: ("BUNDLE", url),
    )
    _, url = cems.download_activation_bundle(activation)
    assert url == activation["productsPath"]


def test_download_activation_bundle_missing_url(monkeypatch):
    with pytest.raises(ValueError):
        cems.download_activation_bundle({"code": "X", "aois": []})


def test_download_geojson(monkeypatch):
    gdf_in = gpd.GeoDataFrame(
        {"a": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
    )
    monkeypatch.setattr(
        cems, "download_file", lambda url: gdf_in.to_json().encode()
    )

    # From a catalog-style row.
    gdf = cems.download_geojson({"geojson_url": "http://x/l.json"})
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    # From a raw layer dict (`json` key).
    assert len(cems.download_geojson({"json": "http://x/l.json"})) == 1
    with pytest.raises(ValueError):
        cems.download_geojson({"name": "no-url"})


def test_to_blob_requires_stratus(monkeypatch):
    # ocha_stratus is not installed in the test env → ImportError surfaces.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ocha_stratus":
            raise ImportError("no stratus")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="ocha-stratus"):
        cems.to_blob(b"x", "cems/x.zip")
