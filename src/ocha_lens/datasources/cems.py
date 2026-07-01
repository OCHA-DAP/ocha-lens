"""Copernicus EMS Rapid Mapping — emergency activation products datasource.

The Copernicus Emergency Management Service (CEMS) Rapid Mapping component
publishes geospatial damage-assessment products for activations triggered by
disasters (earthquakes, floods, wildfires, ...). Each *activation* (e.g.
``EMSR884``, "Earthquake in Venezuela") is broken into one or more *AOIs*
(areas of interest), each carrying one or more *products* (a First Estimate,
Delineation, Grading/damage assessment, ...). Every product is downloadable
as a zip and also exposed as individual *layers* (GeoJSON / vector tiles /
COG, each with an SLD style file).

Access is via two public, unauthenticated JSON endpoints (no API key):

* list   : ``…/dashboard-api/public-activations-info/``   (one row per
  activation, paginated through ``next`` URLs)
* detail : ``…/dashboard-api/public-activations/?code=…`` (the full nested
  activation → AOIs → products → layers/images/stats tree)

This module mirrors that hierarchy:

* :func:`get_activations`  — discover what activations exist (DataFrame).
* :func:`get_activation`   — the full nested detail dict for one code.
* :func:`get_products`     — flatten to one row per product (the download
  targets; this is the primary use case).
* :func:`get_catalog`      — flatten to one row per layer (individual geo
  files + their GeoJSON/SLD URLs).
* :func:`get_stats`        — flatten the per-product damage-statistics table.

Downloading (the main workflow — products individually and in bulk):

* :func:`download_product`         — one product zip → bytes or local file.
* :func:`download_products`        — every (filtered) product for an
  activation → dict of {filename: bytes|Path}.
* :func:`download_activation_bundle` — the activation-wide ``_products.zip``.
* :func:`download_geojson`         — one layer's GeoJSON → GeoDataFrame.
* :func:`download_file`            — generic URL → bytes or local file.

Everything returns in memory by default; pass ``dest``/``dest_dir`` to write
locally, or use :func:`to_blob` to push bytes to the OCHA Azure blob store.

Note the host is ``rapidmapping.emergency.copernicus.eu`` — the ``mapping.``
host that appears in some published OpenAPI specs 404s.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CEMS_BASE = "https://rapidmapping.emergency.copernicus.eu/backend"
_LIST_URL = f"{CEMS_BASE}/dashboard-api/public-activations-info/"
_DETAIL_URL = f"{CEMS_BASE}/dashboard-api/public-activations/"
_TIMEOUT = 60
# Streamed downloads (product zips, COGs) can be large; read in chunks.
_CHUNK = 1 << 20  # 1 MiB

# Default Azure blob container for to_blob(); callers can override.
_DEFAULT_BLOB_CONTAINER = "raster"

# Activation / AOI / product accept a code (str) or an already-fetched
# activation dict — see _resolve_activation.
ActivationRef = Union[str, Dict[str, Any]]

# Copernicus EMS Rapid Mapping product types, keyed by the ``type`` code the
# API returns on each product (the ``product_type`` column). Definitions per
# the Rapid Mapping portfolio:
# https://mapping.emergency.copernicus.eu/about/rapid-mapping-portfolio/
# Map onto a products table with e.g.
# ``df["product_type"].map(cems.PRODUCT_TYPES)``.
PRODUCT_TYPES = {
    "REF": (
        "Reference — pre-event baseline of the territory and exposed assets "
        "(provided only for activations outside Europe)."
    ),
    "FEP": (
        "First Estimate — extremely fast, rough assessment of the most "
        "affected locations."
    ),
    "DEL": (
        "Delineation — assessment of event impact and extent, with optional "
        "monitoring updates."
    ),
    "GRA": (
        "Grading — damage grade, its spatial distribution and extent; a "
        "superset of the Delineation product."
    ),
    "GRM": (
        "Ground Movement — ground-displacement mapping for earthquakes and "
        "volcanic activity (SAR-derived)."
    ),
}


# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------

_ACTIVATION_COLS = [
    "code",
    "name",
    "category",
    "countries",
    "event_time",
    "activation_time",
    "closed",
    "gdacs_id",
    "n_aois",
    "n_products",
    "last_update",
]

ACTIVATION_SCHEMA = pa.DataFrameSchema(
    {
        "code": pa.Column(str),
        "name": pa.Column(str),
        "category": pa.Column(str),
        # countries flattened to a "; "-joined string (the list/detail
        # endpoints disagree on shape; see _join_countries).
        "countries": pa.Column(str, nullable=True),
        "event_time": pa.Column(pa.DateTime, nullable=True),
        "activation_time": pa.Column(pa.DateTime, nullable=True),
        "closed": pa.Column(bool),
        # gdacsId is genuinely null for non-GDACS-linked activations.
        "gdacs_id": pa.Column(str, nullable=True),
        "n_aois": pa.Column("Int64", nullable=True),
        "n_products": pa.Column("Int64", nullable=True),
        "last_update": pa.Column(pa.DateTime, nullable=True),
    },
    strict=True,
    coerce=True,
)


_PRODUCT_COLS = [
    "code",
    "aoi_number",
    "aoi_name",
    "product_id",
    "product_type",
    "monitoring",
    "monitoring_number",
    "feasible",
    "status_code",
    "version_number",
    "delivery_time",
    "expected_delivery",
    "n_layers",
    "n_images",
    "download_url",
]

PRODUCT_SCHEMA = pa.DataFrameSchema(
    {
        "code": pa.Column(str),
        "aoi_number": pa.Column("Int64", nullable=True),
        "aoi_name": pa.Column(str, nullable=True),
        "product_id": pa.Column("Int64", nullable=True),
        # type: FEP/REF/DEL/GRA/GRM/... — not enum-checked, the service adds
        # new product types over time and we don't want to reject them.
        "product_type": pa.Column(str),
        "monitoring": pa.Column(bool, nullable=True),
        "monitoring_number": pa.Column("Int64", nullable=True),
        # feasible=False means the product was requested but not produced; it
        # has no usable download. download_products skips these by default.
        "feasible": pa.Column(bool, nullable=True),
        "status_code": pa.Column(str, nullable=True),
        "version_number": pa.Column("Int64", nullable=True),
        "delivery_time": pa.Column(pa.DateTime, nullable=True),
        "expected_delivery": pa.Column(pa.DateTime, nullable=True),
        "n_layers": pa.Column("Int64", nullable=True),
        "n_images": pa.Column("Int64", nullable=True),
        # download_url null when the service hasn't published the zip yet.
        "download_url": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
)


_CATALOG_COLS = [
    "code",
    "aoi_number",
    "aoi_name",
    "product_id",
    "product_type",
    "layer_name",
    "layer_format",
    "geojson_url",
    "sld_url",
    "product_zip_url",
]

CATALOG_SCHEMA = pa.DataFrameSchema(
    {
        "code": pa.Column(str),
        "aoi_number": pa.Column("Int64", nullable=True),
        "aoi_name": pa.Column(str, nullable=True),
        "product_id": pa.Column("Int64", nullable=True),
        "product_type": pa.Column(str),
        # Layer fields are null for products that expose no individual layers
        # (we still emit one row so the product_zip_url isn't lost).
        "layer_name": pa.Column(str, nullable=True),
        "layer_format": pa.Column(str, nullable=True),
        "geojson_url": pa.Column(str, nullable=True),
        "sld_url": pa.Column(str, nullable=True),
        "product_zip_url": pa.Column(str, nullable=True),
    },
    strict=True,
    coerce=True,
)


_STATS_COLS = [
    "code",
    "aoi_number",
    "aoi_name",
    "product_id",
    "product_type",
    "category",
    "subcategory",
    "unit",
    "total",
    "affected",
]

STATS_SCHEMA = pa.DataFrameSchema(
    {
        "code": pa.Column(str),
        "aoi_number": pa.Column("Int64", nullable=True),
        "aoi_name": pa.Column(str, nullable=True),
        "product_id": pa.Column("Int64", nullable=True),
        "product_type": pa.Column(str),
        "category": pa.Column(str),
        "subcategory": pa.Column(str, nullable=True),
        "unit": pa.Column(str, nullable=True),
        # total/affected are coerced to float; non-numeric placeholders the
        # service uses (e.g. the literal "NA") become null.
        "total": pa.Column("Float64", nullable=True),
        "affected": pa.Column("Float64", nullable=True),
    },
    strict=True,
    coerce=True,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ActivationNotFoundError(ValueError):
    """No activation matched the requested code."""


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

_session = requests.Session()


def _get_json(
    url: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """GET a CEMS dashboard-api endpoint and return parsed JSON.

    Unlike some OCHA datasources (ADAM base64-wraps its payloads), CEMS
    returns plain JSON, so this is a thin wrapper over requests.
    """
    resp = _session.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Small parse helpers
# ---------------------------------------------------------------------------


def _clean_dt(value: Any) -> Any:
    """Parse a CEMS timestamp string to a pandas Timestamp (NaT if absent).

    We parse here rather than leaning on pandera's column-level coercion
    because the service mixes ISO8601 precisions within a single activation —
    some timestamps carry microseconds (deliveryTime) and some don't
    (expectedDelivery). pandas infers a single format from the first value and
    then errors on the rest; ``format="ISO8601"`` parses each independently.
    Empty strings ('', used for not-yet-known times) become NaT.
    """
    if value is None:
        return pd.NaT
    if isinstance(value, str) and not value.strip():
        return pd.NaT
    return pd.to_datetime(value, format="ISO8601")


def _clean_str(value: Any) -> Optional[str]:
    """Coerce empties to None; leave other values to pandera's str coercion."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _join_countries(countries: Optional[List[Any]]) -> Optional[str]:
    """Flatten a CEMS ``countries`` array to a '; '-joined string.

    The list endpoint returns plain strings (``["Spain"]``) while the detail
    endpoint returns objects (``[{"name": "Venezuela"}]``); handle both.
    """
    if not countries:
        return None
    names = [
        c if isinstance(c, str) else (c or {}).get("name", "")
        for c in countries
    ]
    joined = "; ".join(n for n in names if n)
    return joined or None


def _to_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion for stats totals (the service mixes
    numbers with placeholder strings like 'NA'). Non-numeric → None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_activation(ref: ActivationRef) -> Dict[str, Any]:
    """Accept a code (fetch) or an already-fetched activation dict (passthrough).

    Lets callers fetch ``get_activation(code)`` once and feed it to several of
    the flatteners/downloaders without re-hitting the API each time.
    """
    if isinstance(ref, dict):
        return ref
    return get_activation(ref)


def _url_filename(url: str) -> str:
    """Basename of a URL path, without query string."""
    return Path(urlsplit(url).path).name


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _activation_info_to_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one ``public-activations-info`` record to a row dict."""
    return {
        "code": rec["code"],
        "name": rec.get("name"),
        "category": rec.get("category"),
        "countries": _join_countries(rec.get("countries")),
        "event_time": _clean_dt(rec.get("eventTime")),
        "activation_time": _clean_dt(rec.get("activationTime")),
        "closed": rec.get("closed"),
        "gdacs_id": _clean_str(rec.get("gdacsId")),
        "n_aois": rec.get("n_aois"),
        "n_products": rec.get("n_products"),
        "last_update": _clean_dt(rec.get("lastUpdate")),
    }


def get_activations(
    category: Optional[str] = None,
    closed: Optional[bool] = None,
    country: Optional[str] = None,
) -> pd.DataFrame:
    """List all CEMS Rapid Mapping activations (one row per activation).

    Walks the paginated ``public-activations-info`` endpoint (following each
    response's ``next`` URL) and returns a tidy table. Filtering is applied
    client-side after the full list is fetched.

    Parameters
    ----------
    category : str, optional
        Case-insensitive exact match on the event category (e.g.
        ``"Earthquake"``, ``"Flood"``, ``"Wildfire"``).
    closed : bool, optional
        Keep only ongoing (``False``) or closed (``True``) activations. None
        returns both.
    country : str, optional
        Case-insensitive substring match against the joined countries string
        (so ``"venezu"`` matches "Venezuela").

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`ACTIVATION_SCHEMA`, sorted by activation code
        descending (newest first).
    """
    rows: List[Dict[str, Any]] = []
    url: Optional[str] = _LIST_URL
    while url:
        data = _get_json(url)
        results = data.get("results", [])
        for rec in results:
            rows.append(_activation_info_to_row(rec))
        logger.info(
            "CEMS activations page: %d records (total: %d)",
            len(results),
            len(rows),
        )
        url = data.get("next")

    if not rows:
        return ACTIVATION_SCHEMA.validate(
            pd.DataFrame(columns=_ACTIVATION_COLS)
        )

    df = pd.DataFrame(rows)

    if category is not None:
        df = df[df["category"].str.casefold() == category.casefold()]
    if closed is not None:
        df = df[df["closed"] == closed]
    if country is not None:
        # regex=False: `country` is a literal substring (the documented
        # contract), not a pattern. Without it, names/inputs containing regex
        # metacharacters misbehave — e.g. "Congo (the" raises re.error on the
        # unbalanced paren, and "a.b" would match "aXb".
        df = df[
            df["countries"]
            .fillna("")
            .str.casefold()
            .str.contains(country.casefold(), regex=False)
        ]

    # Sort newest-first by the numeric suffix of the code, not lexicographically:
    # once codes reach 4 digits, "EMSR999" would sort after "EMSR1000" as a
    # string. Non-numeric/unmatched codes (shouldn't occur) sort last, with the
    # raw code as a stable tiebreak.
    order = pd.to_numeric(
        df["code"].str.extract(r"(\d+)", expand=False), errors="coerce"
    )
    df = (
        df.assign(_order=order)
        .sort_values(["_order", "code"], ascending=False)
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return ACTIVATION_SCHEMA.validate(df)


def get_activation(code: str) -> Dict[str, Any]:
    """Fetch the full nested detail tree for one activation code.

    Returns the raw (lightly-validated) activation dict — activation →
    ``aois`` → ``products`` → ``layers``/``images``/``stats`` — rather than a
    DataFrame, because the hierarchy doesn't flatten to a single table without
    losing the AOI/product/layer relationships. Use :func:`get_products`,
    :func:`get_catalog`, and :func:`get_stats` for tabular views.

    Parameters
    ----------
    code : str
        Activation code, e.g. ``"EMSR884"``.

    Raises
    ------
    ActivationNotFoundError
        The endpoint returned no result for ``code``.
    """
    data = _get_json(_DETAIL_URL, params={"code": code})
    results = data.get("results") or []
    if not results:
        raise ActivationNotFoundError(
            f"no CEMS activation found for code {code!r}"
        )
    return results[0]


# ---------------------------------------------------------------------------
# Flatteners
# ---------------------------------------------------------------------------


def _iter_products(activation: Dict[str, Any]):
    """Yield (aoi, product) pairs across every AOI of an activation."""
    for aoi in activation.get("aois", []) or []:
        for product in aoi.get("products", []) or []:
            yield aoi, product


def get_products(ref: ActivationRef) -> pd.DataFrame:
    """Flatten an activation to one row per product (the download targets).

    This is the primary entry point for the bulk-download workflow: each row
    carries the product's ``download_url`` (its zip) plus enough metadata to
    filter (type, AOI, feasibility, version/status).

    Parameters
    ----------
    ref : str | dict
        An activation code (fetched via :func:`get_activation`) or an
        already-fetched activation dict.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`PRODUCT_SCHEMA`, one row per (AOI, product).
    """
    activation = _resolve_activation(ref)
    code = activation["code"]
    rows: List[Dict[str, Any]] = []
    for aoi, p in _iter_products(activation):
        version = p.get("version") or {}
        rows.append(
            {
                "code": code,
                "aoi_number": aoi.get("number"),
                "aoi_name": _clean_str(aoi.get("name")),
                "product_id": p.get("id"),
                "product_type": p.get("type"),
                "monitoring": p.get("monitoring"),
                "monitoring_number": p.get("monitoringNumber"),
                "feasible": p.get("feasible"),
                "status_code": _clean_str(version.get("statusCode")),
                "version_number": version.get("number"),
                "delivery_time": _clean_dt(version.get("deliveryTime")),
                "expected_delivery": _clean_dt(p.get("expectedDelivery")),
                "n_layers": len(p.get("layers") or []),
                "n_images": len(p.get("images") or []),
                "download_url": _clean_str(p.get("downloadPath")),
            }
        )

    if not rows:
        return PRODUCT_SCHEMA.validate(pd.DataFrame(columns=_PRODUCT_COLS))
    return PRODUCT_SCHEMA.validate(pd.DataFrame(rows, columns=_PRODUCT_COLS))


def get_catalog(ref: ActivationRef) -> pd.DataFrame:
    """Flatten an activation to one row per layer (individual geo files).

    Each layer row exposes the directly-downloadable ``geojson_url`` and its
    ``sld_url`` style file. Products that publish no individual layers still
    appear as a single row (layer fields null) so their ``product_zip_url`` is
    never dropped.

    Parameters
    ----------
    ref : str | dict
        Activation code or an already-fetched activation dict.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`CATALOG_SCHEMA`.
    """
    activation = _resolve_activation(ref)
    code = activation["code"]
    rows: List[Dict[str, Any]] = []
    for aoi, p in _iter_products(activation):
        base = {
            "code": code,
            "aoi_number": aoi.get("number"),
            "aoi_name": _clean_str(aoi.get("name")),
            "product_id": p.get("id"),
            "product_type": p.get("type"),
            "product_zip_url": _clean_str(p.get("downloadPath")),
        }
        layers = p.get("layers") or []
        if not layers:
            rows.append(
                {
                    **base,
                    "layer_name": None,
                    "layer_format": None,
                    "geojson_url": None,
                    "sld_url": None,
                }
            )
            continue
        for layer in layers:
            rows.append(
                {
                    **base,
                    "layer_name": _clean_str(layer.get("name")),
                    "layer_format": _clean_str(layer.get("format")),
                    "geojson_url": _clean_str(layer.get("json")),
                    "sld_url": _clean_str(layer.get("sld")),
                }
            )

    if not rows:
        return CATALOG_SCHEMA.validate(pd.DataFrame(columns=_CATALOG_COLS))
    return CATALOG_SCHEMA.validate(pd.DataFrame(rows, columns=_CATALOG_COLS))


def get_stats(ref: ActivationRef) -> pd.DataFrame:
    """Flatten the per-product damage-statistics tables.

    The service nests stats as ``category → subcategory → {unit, total,
    affected}``. This emits one row per (product, category, subcategory).
    Totals/affected are coerced to float; placeholder strings (e.g. ``"NA"``)
    become null.

    Parameters
    ----------
    ref : str | dict
        Activation code or an already-fetched activation dict.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`STATS_SCHEMA`. Products without a stats table
        contribute no rows.
    """
    activation = _resolve_activation(ref)
    code = activation["code"]
    rows: List[Dict[str, Any]] = []
    for aoi, p in _iter_products(activation):
        stats = p.get("stats")
        if not isinstance(stats, dict):
            continue
        for category, subcats in stats.items():
            if not isinstance(subcats, dict):
                continue
            for subcat, vals in subcats.items():
                vals = vals or {}
                rows.append(
                    {
                        "code": code,
                        "aoi_number": aoi.get("number"),
                        "aoi_name": _clean_str(aoi.get("name")),
                        "product_id": p.get("id"),
                        "product_type": p.get("type"),
                        "category": category,
                        "subcategory": _clean_str(subcat),
                        "unit": _clean_str(vals.get("unit")),
                        "total": _to_float(vals.get("total")),
                        "affected": _to_float(vals.get("affected")),
                    }
                )

    if not rows:
        return STATS_SCHEMA.validate(pd.DataFrame(columns=_STATS_COLS))
    return STATS_SCHEMA.validate(pd.DataFrame(rows, columns=_STATS_COLS))


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------


def download_file(
    url: str, dest: Optional[Union[str, Path]] = None
) -> Union[bytes, Path]:
    """Download a URL to bytes (in memory) or to a local file.

    The generic primitive the other downloaders build on. When ``dest`` is
    given the body is streamed to disk in chunks (so large product zips/COGs
    don't have to fit in memory) and the written :class:`~pathlib.Path` is
    returned; otherwise the full content is returned as ``bytes``.

    Parameters
    ----------
    url : str
        File URL.
    dest : str | Path, optional
        Local destination. If it is an existing directory (or ends with a
        path separator), the server-side filename is appended.

    Returns
    -------
    bytes | pathlib.Path
    """
    if dest is None:
        resp = _session.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    dest = Path(dest)
    if dest.is_dir() or str(dest).endswith(("/", "\\")):
        dest = dest / _url_filename(url)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with _session.get(url, timeout=_TIMEOUT, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
    return dest


def _product_url(product: Union[str, Dict[str, Any], pd.Series]) -> str:
    """Resolve a product reference to its zip download URL.

    Accepts a raw URL, a product dict from the detail tree (``downloadPath``),
    or a catalog/product DataFrame row (``download_url``/``product_zip_url``).
    """
    if isinstance(product, str):
        return product
    if isinstance(product, pd.Series):
        product = product.to_dict()
    for key in ("download_url", "downloadPath", "product_zip_url"):
        val = product.get(key)
        if val:
            return val
    raise ValueError(
        "could not resolve a download URL from product reference "
        f"(keys: {list(product.keys())})"
    )


def download_product(
    product: Union[str, Dict[str, Any], pd.Series],
    dest: Optional[Union[str, Path]] = None,
) -> Union[bytes, Path]:
    """Download a single product's zip.

    Parameters
    ----------
    product : str | dict | pandas.Series
        A product zip URL, a product dict (from :func:`get_activation`), or a
        row from :func:`get_products` / :func:`get_catalog`.
    dest : str | Path, optional
        Local destination (file or directory). None → return bytes.

    Returns
    -------
    bytes | pathlib.Path
    """
    return download_file(_product_url(product), dest=dest)


def download_products(
    ref: ActivationRef,
    dest_dir: Optional[Union[str, Path]] = None,
    product_types: Optional[List[str]] = None,
    aoi_numbers: Optional[List[int]] = None,
    feasible_only: bool = True,
) -> Dict[str, Union[bytes, Path]]:
    """Download every (filtered) product of an activation.

    The headline bulk-download helper. Iterates the activation's products,
    applies the optional filters, and downloads each one — to memory by
    default, or to ``dest_dir`` on disk.

    Parameters
    ----------
    ref : str | dict
        Activation code or an already-fetched activation dict.
    dest_dir : str | Path, optional
        Directory to write zips into (created if missing). None → return
        each product's bytes in memory.
    product_types : list[str], optional
        Keep only these product types (e.g. ``["GRA"]`` for grading/damage).
        None keeps all types.
    aoi_numbers : list[int], optional
        Keep only these AOI numbers. None keeps all AOIs.
    feasible_only : bool, default True
        Skip products marked ``feasible=False`` (requested but not produced,
        so they have no usable zip).

    Returns
    -------
    dict[str, bytes | pathlib.Path]
        Maps each product's zip filename to its bytes (in memory) or written
        path (``dest_dir`` given). Products with no published download URL —
        the normal state for products still ``W`` (waiting) / ``N`` (not
        produced) — are skipped; a one-line INFO summary reports the counts,
        with per-product detail logged at DEBUG.
    """
    activation = _resolve_activation(ref)
    code = activation.get("code")
    types = {t.upper() for t in product_types} if product_types else None
    aois = set(aoi_numbers) if aoi_numbers is not None else None

    out: Dict[str, Union[bytes, Path]] = {}
    n_no_url = 0
    n_infeasible = 0
    for aoi, p in _iter_products(activation):
        if feasible_only and p.get("feasible") is False:
            n_infeasible += 1
            continue
        if types is not None and str(p.get("type", "")).upper() not in types:
            continue
        if aois is not None and aoi.get("number") not in aois:
            continue
        url = p.get("downloadPath")
        if not url:
            # Expected, not actionable: the product just isn't published yet.
            # Detail at DEBUG; the count rolls up into the summary below.
            n_no_url += 1
            logger.debug(
                "CEMS %s AOI %s product %s (%s): no download URL, skipping",
                code,
                aoi.get("number"),
                p.get("id"),
                p.get("type"),
            )
            continue
        name = _url_filename(url)
        out[name] = download_file(url, dest=dest_dir)
        logger.debug("CEMS downloaded product %s", name)

    logger.info(
        "CEMS %s download_products: %d downloaded, %d skipped (no published "
        "zip), %d infeasible",
        code,
        len(out),
        n_no_url,
        n_infeasible,
    )
    return out


def download_activation_bundle(
    ref: ActivationRef, dest: Optional[Union[str, Path]] = None
) -> Union[bytes, Path]:
    """Download the activation-wide ``_products.zip`` bundle.

    A single archive containing the latest version of every product for the
    activation (the ``productsPath`` field). Convenient when you want
    everything in one shot rather than per-product via
    :func:`download_products`.

    Parameters
    ----------
    ref : str | dict
        Activation code or an already-fetched activation dict.
    dest : str | Path, optional
        Local destination. None → return bytes.

    Raises
    ------
    ValueError
        The activation has no ``productsPath``.
    """
    activation = _resolve_activation(ref)
    url = activation.get("productsPath")
    if not url:
        raise ValueError(
            f"activation {activation.get('code')!r} has no products bundle URL"
        )
    return download_file(url, dest=dest)


def download_geojson(
    layer: Union[str, Dict[str, Any], pd.Series],
) -> gpd.GeoDataFrame:
    """Download one layer's GeoJSON into a GeoDataFrame (in memory).

    Parameters
    ----------
    layer : str | dict | pandas.Series
        A GeoJSON URL, a layer dict from the detail tree (``json`` key), or a
        catalog row from :func:`get_catalog` (``geojson_url``).

    Returns
    -------
    geopandas.GeoDataFrame

    Raises
    ------
    ValueError
        No GeoJSON URL could be resolved from ``layer``.
    """
    if isinstance(layer, str):
        url = layer
    else:
        if isinstance(layer, pd.Series):
            layer = layer.to_dict()
        url = layer.get("geojson_url") or layer.get("json")
        if not url:
            raise ValueError(
                "could not resolve a GeoJSON URL from layer reference "
                f"(keys: {list(layer.keys())})"
            )
    content = download_file(url)  # bytes
    return gpd.read_file(BytesIO(content))


# ---------------------------------------------------------------------------
# Optional blob persistence (thin wrapper; ocha-stratus is not a hard dep)
# ---------------------------------------------------------------------------


def to_blob(
    data: bytes,
    blob_name: str,
    stage: str = "dev",
    container_name: str = _DEFAULT_BLOB_CONTAINER,
) -> None:
    """Upload bytes to the OCHA Azure blob store via ``ocha-stratus``.

    A thin convenience wrapper so a download can be pushed straight to blob,
    e.g. ``to_blob(download_product(row), "cems/EMSR884/grading.zip")``.
    ``ocha-stratus`` is imported lazily here so it stays an optional
    dependency — the rest of this module works without it.

    Parameters
    ----------
    data : bytes
        Payload (e.g. the return of :func:`download_product` /
        :func:`download_file` with ``dest=None``).
    blob_name : str
        Destination blob path/key.
    stage : {"dev", "prod"}, default "dev"
        Azure stage.
    container_name : str, default "raster"
        Target container.

    Raises
    ------
    ImportError
        ``ocha-stratus`` is not installed.
    """
    try:
        import ocha_stratus as stratus
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised where stratus absent
        raise ImportError(
            "to_blob requires ocha-stratus; install it (it is an optional "
            "dependency of ocha-lens) to use blob persistence."
        ) from exc

    stratus.upload_blob_data(
        data=data,
        blob_name=blob_name,
        stage=stage,
        container_name=container_name,
    )
