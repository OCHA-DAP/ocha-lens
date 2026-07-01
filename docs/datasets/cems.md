# Copernicus EMS Rapid Mapping

This module provides access to [Copernicus EMS Rapid Mapping](https://rapidmapping.emergency.copernicus.eu/) emergency activation products. The Copernicus Emergency Management Service (CEMS) Rapid Mapping component publishes geospatial damage-assessment products for disasters (earthquakes, floods, wildfires, …). It is not storm-specific — the module works for any activation on the API.

Each **activation** (e.g. `EMSR884`, "Earthquake in Venezuela") is divided into one or more **AOIs** (areas of interest), each carrying one or more **products** (First Estimate, Delineation, Grading/damage assessment, …). Every product is downloadable as a zip and also exposed as individual **layers** (GeoJSON / vector tiles / COG, each with an SLD style file). Access is via two public, unauthenticated JSON endpoints — no API key required.

## Quick Start

### Finding activations

```python
import ocha_lens as lens

# All activations (one row per activation), or filter
df = lens.cems.get_activations()
quakes = lens.cems.get_activations(category="Earthquake", closed=False)
venezuela = lens.cems.get_activations(country="venezuela")
```

`get_activations` walks the paginated listing endpoint and returns a tidy table sorted newest-first. Filtering (category / closed / country) is applied client-side. Pass `closed=False` for the catalog of ongoing activations.

The `product_type` codes that appear in `get_products`/`get_catalog` (e.g. `GRA`) are described in the `lens.cems.PRODUCT_TYPES` dictionary — `REF` (reference), `FEP` (first estimate), `DEL` (delineation), `GRA` (grading/damage), `GRM` (ground movement). Map it onto a products table with `products["product_type"].map(lens.cems.PRODUCT_TYPES)`.

### Inspecting one activation

```python
# Full nested detail tree (activation → AOIs → products → layers/stats)
act = lens.cems.get_activation("EMSR884")

# Flatten to tabular views (each accepts a code or the fetched dict)
products = lens.cems.get_products(act)   # one row per product (download targets)
catalog = lens.cems.get_catalog(act)     # one row per layer (geojson/sld URLs)
stats = lens.cems.get_stats(act)         # per-product damage-statistics table
```

`get_activation` returns the raw dict rather than a DataFrame because the hierarchy doesn't flatten to a single table without losing the AOI/product/layer relationships. The three flatteners give the tabular views.

### Downloading products (primary workflow)

```python
# Every produced product for the activation, in memory ({filename: bytes})
zips = lens.cems.download_products("EMSR884")

# Filter by product type / AOI, and write to disk instead of memory
lens.cems.download_products(
    "EMSR884",
    product_types=["GRA"],   # grading / damage assessment only
    aoi_numbers=[0, 2],
    dest_dir="./emsr884",    # returns {filename: Path}
)

# A single product (from a get_products row, a product dict, or a URL)
row = products.iloc[0]
data = lens.cems.download_product(row)                  # bytes
path = lens.cems.download_product(row, dest="./out")    # Path

# The whole-activation bundle (one big zip of every product)
lens.cems.download_activation_bundle("EMSR884", dest="./emsr884.zip")
```

By default `download_products` skips products marked `feasible=False` (requested but not produced) and those with no published zip URL (logged at WARNING).

### Working with individual layers

```python
# Pull one layer's GeoJSON straight into a GeoDataFrame
geojson_layers = catalog[catalog["geojson_url"].notna()]
gdf = lens.cems.download_geojson(geojson_layers.iloc[0])

# Generic download primitive for anything (COG, SLD, …)
raw = lens.cems.download_file(url)              # bytes
path = lens.cems.download_file(url, dest="./")  # streamed to disk
```

### Persisting to blob

```python
# Push any downloaded bytes to the OCHA Azure blob store
data = lens.cems.download_product(row)
lens.cems.to_blob(data, "cems/EMSR884/grading.zip", stage="dev")
```

`to_blob` is a thin wrapper around `ocha-stratus`, imported lazily so it stays an optional dependency — the rest of the module works without it.

## Notes

- **Host:** the module targets `rapidmapping.emergency.copernicus.eu`. The `mapping.emergency.copernicus.eu` host that appears in some published OpenAPI specs 404s.
- **In-memory by default:** every download returns bytes / GeoDataFrames unless you pass `dest` / `dest_dir`. Layer GeoJSONs can be large (tens of MB), so prefer streaming to disk for bulk work.
- **Mixed timestamp precision:** the service mixes ISO8601 precisions within one activation (some timestamps carry microseconds, some don't); they are parsed per-value rather than with a single inferred format.
- **Stats coercion:** statistic totals/affected are coerced to float; the service's placeholder strings (e.g. `"NA"`) become null.

See the [API reference](../api.md#copernicus-ems-rapid-mapping) for full function signatures.
