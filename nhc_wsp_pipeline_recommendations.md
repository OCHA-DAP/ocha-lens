# NHC Wind Speed Probability Pipeline — Recommendations

## Summary recommendation

Use the **GRIB2 FTP archive** as the single source for both historical backfill and operational updates. It has the longest archive (2006–present), a stable format, and a deterministic URL pattern. Generate vector polygons locally following NHC's own method (bilinear interpolation → bin → vectorise), and store them in PostGIS alongside the existing track tables.

---

## Source selection

| Need | Source | Rationale |
|---|---|---|
| Historical (2006–present) | GRIB2 FTP archive | Longest consistent archive, stable format, no gaps observed |
| Operational updates | GRIB2 FTP archive (same path) | Preliminary file available ~1 hr after cycle; archive copy ~2 hrs after. Use archive copy for simplicity — only 1 hr behind preliminary |
| Pre-2006 | Not available | No GRIB2 before 2006-06-15; GRIB1 available from 2005-06 but requires separate parameter decoding |

Avoid maintaining two fetch paths (preliminary + archive). The archive copy appears within 2 hours of issuance; this is acceptable latency for most operational uses. If sub-2-hour latency is required, add a preliminary fetch as a secondary path.

---

## PostGIS schema

Two new tables, designed to sit alongside the existing `nhc_storm` and `nhc_track` tables.

```sql
-- One row per (issuance, wind threshold, probability band)
-- Equivalent schema to the official NHC 5km GIS shapefiles
CREATE TABLE nhc_wsp_polygon (
    id                BIGSERIAL PRIMARY KEY,
    issuance          TIMESTAMPTZ NOT NULL,
    wind_threshold_kt SMALLINT    NOT NULL CHECK (wind_threshold_kt IN (34, 50, 64)),
    percentage        VARCHAR(8)  NOT NULL,   -- '<5%', '5-10%', ..., '>90%'
    geom              GEOMETRY(MULTIPOLYGON, 4326) NOT NULL
);

CREATE INDEX ON nhc_wsp_polygon (issuance);
CREATE INDEX ON nhc_wsp_polygon (wind_threshold_kt);
CREATE INDEX ON nhc_wsp_polygon USING GIST (geom);

-- Storm–issuance link table: which ATCF IDs were active at each issuance
-- This is the join key between wsp_polygon and nhc_storm/nhc_track
CREATE TABLE nhc_wsp_storm_link (
    issuance TIMESTAMPTZ  NOT NULL,
    atcf_id  VARCHAR(8)   NOT NULL REFERENCES nhc_storm(atcf_id),
    PRIMARY KEY (issuance, atcf_id)
);

CREATE INDEX ON nhc_wsp_storm_link (atcf_id);
```

**Why a separate link table?** The probability field is basin-wide — one file covers all active storms simultaneously. The link table records which storms were active at each issuance, enabling joins to `nhc_track` without duplicating the geometry for each storm.

---

## Linking to track data

```sql
-- All probability polygons relevant to a specific storm
SELECT p.issuance, p.wind_threshold_kt, p.percentage, p.geom
FROM nhc_wsp_polygon p
JOIN nhc_wsp_storm_link l USING (issuance)
WHERE l.atcf_id = 'AL082023';

-- Probability at a storm's forecast track positions
-- (spatial: which probability band does each track point fall in?)
SELECT t.valid_time, t.wind_speed, p.percentage
FROM nhc_track t
JOIN nhc_wsp_storm_link l ON l.atcf_id = t.atcf_id
JOIN nhc_wsp_polygon p
  ON p.issuance = l.issuance
  AND p.wind_threshold_kt = 34
  AND ST_Within(t.geometry, p.geom)
WHERE t.atcf_id = 'AL082023';
```

The link table is populated at fetch time from `CurrentStorms.json` (operational) or by cross-referencing issuance timestamps against the ATCF storm table (archive). For the historical backfill, NHC's [storm table](https://ftp.nhc.ncep.noaa.gov/atcf/archive/storm.table) lists season, dates, and ATCF IDs for every storm — use this to determine which storms were active at each 6-hourly GRIB2 issuance.

---

## Pipeline design

### Backfill (run once)

```
for each YYYY/MM in 2006-06 → present:
    list files from ftp.nhc.ncep.noaa.gov/wsp/YYYY/MM/
    for each tpcprblty.YYYYMMDDHH.grib2.gz:
        fetch + decompress → tempfile
        extract 34/50/64 kt, stepRange=0-120 (cfgrib filter_by_keys)
        bilinear interpolate 0.5° → 5km (scipy.interpolate.RegularGridInterpolator)
        bin → vectorise (rasterio.features.shapes)
        upsert nhc_wsp_polygon
        populate nhc_wsp_storm_link from storm.table
        delete tempfile
```

At ~160 KB/file compressed, ~28,000 files total ≈ 4.5 GB download. Process sequentially with a simple retry loop; no parallelism needed given the one-time nature.

### Operational update (run every 6 hours)

```
compute expected issuance = floor(utcnow, 6h) - 2h buffer
if issuance not in nhc_wsp_polygon:
    fetch GRIB2 from archive URL
    run same extract → interpolate → vectorise → upsert pipeline
    populate nhc_wsp_storm_link from CurrentStorms.json
```

This is identical code to the backfill — no separate operational path to maintain.

---

## Consistency with existing module

| Concern | Approach |
|---|---|
| **ATCF ID casing** | Normalise to uppercase (`AL082023`) — matches existing `nhc_storm.atcf_id` |
| **Timestamps** | Store as `TIMESTAMPTZ` (UTC) — matches `issued_time` in `nhc_track` |
| **CRS** | EPSG:4326 throughout — matches `nhc_track.geometry` |
| **Upsert safety** | Use `ON CONFLICT DO NOTHING` on `(issuance, wind_threshold_kt, percentage)` — safe to re-run |
| **Dual-mode pattern** | Operational and archive use the same fetch/transform function, same as `load_nhc()` |
| **`load_nhc_wsp()`** | Follow the same signature: no args = current; `issuance=` or `year=` = archive |

---

## What to deprioritise

- **GIS shapefiles**: shorter archive (from 2017 only), stopped updating after 2025-10-31, adds `geopandas` zip-handling complexity. Only useful if exact NHC polygon boundaries are required rather than the equivalent locally-generated ones.
- **GRIB1**: same data as GRIB2 but with opaque NWS local parameter codes that `cfgrib` cannot decode without a custom table. No benefit over GRIB2.
- **Preliminary file** (`/wsp/download/`): 1–2 hr advantage over archive copy, at the cost of a separate filename pattern and an uncompressed 1.9 MB file. Not worth the added code path unless latency is critical.
- **Incremental 6-hr step ranges**: available in GRIB2 but not in the GIS products. Store only if there is a specific use case — they add 19× more rows per issuance.
