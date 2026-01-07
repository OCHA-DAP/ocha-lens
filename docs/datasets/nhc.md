# NHC Tropical Cyclone Data

This module provides an interface for accessing National Hurricane Center (NHC) and Central Pacific Hurricane Center (CPHC) tropical cyclone data in two modes: **real-time current storms** and **historical archive forecasts**. The data is processed into a standardized format suitable for immediate analysis and interoperable with other cyclone track data sources.

The [National Hurricane Center](https://www.nhc.noaa.gov/) is responsible for tropical cyclone forecasting in the North Atlantic and Eastern North Pacific basins, while the [Central Pacific Hurricane Center](https://www.prh.noaa.gov/cphc/) covers the Central Pacific. Together, they provide comprehensive coverage of tropical cyclones affecting North America and the Pacific.

## Data Modes

### Current Storms (Real-time API)
Downloads active storm data from NHC's CurrentStorms.json API, including current observations and forecast advisories. This mode fetches text-based forecast advisories and parses them to extract forecast positions and wind speeds.

**Coverage:** Active storms only
**Update Frequency:** Every 3-6 hours during active storms
**Data Fields:** Position, wind speed, pressure (observations only)

### Historical Archive (ATCF)
Downloads historical forecast data from the ATCF (Automated Tropical Cyclone Forecasting) archive. This provides official NHC forecasts from 1850-2024 in a standardized comma-delimited format.

**Coverage:** 1850-2024 (170+ years)
**Format:** ATCF A-deck files
**Data Fields:** Position, wind speed, pressure, forecast lead times

## Quick Start

### Current Storms
```python
import ocha_lens as lens

# Load current active storms
df = lens.nhc.load_nhc()

# Extract storm metadata
df_storms = lens.nhc.get_storms(df)

# Get track data
gdf_tracks = lens.nhc.get_tracks(df)

# Filter observations vs forecasts
observations = gdf_tracks[gdf_tracks.leadtime == 0]
forecasts = gdf_tracks[gdf_tracks.leadtime > 0]
```

### Historical Archive
```python
import ocha_lens as lens

# Load all Atlantic storms from 2023
df = lens.nhc.load_nhc(year=2023, basin="AL")

# Load specific storm (Hurricane Idalia - AL10)
df = lens.nhc.load_nhc(year=2023, storm_number=10, basin="AL")

# Load multiple storms
df = lens.nhc.load_nhc(
    year=2023,
    storm_number=[10, 14, 15],  # Idalia, Lee, Margot
    basin="AL"
)

# Load Eastern Pacific storms
df = lens.nhc.load_nhc(year=2024, basin="EP")

# Extract storms and tracks
df_storms = lens.nhc.get_storms(df)
gdf_tracks = lens.nhc.get_tracks(df)
```

## Output Data Structure

The primary goal of this module is to provide easy access to NHC data in a tabular, analysis-ready format. The schemas are designed to be interoperable with other cyclone track data sources (e.g., IBTrACS, ECMWF).

### `lens.nhc.get_storms()`

This function outputs a table containing one row per unique storm (identified by `atcf_id`). This data provides storm-level metadata and can be used to link forecast points to specific storms.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `atcf_id` | `str` | **Required** | Part of unique constraint | ATCF storm identifier (e.g., "AL102023") |
| `name` | `str` | Optional | - | Storm name, or None for unnamed systems |
| `number` | `str` | **Required** | - | Storm number (e.g., "10") |
| `season` | `int64` | **Required** | 2000-2050 range | Storm season year |
| `genesis_basin` | `str` | **Required** | Must match basin mapping[^1] | Storm's designation basin (where it originated)[^4] |
| `provider` | `str` | Optional | - | NHC or CPHC |
| `storm_id` | `str` | Optional | Part of unique constraint | Concatenation of `<name>_<basin>_<season>` (lowercase) |

See the enforced schema in the [source code](https://github.com/OCHA-DAP/ocha-lens/blob/main/src/ocha_lens/datasources/nhc.py).

### `lens.nhc.get_tracks()`

This function outputs track data for all forecast points, including both observations (leadtime=0) and forecasts (leadtime>0). Each row represents a single position at a specific valid time.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `atcf_id` | `str` | **Required** | Part of unique constraint | ATCF storm identifier |
| `point_id` | `str` | **Required** | - | Unique identifier (UUID) for this point |
| `storm_id` | `str` | Optional | Part of unique constraint | Links to storm metadata |
| `provider` | `str` | **Required** | - | NHC or CPHC |
| `basin` | `str` | **Required** | Must match basin mapping[^1] | Storm's designation basin (from ATCF ID)[^4] |
| `issued_time` | `pd.Timestamp` | **Required** | - | When the forecast/observation was issued |
| `valid_time` | `pd.Timestamp` | **Required** | Part of unique constraint | Time this position is valid for |
| `leadtime` | `Int64` | **Required** | ≥ 0, Part of unique constraint | Hours ahead of issue time (0 for observations)[^2] |
| `wind_speed` | `float` | Optional | 0-300 knots range | Maximum sustained winds |
| `pressure` | `float` | Optional | 800-1100 hPa range | Central pressure[^3] |
| `quadrant_radius_34` | `list` | Optional | List of 4 integers [NE, SE, SW, NW] | 34-knot wind radii by quadrant (nautical miles) |
| `quadrant_radius_50` | `list` | Optional | List of 4 integers [NE, SE, SW, NW] | 50-knot wind radii by quadrant (nautical miles) |
| `quadrant_radius_64` | `list` | Optional | List of 4 integers [NE, SE, SW, NW] | 64-knot wind radii by quadrant (nautical miles) |
| `number` | `str` | Optional | - | Storm number |
| `geometry` | `gpd.array.GeometryDtype` | **Required** | EPSG:4326, valid lat/lon | Geographic location |

See the enforced schema in the [source code](https://github.com/OCHA-DAP/ocha-lens/blob/main/src/ocha_lens/datasources/nhc.py).

## Usage Considerations

### Current API vs Archive Data

**Current API:**
- Pressure available only for observations (current position), not forecasts
- Wind radii parsed from text advisories (when available)
- Forecasts parsed from text advisories (may have parsing limitations)
- Only includes currently active storms
- Updated every 3-6 hours

**Archive Data:**
- Pressure available for both observations and forecasts
- Wind radii available in standardized ATCF format
- Standardized ATCF format (more reliable)
- Complete historical record (1850-2024)
- Includes all official NHC forecasts

### Basin Coverage

NHC/CPHC only covers three basins:
- **AL** (Atlantic / North Atlantic) - NHC responsibility
- **EP** (Eastern Pacific) - NHC responsibility
- **CP** (Central Pacific) - CPHC responsibility

For global coverage, use the IBTrACS or ECMWF modules.

### Basin Designation vs Geographic Location

**Important:** The `basin` field represents a storm's **designation basin** (where it originated), not its current geographic location. This is a fundamental aspect of the ATCF system.

**Key Points:**
- Storms retain their original basin designation throughout their lifecycle
- The basin field comes from the ATCF ID and never changes
- Storms that cross basin boundaries keep their original designation

**Example:**
Hurricane Gilma (2024) originated in the Eastern Pacific with ATCF ID `EP072024`. When it crossed 140°W into the Central Pacific on August 27, 2024:
- Geographic location: Central Pacific (west of 140°W)
- Forecast responsibility: Transferred from NHC to CPHC
- `basin` field in data: Remained **"EP"** (Eastern Pacific)
- ATCF ID: Remained `EP072024`

This means all track points for Gilma have `basin='EP'`, even those west of 140°W. To determine actual geographic location, use the `longitude` field:
- East of 140°W = Eastern Pacific geographic region
- 140°W to 180°W = Central Pacific geographic region

**Why This Matters:**
- For filtering storms by origin: Use the `basin` field
- For geographic analysis: Use `longitude` and `latitude` coordinates
- For understanding forecast ownership: The designation basin indicates which agency "owns" the storm's data

### Forecast vs Best Track Data

The NHC module provides **forecast data** (what NHC predicted at the time), not post-season best track data. For verified "ground truth" observations:
- Use `lens.ibtracs` for historical best tracks
- Or implement HURDAT2 parsing (best track data from NHC)

Forecast data is valuable for:
- Forecast verification studies
- Understanding forecast uncertainty
- Real-time decision making
- Analyzing forecast skill evolution

### Lead Time Conventions

The `leadtime` field indicates the hours ahead of the issue time for each track point:

- **`leadtime = 0`**: Observations (current position) - filter with `tracks[tracks.leadtime == 0]`
- **`leadtime > 0`**: Forecasts - filter with `tracks[tracks.leadtime > 0]`
- **Standard forecast hours**: 12, 24, 36, 48, 72, 96, 120

**Forecast vs Outlook:**

NHC distinguishes between **forecasts** (≤120h) and **outlooks** (>120h) based on forecast confidence:

- **Forecasts (0-120h)**: Standard operational forecasts with detailed predictions (position, intensity, wind radii). These appear as "FORECAST VALID" in text advisories and are included in the forecast cone.

- **Outlooks (>120h)**: Extended outlooks beyond the 5-day forecast period with lower confidence and less detail. These appear as "OUTLOOK VALID" in text advisories when NHC expects the system to persist beyond 5 days.

The 120-hour (5-day) cutoff reflects NHC's operational forecast period. Forecast errors increase significantly with lead time (averaging ~60 nm at 24h, ~200+ nm at 120h), making predictions beyond 5 days much less reliable.

See [How To Read The Forecast/Advisory](https://www.nhc.noaa.gov/help/tcm.shtml) for NHC's official documentation on forecast terminology.

### Multiple Forecasts Per Storm

Archive data contains multiple forecasts issued at different times for the same storm. To analyze forecast evolution:

```python
# Get all 24-hour forecasts for a storm
forecasts_24h = gdf_tracks[
    (gdf_tracks.atcf_id == "AL102023") &
    (gdf_tracks.leadtime == 24)
]

# Group by issue time to see forecast evolution
forecasts_24h.groupby('issued_time')[['latitude', 'longitude']].mean()
```

### Handling Storms that Cross the Antimeridian

All track points are normalized to the [-180, 180] longitude range. Analyses such as distance calculations close to the antimeridian may require special handling.

### Unnamed Systems

Not all systems tracked by NHC receive names. Invest systems (potential developments) and weak systems may have `name = None`. These systems will not have a `storm_id` (which requires a name).

```python
# Filter for named storms only
named_storms = df_storms[df_storms.storm_id.notna()]

# Get all systems (including unnamed)
all_systems = df_storms
```

## File Storage Structure

```
cache_dir/
├── raw/
│   ├── nhc_YYYYMMDD_HHMM.json          # Current storms (JSON)
│   ├── atcf/
│   │   ├── atcf_al_01_2023.dat         # Atlantic storm 1, 2023
│   │   ├── atcf_al_10_2023.dat         # Hurricane Idalia
│   │   ├── atcf_ep_09_2023.dat         # Eastern Pacific storm 9
│   │   └── atcf_cp_01_2024.dat         # Central Pacific storm 1
```

Files are automatically cached to avoid re-downloading. Set `use_cache=False` to force fresh downloads.

## Additional Resources

- [NHC Official Website](https://www.nhc.noaa.gov/)
- [CPHC Official Website](https://www.prh.noaa.gov/cphc/)
- [NHC Data Archive](https://www.nhc.noaa.gov/data/)
- [How To Read The Forecast/Advisory](https://www.nhc.noaa.gov/help/tcm.shtml) - Official guide to NHC forecast terminology
- [ATCF Archive (FTP)](https://ftp.nhc.noaa.gov/atcf/archive/)
- [ATCF README](https://ftp.nhc.noaa.gov/atcf/README) - Official NHC ATCF format documentation
- [ATCF Data Parser](https://palewi.re/docs/atcf-data-parser/) - Python parser with format details
- [ATCF on Wikipedia](https://en.wikipedia.org/wiki/Automated_Tropical_Cyclone_Forecasting_System) - Background on ATCF system
- [NHC Forecast Verification](https://www.nhc.noaa.gov/verification/)
- [HURDAT2 Best Track Data](https://www.nhc.noaa.gov/data/hurdat/)
- [Current Storms JSON API](https://www.nhc.noaa.gov/CurrentStorms.json)

[^1]: NHC basins are `NA` (North Atlantic, mapped from AL), `EP` (Eastern Pacific), and `CP` (Central Pacific). These codes are standardized to match IBTrACS and ECMWF conventions.

[^2]: Use `leadtime` to distinguish between observations and forecasts: `leadtime == 0` for current positions (observations), `leadtime > 0` for forecasts. Standard forecast intervals are 12, 24, 36, 48, 72, 96, and 120 hours.

[^3]: Pressure is only available in observations (leadtime=0) for current API data. Archive data includes pressure for both observations and forecasts when available in the ATCF record.

[^4]: The `basin` field represents the storm's designation basin from the ATCF ID, not its current geographic location. Storms that cross basin boundaries (e.g., Eastern Pacific storms crossing 140°W into Central Pacific) retain their original basin designation. See the "Basin Designation vs Geographic Location" section for details.
