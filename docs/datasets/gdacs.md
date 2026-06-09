# GDACS Tropical Cyclones

This module provides an interface to the [GDACS](https://www.gdacs.org/) (Global Disaster Alert and Coordination System) tropical cyclone API. It exposes event and episode metadata, per-advisory storm timelines, country- and admin-level population exposure, and a function to match a GDACS event to an NHC `atcf_id`. No authentication is required.

GDACS aggregates tropical cyclone advisories from multiple forecasting agencies (NOAA/NHC, JTWC). For storms forecast by the NHC, the GDACS forecast cone is the same underlying forecast as the [NHC data](nhc.md), which is what makes matching a GDACS event to an NHC `atcf_id` possible (see [Matching to NHC](#matching-to-nhc)).

## Quick Start

### Finding events

```python
import ocha_lens as lens

# Active and recent TC events (all alert levels)
gdf_events = lens.gdacs.get_events()

# Filter by date range and source
gdf_events = lens.gdacs.get_events(
    from_date="2024-09-01",
    to_date="2024-10-01",
    source="NOAA",
)
```

### Storm timeline

```python
# Per-advisory snapshots: position, wind speed, and population exposure
df_timeline = lens.gdacs.get_timeline(eventid=1000123)
```

Each row is one advisory. The `actual` flag distinguishes observed positions (`"True"`) from forecast-cone positions (`"False"`). `pop39`/`pop74` are instantaneous population snapshots within the 39 kt / 74 kt wind bands — not cumulative.

### Population exposure

```python
# Cumulative population exposure per wind buffer (~2022 onward)
adm0 = lens.gdacs.get_exposure_adm0(eventid=1000123)  # {"buffer39": df, "buffer74": df}
adm1 = lens.gdacs.get_exposure_adm1(eventid=1000123)  # sub-national grain
```

Each call returns a dict keyed by wind buffer (`"buffer39"`, `"buffer74"`). The `iso3` column carries GDACS' native `GMI_CNTRY` code — apply `lens.gdacs.to_iso3()` to standardize the handful of X-prefixed territory codes to ISO 3166-1 alpha-3.

### Matching to NHC

```python
# Match a GDACS timeline to an NHC atcf_id
atcf_id = lens.gdacs.match_to_atcf(df_timeline, nhc_tracks)
```

`match_to_atcf` votes the GDACS forecast-cone points onto NHC `atcf_id`s by exact `valid_time` (the shared forecast cone agrees to ~0°), with a single-point genesis match as a fallback for completed storms. It returns `None` for non-NHC (JTWC/RSMC) storms. The caller must pass NHC tracks de-duplicated to one row per `(atcf_id, valid_time)` at the freshest issuance — see the function's API reference for the exact contract.

## Notes

- **Coverage:** timelines are available for events from 2015 onward; per-buffer population exposure is available for events from roughly 2022 onward.
- **Errors:** the module raises explicit exceptions (`NoEpisodesError`, `NoTimelineError`, `EpisodeUrlFormatError`) rather than silently returning empty data, so API-contract changes surface loudly.

See the [API reference](../api.md#gdacs-tropical-cyclone-data) for full function signatures.
