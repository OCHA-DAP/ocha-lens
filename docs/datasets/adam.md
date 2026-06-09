# ADAM Tropical Cyclones

This module provides access to WFP [ADAM](https://gis.wfp.org/adam/) (Automatic Disaster Analysis and Mapping) tropical cyclone population-exposure data. It exposes a paginated event listing and per-event admin-level (ADM0/ADM1/ADM2) population exposure, returned in a long-form, analysis-ready shape.

ADAM uses the same `event_id` as [GDACS](gdacs.md) for the same physical storm, so the two sources join directly on `event_id` — no separate cross-source matching pass is needed.

## Quick Start

### Finding events

```python
import ocha_lens as lens

# Latest episode per event (cumulative snapshot at storm end), NOAA source
df_events = lens.adam.get_events(source="NOAA")

# Every episode for every event, over an overlap window
df_events = lens.adam.get_events(
    from_date="2024-09-01",
    to_date="2024-10-15",
    all_episodes=True,
)
```

By default `get_events` de-duplicates to the latest episode per `event_id`. Pass `all_episodes=True` to keep every `(event_id, episode_id)` row, each with its own `population_csv_url`. Date filtering is an **overlap** window (an event is kept if it was active at any point in the range), compared at date granularity.

### Population exposure

```python
# One event's exposure, long-form across ADM0/ADM1/ADM2 and wind thresholds
row = df_events.iloc[0]
df_exposure = lens.adam.get_exposure(
    event_id=row["event_id"],
    population_csv_url=row["population_csv_url"],
)
```

`get_exposure` downloads the per-episode CSV, converts the per-band counts to cumulative ≥-threshold exposure (for parity with GDACS), aggregates the native ADM2 rows up to ADM1 and ADM0, and maps the country name to ISO3. The result has one row per `(admin_level, admin_unit, wind_speed_kt)`, with `wind_speed_kt` in the standard NHC convention (34/50/64 kt).

## Notes

- **Two CSV schemas:** ADAM has published two column layouts — historical `ADM*_NAME` headers and the newer (2026) `LEVEL_n` headers with embedded `"- TOT"` subtotal rows. Both are handled; the pre-aggregated TOT rows are dropped so they don't double-count into the ADM0/ADM1 aggregates.
- **Cumulative conversion:** `lens.adam.make_cumulative` is exposed separately if you need the per-band → ≥-threshold transform on your own frame.
- **ISO3 resolution:** `lens.adam.name_to_iso3` resolves an ADM0 name to ISO3 (override table first, then `pycountry` fuzzy search), returning `None` for genuinely unresolvable names — the `iso3` column is nullable for that reason.
- **Missing CSVs:** events whose latest episode has no `population_csv_url` raise `NoExposureCSVError`; callers typically skip and retry next cycle.

See the [API reference](../api.md#adam-tropical-cyclone-data) for full function signatures.
