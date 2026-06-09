# API Reference

## IBTrACS Data Processing

The `ibtracs` module provides utilities for downloading, loading, and processing IBTrACS (International Best Track Archive for Climate Stewardship) tropical cyclone data.

### Data Loading

```{eval-rst}
.. autofunction:: ocha_lens.ibtracs.download_ibtracs
.. autofunction:: ocha_lens.ibtracs.load_ibtracs
```

### Track Data Extraction

```{eval-rst}
.. autofunction:: ocha_lens.ibtracs.get_tracks
```

### Storm Metadata

```{eval-rst}
.. autofunction:: ocha_lens.ibtracs.get_storms
```

### Utility Functions

```{eval-rst}
.. autofunction:: ocha_lens.ibtracs.normalize_radii
```

## ECMWF Storm Data Processing

The `ecmwf_storm` module provides utilities for downloading, loading, and processing ECMWF cyclone
forecasts.

### Data Loading

```{eval-rst}
.. autofunction:: ocha_lens.ecmwf_storm.download_forecasts
.. autofunction:: ocha_lens.ecmwf_storm.load_forecasts
```

### Track Data Extraction

```{eval-rst}
.. autofunction:: ocha_lens.ecmwf_storm.get_tracks
```

### Storm Metadata

```{eval-rst}
.. autofunction:: ocha_lens.ecmwf_storm.get_storms
.. autofunction:: ocha_lens.ecmwf_storm.get_forecasts
```

## NHC Tropical Cyclone Data Processing

The `nhc` module provides utilities for downloading, loading, and processing National Hurricane Center (NHC) and Central Pacific Hurricane Center (CPHC) tropical cyclone forecast and observation data.

### Data Loading

```{eval-rst}
.. autofunction:: ocha_lens.nhc.download_nhc
.. autofunction:: ocha_lens.nhc.load_nhc
.. autofunction:: ocha_lens.nhc.download_nhc_archive
```

### Track Data Extraction

```{eval-rst}
.. autofunction:: ocha_lens.nhc.get_tracks
```

### Storm Metadata

```{eval-rst}
.. autofunction:: ocha_lens.nhc.get_storms
```

### Wind Speed Probability

```{eval-rst}
.. autofunction:: ocha_lens.nhc.get_wsp
```

## GDACS Tropical Cyclone Data

The `gdacs` module provides a client for the [GDACS](https://www.gdacs.org/) (Global Disaster Alert and Coordination System) tropical cyclone API: event/episode traversal, advisory timelines, country-level population exposure, and matching GDACS events to NHC `atcf_id`s. No authentication is required.

### Event & Episode Traversal

```{eval-rst}
.. autofunction:: ocha_lens.gdacs.get_events
.. autofunction:: ocha_lens.gdacs.get_event_detail
.. autofunction:: ocha_lens.gdacs.get_episode_detail
.. autofunction:: ocha_lens.gdacs.latest_episode_id
.. autofunction:: ocha_lens.gdacs.get_timeline
```

### Population Exposure

```{eval-rst}
.. autofunction:: ocha_lens.gdacs.get_exposure_adm0
.. autofunction:: ocha_lens.gdacs.get_exposure_adm1
```

### Track Matching

```{eval-rst}
.. autofunction:: ocha_lens.gdacs.match_to_atcf
```

### Utility Functions

```{eval-rst}
.. autofunction:: ocha_lens.gdacs.to_iso3
```

## ADAM Tropical Cyclone Data

The `adam` module provides access to WFP [ADAM](https://gis.wfp.org/adam/) (Automatic Disaster Analysis and Mapping) tropical cyclone population-exposure data: a paginated event listing and per-event admin-level (ADM0/1/2) exposure.

### Data Loading

```{eval-rst}
.. autofunction:: ocha_lens.adam.get_events
.. autofunction:: ocha_lens.adam.get_exposure
```

### Utility Functions

```{eval-rst}
.. autofunction:: ocha_lens.adam.make_cumulative
.. autofunction:: ocha_lens.adam.name_to_iso3
```

## Storm Utilities

The `utils.storm` module provides shared geometry and matching helpers used across the cyclone datasources, including wind-buffer construction and matching NHC Wind Speed Probability (WSP) polygons to storm tracks.

### Track Interpolation

```{eval-rst}
.. autofunction:: ocha_lens.utils.storm.interpolate_track
```

### Wind Buffers

```{eval-rst}
.. autofunction:: ocha_lens.utils.storm.calculate_wind_buffers_gdf
```

### WSP–Track Matching

```{eval-rst}
.. autofunction:: ocha_lens.utils.storm.match_wsp_to_tracks
```
