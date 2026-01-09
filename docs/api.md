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
