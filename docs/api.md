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
.. autofunction:: ocha_lens.ibtracs.get_provisional_tracks
.. autofunction:: ocha_lens.ibtracs.get_best_tracks
```

### Storm Metadata

```{eval-rst}
.. autofunction:: ocha_lens.ibtracs.get_storms
```

### Utility Functions

```{eval-rst}
.. autofunction:: ocha_lens.ibtracs.normalize_radii
```
