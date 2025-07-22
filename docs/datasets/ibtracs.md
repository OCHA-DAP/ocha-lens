# IBTrACS

The IBTrACS (International Best Track Archive for Climate Stewardship) module provides comprehensive tools for working with tropical cyclone track data.

## Quick Start

```python
import ocha_lens as lens

# Load IBTrACS data as an `xarray` Dataset
ds = lens.ibtracs.load_ibtracs(dataset="ACTIVE")

# Extract storm metadata
df_storms = lens.ibtracs.get_storms(ds)

# Get track data
gdf_tracks = lens.ibtracs.get_tracks(ds)

```

## Dataset Options

When loading IBTrACS data, you can choose from three dataset options:

- **"ALL"**: Complete historical record (largest file, ~500MB)
- **"ACTIVE"**: Records for active storms only (smaller, good for current season analysis)
- **"last3years"**: Records from the past three years (smaller file, good for testing and recent analysis)

## Data Structure

The package provides two main data products:

### 1. Storm Metadata (`get_storms()`)
One row per storm with basic identifying information:
- Storm ID and name
- Season and genesis basin information
- ATCF ID for cross-referencing
- Provisional status flag (whether the track is a quality-controlled "best" track or if it is still provisional)

### 2. Storm Tracks (`get_tracks()`)
Returns a `geoDataFrame` of point-level data for all storm tracks:
- Position (latitude/longitude) at 6-hourly intervals
- Intensity measurements (wind speed, pressure)
- Wind radii for different intensity thresholds (34kt, 50kt, 64kt)
- Storm characteristics (nature, basin)

The storm intensity measurements (such as wind speed, pressure, etc.) are retrieved differently depending on whether
the storm is provisional or not. Provisional storms pull this data from the relevant USA Agency, while the official "best track"
storms use the values reported by the relevant WMO Agency.

## Data Processing Features

### Wind Radii Normalization
The `normalize_radii()` function converts wind radii data from separate quadrant rows into list format, making it easier to work with the 4-quadrant wind structure data.

### Automatic Downloads
If no file path is specified, the package automatically downloads the requested dataset to a temporary directory and loads it into memory.

### Data Cleaning
All functions include built-in data cleaning:
- Handles missing values appropriately
- Converts data types for optimal performance
- Rounds coordinates and timestamps to reasonable precision
- Generates unique identifiers for each data point

## Example: Basic Analysis

```python
import ocha_lens as lens

# Load recent data
ds = lens.ibtracs.load_ibtracs(dataset="last3years")

# Get storm summary
storms = lens.ibtracs.get_storms(ds)
print(f"Found {len(storms)} storms in the dataset")

# Get best track data
tracks = lens.ibtracs.get_tracks(ds)
print(f"Total track points: {len(tracks)}")

# Filter for major hurricanes (Category 3+, ~111 kt)
major_hurricanes = tracks[tracks['wind_speed'] >= 111]
print(f"Major hurricane track points: {len(major_hurricanes)}")
```

## Data Sources

IBTrACS data is maintained by NOAA's National Centers for Environmental Information (NCEI) and combines tropical cyclone data from multiple global agencies. For more information about the dataset, visit the [official IBTrACS website](https://www.ncei.noaa.gov/products/international-best-track-archive). IBTrACS data can also be downloaded [directly from the Humanitarian Data Exchange](https://data.humdata.org/dataset/ibtracs-global-tropical-storm-tracks).
