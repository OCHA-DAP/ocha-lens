# IBTrACS

The IBTrACS (International Best Track Archive for Climate Stewardship) module provides comprehensive tools for working with tropical cyclone track data.

## Quick Start

```python
import ocha_lens as lens

# Load IBTrACS data (downloads automatically if no file specified)
ds = lens.ibtracs.load_ibtracs(dataset="ACTIVE")

# Extract storm metadata
storms_df = lens.ibtracs.get_storms(ds)

# Get best track data (official, quality-controlled tracks)
best_tracks_df = lens.ibtracs.get_best_tracks(ds)

# Get provisional track data (recent storms, often USA agency data only)
provisional_tracks_df = lens.ibtracs.get_provisional_tracks(ds)
```

## Dataset Options

When loading IBTrACS data, you can choose from three dataset options:

- **"ALL"**: Complete historical record (largest file, ~500MB)
- **"ACTIVE"**: Records for active storms only (smaller, good for current season analysis)
- **"last3years"**: Records from the past three years (smaller file, good for testing and recent analysis)

## Data Structure

The package provides three main data products:

### 1. Storm Metadata (`get_storms()`)
One row per storm with basic identifying information:
- Storm ID and name
- Season and basin information
- ATCF ID for cross-referencing
- Provisional status flag

### 2. Best Tracks (`get_best_tracks()`)
Official, quality-controlled storm tracks from designated WMO agencies:
- Position (latitude/longitude) at 6-hourly intervals
- Intensity measurements (wind speed, pressure)
- Wind radii for different intensity thresholds (34kt, 50kt, 64kt)
- Storm characteristics (nature, basin)

### 3. Provisional Tracks (`get_provisional_tracks()`)
Recent storm data that hasn't been fully processed yet:
- Typically contains USA agency data only
- Same structure as best tracks but may have data gaps
- Usually available sooner than best tracks for recent storms

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
tracks = lens.ibtracs.get_best_tracks(ds)
print(f"Total track points: {len(tracks)}")

# Filter for major hurricanes (Category 3+, ~111 kt)
major_hurricanes = tracks[tracks['wind_speed'] >= 111]
print(f"Major hurricane track points: {len(major_hurricanes)}")
```

## Data Sources

IBTrACS data is maintained by NOAA's National Centers for Environmental Information (NCEI) and combines tropical cyclone data from multiple global agencies. For more information about the dataset, visit the [official IBTrACS website](https://www.ncei.noaa.gov/products/international-best-track-archive).
