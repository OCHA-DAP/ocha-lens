# IBTrACS Cyclone Tracks

This module provides an interface for downloading IBTrACS data and performing basic cleaning and processing operations to make it suitable for immediate analysis.

[IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) (International Best Track Archive for Climate Stewardship) is a comprehensive global dataset that combines tropical cyclone data from multiple global agencies. It contains detailed information about each storm's path, strength (wind speeds and pressure), timing, and other technical measurements like wind radii at different intensities. The dataset includes both official quality-controlled "best tracks" and "provisional tracks" for recent storms, making it valuable for analyzing historical storm patterns and climate trends.

IBTrACS data can also be downloaded [directly from the Humanitarian Data Exchange](https://data.humdata.org/dataset/ibtracs-global-tropical-storm-tracks).


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

## Output Data Structure

The primary goal of this module is to provide easy access to IBTrACS data in a tabular, analysis-ready format.
The following two output schemas are enforced.

### `get_storms()`

This function outputs a table that contains one row per unique storm (as identified by the SID). This data can be used to map between different storm identification systems (eg. SID to ATCF ID) and obtain storm-level metadata.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `sid` | String | **Required** | Part of unique constraint | IBTrACS serial identifier |
| `atcf_id` | String | Optional | - | ATCF (Automated Tropical Cyclone Forecasting) ID |
| `number` | 16-bit Integer | **Required** | - | Storm number in season |
| `season` | 64-bit Integer | **Required** | 1840-2100 | Storm season year |
| `name` | String | Optional | - | Official storm name |
| `genesis_basin` | String | **Required** | - | Basin where storm first formed |
| `provisional` | Boolean | **Required** | - | Whether data is provisional/preliminary |
| `storm_id` | String | **Required** | Part of unique constraint | Standardized storm identifier |

See more details of the enforced schema from [this validation](https://github.com/OCHA-DAP/ocha-lens/blob/1575856776618427e8098104fdc5d67f20c82584/src/ocha_lens/datasources/ibtracs.py#L25-L42) in the source code.

### `get_tracks()`

This function outputs cleaned tracks for all cyclones in the raw input data. The cyclone intensity measurements (such as wind speed, pressure, etc.) are retrieved differently depending on whether the storm is provisional or not. Provisional storms pull this data from the relevant USA Agency, while the official “best track” storms use the values reported by the relevant WMO Agency.

Note that track points interpolated by the IBTrACS algorithm are dropped from this dataset. As is described in more detail [below](#comparing-wind-speeds-between-storms), great care must be taken when comparing wind speeds between storms.


| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `sid` | String | **Required** | Part of unique constraint | IBTrACS serial identifier |
| `valid_time` | Timestamp | **Required** | Part of unique constraint | Observation time |
| `geometry` | Geometry | **Required** | EPSG:4326, valid lat/lon | Geographic location |
| `wind_speed` | 64-bit Integer | Optional | -1 to 300 knots | Maximum sustained winds |
| `pressure` | 64-bit Integer | Optional | 800-1100 hPa | Central pressure |
| `max_wind_radius` | 64-bit Integer | Optional | ≥ 0 | Radius of maximum winds |
| `last_closed_isobar_radius` | 64-bit Integer | Optional | ≥ 0 | Radius of outermost closed isobar |
| `last_closed_isobar_pressure` | 64-bit Integer | Optional | 800-1100 hPa | Pressure of outermost closed isobar |
| `gust_speed` | 64-bit Integer | Optional | 0-400 knots | Maximum gust speed |
| `provider` | String | Optional | - | Original data source |
| `basin` | String | **Required** | - | Current basin location |
| `nature` | String | Optional | - | Storm classification (tropical, extratropical, etc.) |
| `quadrant_radius_34` | Object | **Required** | Custom validation | 34-knot wind radii by quadrant |
| `quadrant_radius_50` | Object | **Required** | Custom validation | 50-knot wind radii by quadrant |
| `quadrant_radius_64` | Object | Optional | Custom validation | 64-knot wind radii by quadrant |
| `point_id` | String | **Required** | - | Unique identifier for this observation |
| `storm_id` | String | **Required** | Part of unique constraint | Links to storm metadata |


See more details of the enforced schema from [this validation](https://github.com/OCHA-DAP/ocha-lens/blob/1575856776618427e8098104fdc5d67f20c82584/src/ocha_lens/datasources/ibtracs.py#L44-L95) in the source code.

## Usage Considerations

## Incorporating additional variables from IBTrACS

The raw IBTrACS data has many many more fields than are output from this API. The majority of these fields cover the same variables, but repeated across various agencies (eg. `usa_wind`, `tokyo_wind`, `bom_wind`, etc). IBTrACS outputs a sparse data structure and so the most fields are not applicable for a given storm (ie. RSCM Tokyo only reports on cyclones in the West Pacific).

To keep analysis simple and interoperable with other cyclone track data sources, this API is relatively opinionated in which fields it preserves. While not yet directly supported, it should be relatively straightforward to merge in additional variables should your analysis require. See [this notebook](https://github.com/OCHA-DAP/ocha-lens/blob/main/examples/ibtracs.ipynb) for an example.

### Handling storms that cross the antimeridian

All track points are normalized to the [-180, 180] longitude range. Therefore, analyses such as distance calculations close to the antimeridian may not return results as expected. The joining of multiple points into tracks for these storms may also need to be handled separately for points on either side of the antimeridian.

### Comparing wind speeds between storms

Different international agencies measure and report tropical cyclone wind speeds in ways that can't be easily converted between each other, even though the numerical differences might seem simple to adjust. These differences come from varying procedures and observation methods that change over time, making it difficult to do quantitative studies comparing global wind speed data across agencies. This means wind speed values from one agency (like JMA) cannot be reliably converted to match another agency's scale (like JTWC).

### Recent data missing for RSMC La Réunion

IBTrACS is missing the best track data from RSMC La Réunion since 2022. See [this thread](https://groups.google.com/g/ibtracs-qa/c/OKzA9-ig0n0/m/GKNE5BeuDAAJ) for updates.


## Additional Resources

- [IBTrACS Technical Documentation](https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2025-04/IBTrACS_version4r01_Technical_Details.pdf)
- Additional API for working with IBTrACS data: [CLIMADA](https://climada-python.readthedocs.io/en/latest/user-guide/climada_hazard_TropCyclone.html)
- [IBTrACS Q&A Forum](https://groups.google.com/g/ibtracs-qa/)
- [Browse IBTrACS data](https://ncics.org/ibtracs/index.php)
