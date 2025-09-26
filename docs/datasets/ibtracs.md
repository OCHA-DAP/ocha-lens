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
See below for the output schemas provided by this module. These schemas are designed to be interoperable with other cyclone track data sources (eg. ECMWF).

### `lens.ibtracs.get_storms()`

This function outputs a table that contains one row per unique storm (as identified by the `sid`). This data can be used to map between different storm identification systems (eg. `sid`to `atcf_id`) and obtain storm-level metadata.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `sid` | `str` | **Required** | Part of unique constraint | IBTrACS serial identifier |
| `atcf_id` | `str` | Optional | - | ATCF (Automated Tropical Cyclone Forecasting) ID |
| `storm_id` | `str` | Optional | Part of unique constraint | Concatenation of `<name>_<basin>_<season>` |
| `number` | `int16` | **Required** | - | Storm number in season |
| `season` | `int64` | **Required** | 1840-2100 range | Storm season year[^1] |
| `name` | `str` | Optional | - | Storm name, all uppercase |
| `genesis_basin` | `str` | **Required** | Must match basin mapping[^2] | Basin where storm first formed |
| `provisional` | `bool` | **Required** | - | Whether data is Provisional |

See more details of the enforced schema from [this validation](https://github.com/OCHA-DAP/ocha-lens/blob/1575856776618427e8098104fdc5d67f20c82584/src/ocha_lens/datasources/ibtracs.py#L25-L42) in the source code.

### `lens.ibtracs.get_tracks()`

This function outputs cleaned tracks for all cyclones in the raw input data. The cyclone intensity measurements (such as wind speed, pressure, etc.) are retrieved differently depending on whether the storm is provisional or not. Provisional storms pull this data from the relevant USA Agency, while the official “best track” storms use the values reported by the relevant WMO Agency.

Note that track points interpolated by the IBTrACS algorithm are dropped from this dataset. As is described in more detail [below](#comparing-wind-speeds-between-storms), great care must be taken when comparing wind speeds between storms.


| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `sid` | `str` | **Required** | Part of unique constraint | IBTrACS serial identifier |
| `point_id` | `str` | **Required** | - | Unique identifier for this observation |
| `storm_id` | `str` | **Required** | Part of unique constraint | Links to storm metadata |
| `valid_time` | `pd.Timestamp` | **Required** | Part of unique constraint | Observation time |
| `provider` | `str` | **Required** | - | Agency providing the data |
| `basin` | `str` | **Required** | Must match basin mapping[^2] | Current basin location |
| `nature` | `str` | Optional | - | Storm classification (tropical, extratropical, etc.) |
| `wind_speed` | `Int64` | Optional | -1[^3] to 300 knots range | Maximum sustained winds |
| `pressure` | `Int64` | Optional | 800-1100 hPa range | Central pressure |
| `max_wind_radius` | `Int64` | Optional | ≥ 0 | Radius of maximum winds |
| `last_closed_isobar_radius` | `Int64` | Optional | ≥ 0 | Radius of outermost closed isobar |
| `last_closed_isobar_pressure` | `Int64` | Optional | 800-1100 hPa range | Pressure of outermost closed isobar |
| `gust_speed` | `Int64` | Optional | 0-400 knots range | Maximum gust speed |
| `quadrant_radius_34` | `object` | **Required** | List of 4 elements | 34-knot wind radii by quadrant - [NE, NW, SE, SW] |
| `quadrant_radius_50` | `object` | **Required** | List of 4 elements | 50-knot wind radii by quadrant - [NE, NW, SE, SW] |
| `quadrant_radius_64` | `object` | **Required** | List of 4 elements | 64-knot wind radii by quadrant - [NE, NW, SE, SW] |
| `geometry` | `gpd.array.GeometryDtype` | **Required** | EPSG:4326, valid lat/lon | Geographic location |


See more details of the enforced schema from [this validation](https://github.com/OCHA-DAP/ocha-lens/blob/1575856776618427e8098104fdc5d67f20c82584/src/ocha_lens/datasources/ibtracs.py#L44-L95) in the source code.

## Usage Considerations

### Incorporating additional variables from IBTrACS

The raw IBTrACS data has many many more fields than are output from this API. The majority of these fields cover the same variables, but repeated across various agencies (eg. `usa_wind`, `tokyo_wind`, `bom_wind`, etc). IBTrACS outputs a sparse data structure and so the most fields are not applicable for a given storm (ie. RSMC Tokyo only reports on cyclones in the West Pacific).

To keep analysis simple and interoperable with other cyclone track data sources, this API is relatively opinionated in which fields it preserves. While not yet directly supported, it should be relatively straightforward to merge in additional variables should your analysis require. See [this notebook](https://github.com/OCHA-DAP/ocha-lens/blob/main/examples/ibtracs.ipynb) for an example.

### Handling storms that cross the antimeridian

All track points are normalized to the [-180, 180] longitude range. Therefore, analyses such as distance calculations close to the antimeridian may not return results as expected. The joining of multiple points into tracks for these storms may also need to be handled separately for points on either side of the antimeridian.

### Comparing wind speeds between storms

Different international agencies measure and report tropical cyclone wind speeds in ways that can't be easily converted between each other, even though the numerical differences might seem simple to adjust. These differences come from varying procedures and observation methods that change over time, making it difficult to do quantitative studies comparing global wind speed data across agencies. This means wind speed values from one agency (like JMA) cannot be reliably converted to match another agency's scale (like JTWC).[^4]

### Recent data missing for RSMC La Réunion

IBTrACS is missing the best track data from RSMC La Réunion since 2022. See [this thread](https://groups.google.com/g/ibtracs-qa/c/OKzA9-ig0n0/m/GKNE5BeuDAAJ) for updates.


## Additional Resources

- [IBTrACS Website](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [IBTrACS Technical Documentation](https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2025-04/IBTrACS_version4r01_Technical_Details.pdf)
- [IBTrACS Column Documentation](https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf)
- [CLIMADA package with additional IBTrACS data wrangling](https://climada-python.readthedocs.io/en/latest/user-guide/climada_hazard_TropCyclone.html)
- [IBTrACS Q&A Forum](https://groups.google.com/g/ibtracs-qa/)
- [Browse IBTrACS data](https://ncics.org/ibtracs/index.php)
- [Humanitarian Data Exchange IBTrACS dataset](https://data.humdata.org/dataset/ibtracs-global-tropical-storm-tracks)
- [WMO regional bodies per basin](https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf)

[^1]: Storms in the Southern Hemisphere that begin after July 1 are classified with the following year.
[^2]: Viable basins are `NA` (North Atlantic), `SA` (South Atlantic), `EP` (Eastern North Pacific), `WP` (Western North Pacific), `SP` (South Pacific), `SI` (South Indian), `NI` (North Indian).
[^3]: IBTrACS source data has some instances of a wind speed of -1. Justification for this is still under investigation.
[^4]: See further details from [Knapp and Kruk (2010)](https://journals.ametsoc.org/view/journals/mwre/138/4/2009mwr3123.1.xml)
