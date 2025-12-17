# ECMWF Cyclone Forecasts

This module provides an interface for downloading ECMWF historical forecast data and performing basic processing operations to make it suitable for immediate analysis. ECMWF's forecasts are retrieved from the [THORPEX Interactive Grand Global Ensemble (TIGGE)](https://gdex.ucar.edu/datasets/d330003/) dataset, which provides raw data in [`cxml`](https://www.cawcr.gov.au/research/cyclone-exchange/) format. These forecasts are provided every 12h, with 6h leadtime increments.

This module does not yet provide the ability to download ECMWF's lowest-latency forecasts from the [Dissemination Data Store (DISS)](https://essential.ecmwf.int/). This functionality is forthcoming.

## Quick Start

```python
import ocha_lens as lens
from datetime import datetime

# Load ECMWF forecasts as a pandas dataframe
df = lens.ecmwf_storm.load_hindcasts(
    start_date=datetime(2019, 12, 20),
    end_date=datetime(2020, 1, 15)
)

# Extract storm metadata
df_storms = lens.ecmwf_storm.get_storms(df)

# Get track data
gdf_tracks = lens.ecmwf_storm.get_tracks(df)

```

## Output Data Structure

The primary goal of this module is to provide easy access to ECMWF data in a tabular, analysis-ready format.
See below for the output schemas provided by this module. These schemas are designed to be interoperable with other cyclone track data sources (eg. IBTrACS).

### `lens.ecmwf_storm.get_storms()`

This function outputs a table that contains one row per unique storm (as identified by the `storm_id`). This data can be used to obtain storm-level metadata. Forecasts for unnamed storms (often those in the early development of a storm, or that never materialized into a storm) are not given a `storm_id` and are not given records in this table.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `storm_id` | `str` | **Required** | Must be unique | Concatenation of `<name>_<basin>_<season>` |
| `number` | `str` | **Required** | - | Storm number identifier |
| `season` | `int` | **Required** | 2005-2050 range | Storm season year[^1] |
| `name` | `str` | Optional | - | Storm name, all uppercase |
| `provider` | `str` | Optional | - | Data provider |
| `genesis_basin` | `str` | Optional | Must match basin mapping[^2] | Basin where forecast originated |

See more details of the enforced schema from [this validation](https://github.com/OCHA-DAP/ocha-lens/blob/358489c9af541ef1831b2889b89a5810e339993d/src/ocha_lens/datasources/ecmwf_storm.py#L37-L52) in the source code.

### `lens.ecmwf_storm.get_tracks()`

This function outputs cleaned tracks for all forecasts in the raw input data. Note that there will be many unnamed forecasts (and so without a `storm_id`) present in this table that are not in the storm-level table output above.

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `storm_id` | `str` | Optional | - | Links to storm metadata |
| `point_id` | `str` | **Required** | - | Unique identifier for this track point |
| `forecast_id` | `str` | **Required** | - | Forecast ID from ECMWF |
| `number` | `str` | Optional | - | Storm number identifier |
| `issued_time` | `pd.Timestamp` | **Required** | - | When the forecast was issued |
| `valid_time` | `pd.Timestamp` | **Required** | - | Time this track point is valid for |
| `provider` | `str` | **Required** | - | Forecast provider |
| `basin` | `str` | **Required** | Must match basin mapping[^2] | Basin where forecast originated[^3] |
| `leadtime` | `Int64` | **Required** | ≥ 0 | Hours ahead of forecast issue time |
| `pressure` | `float` | Optional | 800-1100 hPa range | Central pressure |
| `wind_speed` | `float` | Optional | 0-300 knots range | Maximum sustained winds[^4] |
| `geometry` | `gpd.array.GeometryDtype` | **Required** | EPSG:4326, valid lat/lon | Geographic location |

See more details of the enforced schema from [this validation](https://github.com/OCHA-DAP/ocha-lens/blob/358489c9af541ef1831b2889b89a5810e339993d/src/ocha_lens/datasources/ecmwf_storm.py#L54-L91) in the source code.

## Usage Considerations

### Cyclone identification

It can be challenging to identify unique storms from this dataset of historical forecasts. Not all forecasts correspond to a known storm, and forecasts issued from before a storm was given a name may be challenging to group with forecasts that _can_ be identified by the storm's name. Moreover, ECMWF's assigned `forecast_id` may not necessarily be unique across all storms or systems (see below).

While the assigned `storm_id` can be used to group forecasts from known storms, users should query for forecasts based on spatio/temporal bounding boxes to be sure of retrieving all forecasts for a given weather system.

### Non-unique `forecast_id`s

The `forecast_id` field may not necessarily be unique to a given forecasted system. ECMWF followings the `{initialization_time}_{latitude}_{longitude}` convention for creating these IDs. We have observed some IDs with the `00N_00E` (eg. `'2020011512_00N_00E'`), which appear to capture marginal systems around the world. IDs with non-zero coordinate suffixes appear to more reliably be unique to a given system.

### Possibility of multiple `storm_id`s for the same system in ECMWF data

It is expected that systems with forecasts starting in different basins will have multiple `storm_id`s. For example, see `esther_sp_2020`/`esther_si_2020`, and `lisa_ep_2022`/`lisa_na_2022`. This happens because the genesis basin of a forecast is included in its ID. Use caution in your analysis when investigating a storm that is close to a basin boundary, or when your study area is close to a basin boundary. In these cases, queries by a spatial bounding box may be more appropriate.

### Handling storms that cross the antimeridian

All points in `tracks` tables are normalized to the [-180, 180] longitude range. As such, analyses such as distance calculations close to the antimeridian may not return results as expected. The joining of multiple points into tracks for these storms may also need to be handled separately for points on either side of the antimeridian.

### Duplicate forecasts in ECMWF data

Historical ECMWF tracks may have duplicate records for the same issued date. These duplicates come directly from the source `cxml` files, and may have slightly different positional or intensity information in the track. These duplicates are preserved in the database. See `forecast_id='2022123112_310S_682E'` for an example.


## Additional Resources

- [NCAR TIGGE dataset](https://gdex.ucar.edu/datasets/d330003/)
- [ECMWF cyclone charts - latest forecast](https://charts.ecmwf.int/products/cyclone)
- [ECMWF cyclone charts - activity](https://charts.ecmwf.int/?facets=%7B%22Product%20type%22%3A%5B%5D%2C%22Range%22%3A%5B%5D%2C%22Parameters%22%3A%5B%22Tropical%20cyclones%22%5D%7D)
- [ECMWF forecast track processing from CLIMADA](https://climada-petals.readthedocs.io/en/latest/tutorial/climada_hazard_TCForecast.html)

[^1]: Storms in the Southern Hemisphere that begin after July 1 are classified with the following year.
[^2]: Viable basins are `NA` (North Atlantic), `EP` (Northeast Pacific), `WP` (Northwest Pacific), `SP` (South Pacific), `SI` (South Indian), `NI` (North Indian). Note that ECMWF's source data combines SI and SP basins into "Southwest Pacific". We split these tracks into `SI` and `SP` across the 135 degree longitude line. See [here](https://github.com/OCHA-DAP/ocha-lens/blob/358489c9af541ef1831b2889b89a5810e339993d/src/ocha_lens/datasources/ecmwf_storm.py#L516-L529) for specifics.
[^3]: Note that this is slightly different from the IBTrACS definition of this variable. ECMWF only reports on basin assignment at the level of each forecast, rather than for each point.
[^4]: The location of maximum `wind_speed` does not correspond exactly to the center of the storm (which is the location reported by the `geometry` field). A storm's maximum speed is experienced in the [eyewall](https://www.noaa.gov/jetstream/tropical/tropical-cyclone-introduction/tropical-cyclone-structure), which is offset from the eye. The modelled coordinates for this point are contained in the raw `cxml` data, however have been excluded for simplicity.
