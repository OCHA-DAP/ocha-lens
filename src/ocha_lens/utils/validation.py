def check_crs(gdf, expected_crs="EPSG:4326"):
    """Check if GeoDataFrame has the expected CRS."""
    if gdf.crs is None:
        return False
    return str(gdf.crs) == expected_crs or gdf.crs.to_string() == expected_crs


def check_quadrant_list(series):
    """Check if each value is a list of exactly 4 elements."""

    def validate_item(x):
        return isinstance(x, list) and len(x) == 4

    return series.apply(validate_item).all()


def check_coordinate_bounds(gdf):
    """Check if all Point geometries have valid lat/lon coordinates."""

    def validate_point(geom):
        # if geom is None or pd.isna(geom):
        #     return False  # No null geometries allowed
        if hasattr(geom, "x") and hasattr(geom, "y"):
            return (-180 <= geom.x <= 180) and (-90 <= geom.y <= 90)
        return False

    return gdf.geometry.apply(validate_point).all()


def check_unique_when_storm_id_not_null(df):
    # Filter out rows where storm_id is null
    filtered_df = df.dropna(subset=["storm_id"])

    # Check if the combination is unique in the filtered data
    duplicates = filtered_df.duplicated(
        subset=["storm_id", "valid_time", "leadtime"]
    )
    return ~duplicates.any()
