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
