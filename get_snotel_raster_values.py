import os
import numpy as np
import geopandas as gpd
import rasterio
import pandas as pd
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
from rasterio.transform import rowcol

def get_snotel_raster_values(raster_path, shp, value_threshold=0, value_column="raster_val"):
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        raster_crs = src.crs
        raster_transform = src.transform
        raster_nodata = src.nodata
        raster_bounds = src.bounds

        # Reproject shapefile to match raster
        shp_proj = shp.to_crs(raster_crs)

        valid_rows = []
        raster_values = []

        for idx, row in shp_proj.iterrows():
            x, y = row.geometry.x, row.geometry.y

            # Skip if outside raster bounds
            if not (raster_bounds.left <= x <= raster_bounds.right and
                    raster_bounds.bottom <= y <= raster_bounds.top):
                continue

            try:
                r, c = rowcol(raster_transform, x, y)

                # Check if inside raster shape
                if r < 0 or r >= raster.shape[0] or c < 0 or c >= raster.shape[1]:
                    continue

                val = raster[r, c]

                # Validate value
                if (
                    val is not None and
                    (raster_nodata is None or val != raster_nodata) and
                    val > value_threshold
                ):
                    valid_rows.append(row)
                    raster_values.append(val)

            except (IndexError, ValueError):
                continue

    # Build GeoDataFrame with raster values
    if valid_rows:
        gdf = gpd.GeoDataFrame(valid_rows, crs=shp_proj.crs)
        gdf[value_column] = raster_values
        print(f"Found {len(gdf)} valid points with raster values from {raster_path}")
    else:
        gdf = gpd.GeoDataFrame(columns=shp_proj.columns, crs=shp_proj.crs)
        gdf[value_column] = []  # still add the column for consistency
        print(f"No valid points found in {raster_path}")

    return gdf
