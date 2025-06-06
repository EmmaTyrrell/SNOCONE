import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds 
import numpy as np
import os

def read_aligned_raster(src_path, extent, target_shape, nodata_val=-1):
    height, width = target_shape
    transform = from_bounds(*extent, width=width, height=height)

    with rasterio.open(src_path) as src:
        try:
            data = src.read(
                1,
                out_shape=target_shape,
                resampling=rasterio.enums.Resampling.nearest,
                window=src.window(*extent)
            )
        except Exception as e:
            print(f"Failed to read {src_path}: {e}")
            return np.full(target_shape, nodata_val, dtype=np.float32)

        # Handle nodata in source
        src_nodata = src.nodata
        if src_nodata is not None:
            data = np.where(data == src_nodata, np.nan, data)

        # Replace NaNs or invalid with -1
        data = np.where(np.isnan(data), nodata_val, data)

        return data
