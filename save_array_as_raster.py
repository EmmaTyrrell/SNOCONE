
import rasterio
import os
import numpy as np
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds 
from rasterio.transform import from_bounds

def save_array_as_raster(output_path, array, extent, crs, nodata_val=-1):
    height, width = array.shape
    transform = from_bounds(*extent, width=width, height=height)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_val
    ) as dst:
        dst.write(array, 1)
