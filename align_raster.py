# import modules
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import pandas as pd
import os 
import numpy as np
from shapely.geometry import box

def align_raster(input_raster_path, reference_raster_path, output_raster_path, resampling_method=Resampling.nearest):
        """
        Align an input raster to match the spatial extent, resolution, and CRS of a reference raster.
        
        Parameters:
        -----------
        input_raster_path : str
            Path to the input raster to be aligned
        reference_raster_path : str
            Path to the reference raster to align to
        output_raster_path : str
            Path where the aligned raster will be saved
        resampling_method : rasterio.warp.Resampling, optional
            Resampling method to use (default: Resampling.nearest)
            
        Returns:
        --------
        str
            Path to the output aligned raster
        """
        
        # Open reference and input rasters
        with rasterio.open(reference_raster_path) as ref_src, rasterio.open(input_raster_path) as in_src:
            ref_profile = ref_src.profile
            ref_data = ref_src.read(1)
            
            # Prepare destination array with same shape as reference
            aligned_data = np.empty_like(ref_data, dtype=np.float32)
            
            # Perform reprojection/alignment
            reproject(
                source=rasterio.band(in_src, 1),
                destination=aligned_data,
                src_transform=in_src.transform,
                src_crs=in_src.crs,
                dst_transform=ref_src.transform,
                dst_crs=ref_src.crs,
                dst_resolution=ref_src.res,
                resampling=resampling_method
            )
            
            # Prepare output profile
            output_profile = ref_profile.copy()
            output_profile.update(dtype='float32')
            
            # Save the reprojected raster
            with rasterio.open(output_raster_path, 'w', **output_profile) as dst:
                dst.write(aligned_data, 1)
        
        return output_raster_path
