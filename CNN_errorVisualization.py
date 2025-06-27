import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os 
import io
import re
import math
from rasterio.transform import rowcol
import requests
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import fiona
from matplotlib_scalebar.scalebar import ScaleBar
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import time

def safe_read_shapefile(path):
        with fiona.open(path, 'r') as src:
            return gpd.GeoDataFrame.from_features(src, crs=src.crs)

def get_swe_custom_cmap(vmin=0.0001, vmax=3):
        """
        Custom SWE colormap:
          -1 → transparent
           0 → transparent
          0.0001–vmax → blue gradient
          >vmax → clipped to darkest blue
        """
        n_blue = 126
        blues = colormaps["Blues"](np.linspace(0.3, 1.0, n_blue))
    
        # Full colormap: transparent for -1 and 0, then blue for >0
        color_list = np.vstack([
            [1, 1, 1, 0],  # -1 = transparent
            [1, 1, 1, 0],  #  0 = transparent
            blues         # >0.0001 = blue gradient
        ])
        cmap = ListedColormap(color_list)
    
        # Bin edges for -1, 0, and >0
        edges = np.concatenate([
            [-1.5, -0.5],                                      # bin for -1
            np.linspace(-0.5 + vmin, vmax, n_blue + 1)         # bins for >0.0001
        ])
    
        norm = BoundaryNorm(edges, ncolors=cmap.N, clip=True)
        return cmap, norm
    
def get_red_blue_error_cmap(vmin=-100, vcenter=0, vmax=1000, steps=256):
    """
    Custom diverging colormap:
      - Red for underestimates (negative),
      - White at 0,
      - Blue for overestimates (positive).
    Ensures 0 is centered using BoundaryNorm.
    """
    assert vmin < vcenter < vmax, "vcenter must lie between vmin and vmax"

    # Determine how many colors to allocate left and right of center
    total_range = vmax - vmin
    neg_frac = abs(vcenter - vmin) / total_range
    pos_frac = abs(vmax - vcenter) / total_range
    n_neg = int(steps * neg_frac)
    n_pos = int(steps * pos_frac)

    # Sample colormaps
    reds = colormaps["Reds_r"](np.linspace(0.2, 1.0, n_neg))  # from light pink to red
    blues = colormaps["Blues"](np.linspace(0.2, 1.0, n_pos))  # from light blue to dark blue

    # Combine red + white + blue
    white = np.array([[1.0, 1.0, 1.0, 1.0]])
    full_colors = np.vstack([reds, white, blues])
    cmap = ListedColormap(full_colors)

    # Build bin edges so each color bin has equal step size (important for BoundaryNorm)
    boundaries = np.concatenate([
        np.linspace(vmin, vcenter, n_neg, endpoint=False),  # red bins
        [vcenter],
        np.linspace(vcenter, vmax, n_pos + 1)               # blue bins
    ])

    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    return cmap, norm

# mosaic output 
def improved_mosaic_blending_rasterio(input_files, output_path, blend_type='cosine'):
    """
    A more robust approach using rasterio's built-in functionality with custom blending,
    properly preserving NoData values throughout the process.
    
    Parameters:
    - input_files: List of paths to raster files
    - output_path: Path for the output mosaic
    - blend_type: 'cosine' or 'linear'
    """
    # Open all input datasets
    sources = [rasterio.open(path) for path in input_files]
    
    # Get metadata for output
    dest_meta = sources[0].meta.copy()
    nodata_value = dest_meta.get('nodata')
    
    # Determine if we have a valid nodata value to work with
    has_nodata = nodata_value is not None
    
    # Merge datasets with standard rasterio merge to get proper georeferencing
    mosaic, out_transform = merge(sources, method='first')
    
    # Create final output with the correct dimensions
    height, width = mosaic.shape[1], mosaic.shape[2]
    num_bands = mosaic.shape[0]
    
    # Create arrays for weights and output
    weight_sum = np.zeros((height, width), dtype=np.float32)
    blended_output = np.zeros((num_bands, height, width), dtype=np.float32)
    
    # Create a mask to track which pixels are valid (not NoData)
    # Initialize with all False (all NoData)
    valid_mask = np.zeros((height, width), dtype=bool)
    
    # For each source dataset, read data and apply weighted blending
    for src_idx, src in enumerate(sources):
        # Calculate the window in the output mosaic where this source contributes
        src_bounds = src.bounds
        
        # Transform source bounds to pixel coordinates in the output mosaic
        dst_window = rasterio.windows.from_bounds(
            *src_bounds, transform=out_transform
        )
        
        # Round to get integer pixel coordinates
        dst_window = dst_window.round_offsets().round_lengths()
        xmin, ymin, xmax, ymax = map(int, [
            dst_window.col_off, 
            dst_window.row_off, 
            dst_window.col_off + dst_window.width, 
            dst_window.row_off + dst_window.height
        ])
        
        # Ensure bounds are within the output image
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        
        # Skip if window is empty
        if xmin >= xmax or ymin >= ymax:
            continue
            
        # Read the data
        with rasterio.open(input_files[src_idx]) as src:
            # Get the source nodata value (use the dataset's if available)
            src_nodata = src.nodata if src.nodata is not None else nodata_value
            
            # Check if we need to read a window
            if (xmax - xmin != src.width) or (ymax - ymin != src.height):
                # Calculate corresponding window in the source raster
                # This is a simplified approach - exact calculation would need transform conversion
                data = src.read(window=Window(0, 0, xmax - xmin, ymax - ymin))
            else:
                data = src.read()
        
        # Create masks for valid data (not NoData)
        if has_nodata and src_nodata is not None:
            # Create mask for valid data (not NoData) in the current tile
            # Use the first band as a reference for NoData values
            src_valid_mask = data[0] != src_nodata
            
            # For multi-band data, combine all band masks
            if num_bands > 1:
                for b in range(1, min(num_bands, data.shape[0])):
                    src_valid_mask = src_valid_mask & (data[b] != src_nodata)
        else:
            # If no NoData value defined, assume all data is valid
            src_valid_mask = np.ones((data.shape[1], data.shape[2]), dtype=bool)
        
        # Create weights for this tile
        tile_height, tile_width = ymax - ymin, xmax - xmin
        
        # Ensure data dimensions match the window
        if data.shape[1] != tile_height or data.shape[2] != tile_width:
            # Resize data and mask to match the window
            resized_data = np.zeros((data.shape[0], tile_height, tile_width), dtype=data.dtype)
            resized_mask = np.zeros((tile_height, tile_width), dtype=bool)
            
            # Copy what fits
            min_h = min(data.shape[1], tile_height)
            min_w = min(data.shape[2], tile_width)
            resized_data[:, :min_h, :min_w] = data[:, :min_h, :min_w]
            
            if src_valid_mask.shape == (data.shape[1], data.shape[2]):
                resized_mask[:min_h, :min_w] = src_valid_mask[:min_h, :min_w]
            
            data = resized_data
            src_valid_mask = resized_mask
        
        # Create weight matrix based on distance from center (only for valid pixels)
        cy, cx = tile_height // 2, tile_width // 2
        y, x = np.ogrid[:tile_height, :tile_width]
        
        # Calculate distance from center (normalized)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid division by zero
            dist = np.sqrt(((y - cy) / max(cy, 1)) ** 2 + ((x - cx) / max(cx, 1)) ** 2)
        
        # Clip distance to [0, 1]
        dist = np.clip(dist, 0, 1)
        
        # Calculate weights based on blend type
        if blend_type == 'cosine':
            # Cosine weights (higher at center, lower at edges)
            weights = np.cos(dist * np.pi / 2)
        else:
            # Linear weights
            weights = 1 - dist
        
        # Clip weights to [0, 1] to ensure valid range
        weights = np.clip(weights, 0, 1)
        
        # Apply weights only to valid data pixels
        weights = weights * src_valid_mask
        
        # Update valid data mask for the final output
        valid_mask[ymin:ymax, xmin:xmax] |= src_valid_mask
        
        # Apply weights to all bands (only where data is valid)
        for b in range(min(num_bands, data.shape[0])):
            # Create a working copy to avoid modifying original data
            band_data = data[b].copy().astype(np.float32)
            
            # Optionally zero out NoData values to avoid affecting the blend
            if has_nodata and src_nodata is not None:
                # Zero out NoData values for blending
                band_data[~src_valid_mask] = 0
            
            # Apply weight
            weighted_band = band_data * weights
            
            # Add to blended output
            blended_output[b, ymin:ymax, xmin:xmax] += weighted_band
        
        # Add weights to weight sum (only where data is valid)
        weight_sum[ymin:ymax, xmin:xmax] += weights
    
    # Normalize by weight sum
    with np.errstate(divide='ignore', invalid='ignore'):
        for b in range(num_bands):
            # Only normalize pixels that have weights > 0
            normalize_mask = weight_sum > 0
            blended_output[b, normalize_mask] /= weight_sum[normalize_mask]
    
    # Apply NoData values to the final result
    if has_nodata:
        for b in range(num_bands):
            # Apply NoData to all pixels marked as invalid
            blended_output[b, ~valid_mask] = nodata_value
    
    # Update metadata for output
    dest_meta.update({
        'height': height,
        'width': width,
        'transform': out_transform,
        'nodata': nodata_value  # Ensure nodata value is preserved
    })
    
    # Write output
    with rasterio.open(output_path, 'w', **dest_meta) as dst:
        dst.write(blended_output.astype(dest_meta['dtype']))
    
    # Close all input datasets
    for src in sources:
        src.close()
    
def read_raster_info(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        data = np.ma.masked_equal(data, src.nodata)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        return data, extent

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
