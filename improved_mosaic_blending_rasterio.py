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
