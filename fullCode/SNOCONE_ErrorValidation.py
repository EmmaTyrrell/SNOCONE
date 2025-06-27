# import modules
import sys
sys.path.append("D:/ASOML/SNOCONE")
from CNN_errorVisualization import safe_read_shapefile, get_swe_custom_cmap, get_red_blue_error_cmap, improved_mosaic_blending_rasterio, read_raster_info, align_raster 
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import os 
import matplotlib.patches as mpatches
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import fiona
from matplotlib_scalebar.scalebar import ScaleBar
print("import modules")

# set parameters and file paths, this needs to be set up so it can loop through all groups
domain = "Rockies"
outputWorkspace = f"D:/ASOML/{domain}/modelOutputs/fromAlpine/"
testGroupWS = f"D:/ASOML/{domain}/test_groups/"
metaCSV = testGroupWS + "testGroupMetadata.csv"
aspect_CON = r"D:\ASOML\Rockies\features\ASO_CON_aspect_albn83_60m.tif"
elev_path = r"D:\ASOML\Rockies\features\ASO_CON_dem_albn83_60m.tif"
basemap = f"D:/ASOML/{domain}/basemap_data/"
features_binned = f"D:/ASOML/{domain}/features/binned/"
directionary_path = features_binned + "binned_raster_legends.csv"

interations = ["20250613_065322"]
# groups = ["G1", "G2", "G3", "G4", "G5", "G6"]

for modelInteration in interations:
# modelInteration = "20250529_132356"
    print(modelInteration)
    groups = ["G1", "G2"]
    errorComps = f"{outputWorkspace}{modelInteration}/errorReview/"
    alignedBinned = f"{errorComps}/aligned/"
    
    os.mkdir(errorComps)
    aligned_dir = alignedBinned
    os.makedirs(aligned_dir, exist_ok=True)
    
    ##START LOOP HERE
    for group in groups:
        # make folders
        mosaicWorkspace = f"{outputWorkspace}{modelInteration}/outTifs_{group}_yPreds_tifs/mosaic_output/"
        vettingWorkspace = f"{outputWorkspace}{modelInteration}/outTifs_{group}_yPreds_tifs/vetting/"
        os.mkdir(mosaicWorkspace)
        os.mkdir(vettingWorkspace)
        print("directories created")

        # apply function to mosaic
        if __name__ == "__main__":
            # Example tiles
            raster_files = []
            for file in os.listdir(f"{outputWorkspace}{modelInteration}/outTifs_{group}_yPreds_tifs/"):
                if file.endswith(".tif"):
                    full_path = os.path.join(f"{outputWorkspace}{modelInteration}/outTifs_{group}_yPreds_tifs/", file)
                    raster_files.append(full_path)
                    
            output_path = mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic.tif"
            
            # Create blended mosaic using cosine fade
            improved_mosaic_blending_rasterio(raster_files, output_path, blend_type='cosine')
            print(f"\n Mosaic completed for group {group} with cosine blended")
        
        # plot 1
        # grab valid file path
        meta_df = pd.read_csv(metaCSV)
        meta_df = meta_df[meta_df['GroupNum'] == f"{group}"]
        year = meta_df.iloc[0]['Year']
        test_doy = meta_df.iloc[0]['TestDOY']
        test_basin = meta_df.iloc[0]['TestBasin']
        validASO = f"D:/ASOML/{domain}/{year}/SWE_processed/{test_basin}_{test_doy}_albn83_60m_SWE.tif"
        
        ## plot 1
        # grab valid file path
        meta_df = pd.read_csv(metaCSV)
        meta_df = meta_df[meta_df['GroupNum'] == f"{group}"]
        year = meta_df.iloc[0]['Year']
        test_doy = meta_df.iloc[0]['TestDOY']
        test_basin = meta_df.iloc[0]['TestBasin']
        validASO = f"D:/ASOML/{domain}/{year}/SWE_processed/{test_basin}_{test_doy}_albn83_60m_SWE.tif"
        print(validASO)
        
        # Open both rasters
        with rasterio.open(validASO) as ref_src, rasterio.open(mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic.tif") as in_src:
            # Compare CRS, cell size, extent etc. 
            same_crs = ref_src.crs == in_src.crs
            same_transform = ref_src.transform == in_src.transform
            same_resolution = ref_src.res == in_src.res
            same_shape = (ref_src.width == in_src.width) and (ref_src.height == in_src.height)
            same_extent = ref_src.bounds == in_src.bounds
        
            if all([same_crs, same_transform, same_resolution, same_shape]):
                print("Rasters are perfectly aligned and compatible.")
                pred_path = mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic.tif"
            else:
                aligned_raster_path = align_raster(
                input_raster_path=mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic.tif",
                reference_raster_path=validASO,
                output_raster_path=mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic_align.tif"
                )
                pred_path = mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic_align.tif"
                
        # percent diff
        with rasterio.open(validASO) as ref_src, rasterio.open(pred_path) as pred_src:
            ref = ref_src.read(1).astype(np.float32)
            pred = pred_src.read(1).astype(np.float32)
        
            # Get nodata values
            ref_nodata = ref_src.nodata
            pred_nodata = pred_src.nodata
            valid_mask = (~np.isnan(ref)) & (~np.isnan(pred))
        
            if ref_nodata is not None:
                valid_mask &= (ref != ref_nodata)
            if pred_nodata is not None:
                valid_mask &= (pred != pred_nodata)
            
            # Exclude ASO (ref) = 0
            valid_mask &= (ref != 0)
            
            # NEW: Exclude cells where both are 0
            valid_mask &= ~((ref == 0) & (pred == 0))
            
            # Create percent difference array
            percent_diff = np.full(ref.shape, np.nan, dtype=np.float32)
            percent_diff[valid_mask] = ((pred[valid_mask] - ref[valid_mask]) / ref[valid_mask]) * 100
        
            # Save result (optional)
            precentErr = mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic_percentDiff.tif"
            profile = ref_src.profile
            profile.update(dtype=rasterio.float32, nodata=np.nan)
        
        with rasterio.open(precentErr, 'w', **profile) as dst:
            dst.write(percent_diff, 1)
        
        
        # plot rasters
        raster_paths =[validASO, pred_path, precentErr]
        titles = ["ASO SWE (m)", "CNN SWE (M)", "% Error"]
        raster_data = []
        
        for i, path in enumerate(raster_paths):
            with rasterio.open(path) as src:
                data = src.read(1).astype(float)
                nodata = src.nodata
                
                # Handle nodata properly
                if nodata is not None:
                    if np.isnan(nodata):
                        # For NaN nodata, the data already has NaN where invalid
                        # No additional masking needed
                        pass
                    else:
                        # For numeric nodata values
                        data[data == nodata] = np.nan
                
                # Clip the percent error raster only (3rd one)
                if i == 2:
                    data = np.clip(data, -100, 1000)
                raster_data.append(data)
        # Rename for clarity
        aso_swe = raster_data[0]
        cnn_swe = raster_data[1]
        error_raw = raster_data[2]
    
        aso_masked = np.ma.masked_where((aso_swe <= 0), aso_swe)
        cnn_masked = np.ma.masked_where((cnn_swe <= 0), cnn_swe)
        
        # Mask error where both CNN and ASO are 0 or nan
        error_mask = (
        (np.isnan(cnn_swe) | (cnn_swe == 0)) &
        (np.isnan(aso_swe) | (aso_swe == 0))
        )
        
        # Apply the mask
        error_masked = np.copy(error_raw)
        error_masked[error_mask] = np.nan
        # SWE range from CNN & ASO
        vmin_swe = np.nanmin(np.stack(raster_data[:2]))
        vmax_swe = np.nanmax(np.stack(raster_data[:2]))
        
        # Error range (symmetric)
        err_max = np.nanmax(np.abs(raster_data[2]))
        first_data = raster_data[0]
        masked_first = np.full_like(first_data, np.nan)
        masked_first[first_data == -1] = -1
        masked_first[first_data == 0] = 0
        masked_first[first_data > 0] = first_data[first_data > 0]
        
        # Get colormaps and norms
        swe_cmap, swe_norm = get_swe_custom_cmap(vmin=0, vmax=3)
        swe_cmap, swe_norm = get_swe_custom_cmap(vmin=0, vmax=3)
        error_cmap, error_norm = get_red_blue_error_cmap(vmin=-100, vcenter=0, vmax=1000)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # CNN SWE with custom blue/transparent/white colormap
        im0 = axes[0].imshow(aso_masked, cmap=swe_cmap, norm=swe_norm)
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        swe_ticks = [0, 0.1, 0.25, 0.5, 1, 1.5, 2]
        fig.colorbar(im0, ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.04, ticks=swe_ticks)
        
        # ASO SWE with viridis
        im1 = axes[1].imshow(cnn_masked, cmap=swe_cmap, norm=swe_norm)
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        swe_ticks = [0, 0.1, 0.25, 0.5, 1, 1.5, 2]
        fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.046, pad=0.04, ticks=swe_ticks)
        
        # Error with diverging RdBu
        im2 = axes[2].imshow(error_masked, cmap=error_cmap, norm=error_norm)
        axes[2].set_title(titles[2])
        axes[2].axis('off')
        axes[2].set_title(titles[2])
        axes[2].axis('off')
        bar_ticks = [-100, 0, 100, 1000]
        cbar = fig.colorbar(im2, ax=axes[2], orientation='horizontal',
                            fraction=0.046, pad=0.04, ticks=bar_ticks)
        cbar.set_label("% Error", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{modelInteration} | {test_basin}, DOY:{test_doy} - ASO vs. CNN SWE Prediction", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(vettingWorkspace + f"Fig01_{modelInteration}_{test_basin}_{test_doy}_SWE_comp.png", dpi=300, bbox_inches='tight')
        plt.show()
    
        # plot spatial extension of all models
        year = meta_df.iloc[0]['Year']
        trainA_doy = meta_df.iloc[0]['TrainDOYA']
        trainA_basin = meta_df.iloc[0]['TrainBasinA']
        trainB_doy = meta_df.iloc[0]['TrainDOYB']
        trainB_basin = meta_df.iloc[0]['TrainBasinB']
        tempStretch = meta_df.iloc[0]['TempStretch']
        trainA = f"D:/ASOML/{domain}/{year}/SWE_processed/{trainA_basin}_{trainA_doy}_albn83_60m_SWE.tif"
        trainB = f"D:/ASOML/{domain}/{year}/SWE_processed/{trainB_basin}_{trainB_doy}_albn83_60m_SWE.tif"
                
        raster_paths = [trainA, trainB, validASO]
        temporal_text = (f"Temporal Stretch (DOY):{tempStretch}\n"
                         f"Train {trainA_basin}: {trainA_doy}\n"
                         f"Train {trainB_basin}: {trainB_doy}\n"
                         f"Test {test_basin}: {test_doy}\n")
        
        cities = safe_read_shapefile(basemap + "USA_Cities_albn83.shp")
        # parks = safe_read_shapefile(basemap + "USA_NatStateParks_albn83.shp")
        states = safe_read_shapefile(basemap + "USA_States_albn83.shp")
        
        # get metadata for shapefiles
        # parks_summary = parks[['NAME', 'Shape__Are']].head(5)
        # if isinstance(parks_summary, gpd.GeoDataFrame):
        #     parks_summary = parks_summary.drop(columns='geometry')
        
        cities_summary = cities[['NAME']].head(5)
        if isinstance(cities_summary, gpd.GeoDataFrame):
            cities_summary = cities_summary.drop(columns='geometry')
        
        # Read all rasters first to get combined extent
        all_extents = []
        all_rasters = []
        
        for path in raster_paths:
            data, extent = read_raster_info(path)
            all_rasters.append((data, extent))
            all_extents.append(extent)
        
        # Compute combined extent
        all_lefts, all_rights, all_bottoms, all_tops = zip(*all_extents)
        combined_extent = [min(all_lefts), max(all_rights), min(all_bottoms), max(all_tops)]
        
        # get values in extent
        bbox_polygon = box(combined_extent[0], combined_extent[2], combined_extent[1], combined_extent[3])
        # parks_in_extent = parks[parks.geometry.intersects(bbox_polygon)]
        cities_in_extent = cities[cities.geometry.intersects(bbox_polygon)]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        raster_labels = [f'Train: {trainA_basin}', f'Train: {trainB_basin}', f'Test: {test_basin}']  
        for i, ((data, extent), label) in enumerate(zip(all_rasters, raster_labels)):
            ax.imshow(data, extent=extent, cmap='gist_gray', origin='upper', alpha=1.0) 
            x_text = extent[0] + (extent[1] - extent[0]) * 0.02
            y_text = extent[3] - (extent[3] - extent[2]) * (0.05 + 0.05 * i)
            ax.text(x_text, y_text, label, color='white', fontsize=12, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
        
        # plot basemap data
        cities.plot(ax=ax, edgecolor='black', facecolor='black', linewidth=1, label='Towns')
        # parks.plot(ax=ax, facecolor='green', alpha=0.3, label='National and State Parks')
        states.plot(ax=ax, edgecolor='red', facecolor='none', linestyle='--', linewidth=1, label='State Lines')
        
        # for data, extent in all_rasters:
        #     ax.imshow(data, extent=extent, cmap='gist_grey', origin='upper')
        
        scalebar = ScaleBar(1, units="m", location='lower right')  # 1 unit = 1 meter
        ax.add_artist(scalebar)
        
        ax.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.05),
                    arrowprops=dict(facecolor='black', width=4, headwidth=10),
                    ha='center', va='center', fontsize=12,
                    xycoords='axes fraction')
        
        ax.set_xlim(combined_extent[0], combined_extent[1])
        ax.set_ylim(combined_extent[2], combined_extent[3])
        ax.set_title(f"{modelInteration} | Spatial Extent of Test and Train Data")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        ax.text(0.2, -0.15, temporal_text,
                transform=ax.transAxes,
                fontsize=11,
                ha='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8))
        
        # Create custom legend handles
        legend_handles = [
            mpatches.Patch(color='black', label='Towns'),
            mpatches.Patch(edgecolor='red', facecolor='none', linestyle='--', label='State Lines', linewidth=1),
        ]
        
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, -0.05))
        
        xmin, xmax, ymin, ymax = combined_extent
        
        for idx, row in cities_in_extent.iterrows():
            centroid = row.geometry.centroid
            x, y = centroid.x, centroid.y
            # Only label if centroid is inside the combined raster extent bounds
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                ax.text(x, y, row['NAME'], fontsize=16, color='gray')
        plt.grid(True)
        plt.savefig(vettingWorkspace + f"Fig02_{modelInteration}_SpatialDis_TestTrain.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # elevation bar graphs
        clipped_elev_path = vettingWorkspace + f"{test_basin}_elev.tif"
        # resample aspect
        align_raster(
            input_raster_path=elev_path,
            reference_raster_path=pred_path,
            output_raster_path=clipped_elev_path,
            resampling_method=Resampling.nearest
        )
        
        # Read ASO SWE
        with rasterio.open(validASO) as aso_src:
            aso_swe = aso_src.read(1).astype(float)
            aso_nodata = aso_src.nodata
            aso_swe[(aso_swe == -1) | (aso_swe == aso_nodata) | (aso_swe < 0)] = np.nan
        
        # Read CNN SWE
        with rasterio.open(pred_path) as cnn_src:
            cnn_swe = cnn_src.read(1).astype(float)
            cnn_nodata = cnn_src.nodata
            cnn_swe[(cnn_swe == -1) | (cnn_swe == cnn_nodata) | (cnn_swe < 0)] = np.nan
        
        with rasterio.open(clipped_elev_path) as elev_src:
            elevation = elev_src.read(1).astype(float)
            elevation[elevation == elev_src.nodata] = np.nan
        
        # pick bands
        min_elev = 1000
        max_elev = 4400
        bin_width = 250
        elev_bins = np.arange(min_elev, max_elev + bin_width, bin_width)
        elev_labels = [f"{low}-{low+bin_width}" for low in elev_bins[:-1]]
        
        # Digitize elevation
        elev_bin_idx = np.digitize(elevation, elev_bins)
        
        aso_means = []
        cnn_means = []
        
        for i in range(1, len(elev_bins)):
            mask = elev_bin_idx == i
            aso_means.append(np.nanmean(aso_swe[mask]))
            cnn_means.append(np.nanmean(cnn_swe[mask]))
        
        # Determine y-axis max for consistent scaling
        ymax = max(np.nanmax(aso_means), np.nanmax(cnn_means)) * 1.1
        
        # Bar positions
        x = np.arange(len(elev_labels))
        
        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        # CNN bar plot
        axes[0].bar(x, cnn_means, color='skyblue', edgecolor='black')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(elev_labels, rotation=45, ha='right')
        axes[0].set_ylabel("Mean SWE (m)")
        axes[0].set_xlabel("Elevation Band (m)")
        axes[0].set_title("CNN SWE by Elevation")
        axes[0].set_ylim(0, ymax)
        axes[0].grid(axis='y')
        
        # ASO bar plot
        axes[1].bar(x, aso_means, color='steelblue', edgecolor='black')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(elev_labels, rotation=45, ha='right')
        axes[1].set_xlabel("Elevation Band (m)")
        axes[1].set_title("ASO SWE by Elevation")
        axes[1].set_ylim(0, ymax)
        axes[1].grid(axis='y')
        
        # Add overall title and save
        fig.suptitle(f"{modelInteration} | Mean SWE by 250 m Elevation Band – {test_basin}, DOY: {test_doy}", fontsize=14, y=1.08)
        plt.tight_layout()
        plt.savefig(vettingWorkspace + f"Fig03_{modelInteration}_{test_basin}_{test_doy}_SWE_elevation.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # get scatter plot
        # Load both rasters
        with rasterio.open(validASO) as aso_src, rasterio.open(pred_path) as cnn_src:
            aso = aso_src.read(1).astype(float)
            cnn = cnn_src.read(1).astype(float)
        
            # Handle nodata
            aso_nodata = aso_src.nodata
            cnn_nodata = cnn_src.nodata
        
            aso[(aso == -1) | (aso == aso_nodata)] = np.nan
            cnn[(cnn == -1) | (cnn == cnn_nodata)] = np.nan
        
        # Build valid mask: only where both ASO and CNN are valid
        valid_mask = (~np.isnan(aso)) & (~np.isnan(cnn))
        
        # Extract valid values
        aso_valid = aso[valid_mask]
        cnn_valid = cnn[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(aso_valid, cnn_valid))
        mse  = mean_squared_error(aso_valid, cnn_valid)
        mae  = mean_absolute_error(aso_valid, cnn_valid)
        
        print(f"RMSE: {rmse:.3f} m")
        print(f"MSE : {mse:.3f} m²")
        print(f"MAE : {mae:.3f} m")
        
        plt.figure(figsize=(6,6))
        plt.scatter(aso_valid, cnn_valid, alpha=0.3, s=5, color='blue', edgecolors='none')
        
        # Plot 1:1 reference line
        lims = [0, max(np.nanmax(aso_valid), np.nanmax(cnn_valid)) * 1.05]
        plt.plot(lims, lims, 'k--', label='1:1 Line')
        
        plt.xlabel("ASO SWE (m)")
        plt.ylabel("CNN SWE (m)")
        plt.title(f"{modelInteration} | {test_basin}_{test_doy}: SWE Comparison\nRMSE: {rmse:.2f} | MAE: {mae:.2f}")
        plt.xlim(lims)
        plt.ylim(lims)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(vettingWorkspace + f"Fig04_{modelInteration}_{test_basin}_{test_doy}_SWE_scatter.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # error stats
        csv_path = errorComps + f"{modelInteration}_error_summary_stats.csv"
        
        error_stats = {
            'GroupNum':[group],
            'basin': [test_basin],
            'doy': [test_doy],
            'year': [year],
            'RMSE': [rmse],
            'MSE': [mse],
            'MAE': [mae],
            "n_valid": len(aso_valid)
        }
        
        df = pd.DataFrame(error_stats)
        
        # Append or write new file
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
    
        #fig 6: error analysis
        raster_dir = features_binned
        raster_results = []
        percDiff = mosaicWorkspace + f"{modelInteration}_{group}_cosine_mosaic_percentDiff.tif"
        for filename in sorted(os.listdir(raster_dir)):
            if filename.endswith(".tif"):
                bin_path = os.path.join(raster_dir, filename)
                aligned_bin_path = os.path.join(aligned_dir, f"aligned_{filename}")
                
                # Align to percDiff
                align_raster(bin_path, percDiff, aligned_bin_path)
        
                # Read rasters
                with rasterio.open(percDiff) as err_src, rasterio.open(aligned_bin_path) as bin_src:
                    err_data = err_src.read(1)
                    bin_data = bin_src.read(1)
                    err_data[err_data > 1000] = 1000
        
                # Mask nodata
                if err_src.nodata is not None:
                    err_data = np.where(err_data == err_src.nodata, np.nan, err_data)
                if bin_src.nodata is not None:
                    bin_data = np.where(bin_data == bin_src.nodata, np.nan, bin_data)
                    
                total_valid_pixels = np.count_nonzero(~np.isnan(err_data))
                unique_bins = np.unique(bin_data[~np.isnan(bin_data)])
        
                mean_errors = []
                std_errors = []
                bin_ids = []
                pixel_counts = []
                relative_weights = []
                normalized_errors = []
        
                for b in unique_bins:
                    mask = (bin_data == b) & ~np.isnan(err_data)
                    bin_errors = err_data[mask]
                    n_pixels = np.count_nonzero(mask)
                
                    if n_pixels > 0:
                        mean_err = np.mean(bin_errors)
                        std_err = np.std(bin_errors)
                        rel_weight = n_pixels / total_valid_pixels
                        # norm_err = mean_err / rel_weight  
                        norm_err = mean_err * (n_pixels / total_valid_pixels) 
                
                        mean_errors.append(mean_err)
                        std_errors.append(std_err)
                        pixel_counts.append(n_pixels)
                        relative_weights.append(rel_weight)
                        normalized_errors.append(norm_err)
                        bin_ids.append(b)
        
                # Store DataFrame
                df = pd.DataFrame({
                    "bin": bin_ids,
                    "mean_error": mean_errors,
                    "std_error": std_errors,
                    "pixel_count": pixel_counts,
                    "rel_weight": relative_weights,
                    "norm_error": normalized_errors
                }).sort_values("bin")
        
                # Save CSV
                csv_out = os.path.join(vettingWorkspace, f"binned_error_stats_{os.path.splitext(filename)[0]}.csv")
                df.to_csv(csv_out, index=False)
        
                raster_results.append((filename, df))
        
        
        # === Plot all subplots ===
        n = len(raster_results)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharey=True)
        axes = axes.flatten()

        if os.path.exists(directionary_path):
            df_directionary = pd.read_csv(directionary_path, index_col=0)
        else:
            df_directionary = None
        
        for i, (fname, df) in enumerate(raster_results):
            ax = axes[i]
        
            # Get the key between second and third underscore
            parts = fname.replace(".tif", "").split("_")
            if len(parts) >= 3:
                key = parts[2]  # e.g., 'aspect'
            else:
                key = None
        
            # Use bin labels from directionary if available
            if df_directionary is not None and f"{key}_srt" in df_directionary.columns and f"{key}_dir" in df_directionary.columns:
                bin_sort = df_directionary[f"{key}_srt"].dropna().values
                bin_labels = df_directionary[f"{key}_dir"].dropna().values
        
                # Map sorted bin values to display labels
                label_dict = {float(k): str(v) for k, v in zip(bin_sort, bin_labels)}
        
                # Apply label mapping if possible
                x_labels = [label_dict.get(float(b), str(b)) for b in df["bin"]]
            else:
                x_labels = [str(b) for b in df["bin"]]
        
            ax.bar(x_labels, df["norm_error"], capsize=4, color='slateblue')
            ax.set_title(fname.replace(".tif", ""), fontsize=10)
            ax.set_xlabel("Bin")
            if i % cols == 0:
                ax.set_ylabel("Mean % Error")
            ax.set_xticklabels(x_labels, rotation=45)
            ax.grid(axis='y', linestyle='--', alpha=0.4)
                
        fig.suptitle(f"Binned Percent Error – {modelInteration} Group {group}", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(vettingWorkspace + f"Fig06_{modelInteration}_{group}_ErrorBarPlot.png")
        plt.show()
    
    
    # out of loop:
    metadata_df = pd.read_csv(metaCSV)
    metadata_df = metadata_df[["GroupNum", "Dist_TrainA(m)", "Dist_TrainB(m)", 'TempStretch']]
    
    error_df = pd.read_csv(csv_path)
    merged_df = pd.merge(error_df, metadata_df, on='GroupNum', how='inner')
    
    merged_df['AvgDist(km)'] = ((merged_df['Dist_TrainA(m)'] + merged_df['Dist_TrainB(m)'])/2)/1000
    merged_df.to_csv(errorComps + f"{modelInteration}_error_summary_stats_merged.csv")
    
    df_melted = merged_df.melt(id_vars=['GroupNum', 'AvgDist(km)'], 
                        value_vars=['RMSE', 'MSE', 'MAE'],
                        var_name='Metric', value_name='Error')
    
    # Sort the melted DataFrame by distance
    df_plot_sorted = df_melted.sort_values('AvgDist(km)')
    
    # Create line plot
    plt.figure(figsize=(10, 6))
    
    for metric in df_plot_sorted['Metric'].unique():
        subset = df_plot_sorted[df_plot_sorted['Metric'] == metric]
        plt.plot(subset['AvgDist(km)'], subset['Error'], marker='o', label=metric)
    
        for i, row in subset.iterrows():
            plt.text(row['AvgDist(km)'] + 0.5, row['Error'], row['GroupNum'], fontsize=9)
    
    plt.xlabel('Distance from ASO Center (km)')
    plt.ylabel('Error Value (m)')
    plt.title(f'{modelInteration}_Error Metrics vs Distance')
    plt.legend(title='Metric')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(errorComps + f"Fig05_{modelInteration}_ErrorStats.png", dpi=300, bbox_inches='tight')
    plt.show()
