#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import modules
import sys
sys.path.append("D:/ASOML/SNOCONE")
from CNN_errorVisualization import safe_read_shapefile
from CNN_SNOTELComparisons import download_and_merge_snotel_data, get_snotel_raster_values
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
print("import modules")

# set parameters and file paths, this needs to be set up so it can loop through all groups
domain = "Rockies"
outputWorkspace = f"D:/ASOML/{domain}/modelOutputs/fromAlpine/"
testGroupWS = f"D:/ASOML/{domain}/test_groups/"
snotelWS = testGroupWS + "snotel_data/"
metaCSV = testGroupWS + "testGroupMetadata.csv"
aspect_CON = r"D:\ASOML\Rockies\features\ASO_CON_aspect_albn83_60m.tif"
elev_path = r"D:\ASOML\Rockies\features\ASO_CON_dem_albn83_60m.tif"
basemap = f"D:/ASOML/{domain}/basemap_data/"
features_binned = f"D:/ASOML/{domain}/features/binned/"
directionary_path = features_binned + "binned_raster_legends.csv"
snotelMeta = testGroupWS + "snotel_metaData.csv"
snotel_shp = basemap + "WW_CDEC_SNOTEL_geon83.shp"

# groups
interations = ["20250613_065322"]
groups = ["G1", "G2"]
meta_df = pd.read_csv(metaCSV)

for interation in interations:
    for group in groups:
        # grab valid file path
        meta_df = pd.read_csv(metaCSV)
        vettingWS = f"{outputWorkspace}/{interation}/outTifs_{group}_yPreds_tifs/vetting/"
        meta_df = meta_df[meta_df['GroupNum'] == f"{group}"]
        # year = meta_df.iloc[0]['Year']
        year = "2025"
        trainA_doy = meta_df.iloc[0]['TrainDOYA']
        trainA_basin = meta_df.iloc[0]['TrainBasinA']
        trainB_doy = meta_df.iloc[0]['TrainDOYB']
        trainB_basin = meta_df.iloc[0]['TrainBasinB']
        test_doy = meta_df.iloc[0]['TestDOY']
        test_basin = meta_df.iloc[0]['TestBasin']
        validASO = f"D:/ASOML/{domain}/{year}/SWE_processed/{test_basin}_{test_doy}_albn83_60m_SWE.tif"
        CNN_out = f"{outputWorkspace}/{interation}/outTifs_{group}_yPreds_tifs/mosaic_output/{interation}_{group}_cosine_mosaic_align.tif"
        trainA = f"D:/ASOML/{domain}/{year}/SWE_processed/{trainA_basin}_{trainA_doy}_albn83_60m_SWE.tif"
        trainB = f"D:/ASOML/{domain}/{year}/SWE_processed/{trainB_basin}_{trainB_doy}_albn83_60m_SWE.tif"
        print(test_doy)
        
        # grab snotel metadata
        snotel_df = pd.read_csv(snotelMeta)
        id_list = snotel_df["site_id"].tolist()
        state_list = snotel_df["state_id"].tolist()
        
        # convert from doy to yyyy-mm-dd
        test_date_obj = datetime.strptime(str(test_doy), "%Y%j")
        sens_startObj = test_date_obj - timedelta(days=7)
        test_end_date= test_date_obj.strftime("%Y-%m-%d")
        test_start_date = sens_startObj.strftime("%Y-%m-%d")
        
        # convert from doy to yyyy-mm-dd
        trnA_date_obj = datetime.strptime(str(trainA_doy), "%Y%j")
        sensA_startObj = trnA_date_obj - timedelta(days=7)
        trnA_end_date= trnA_date_obj.strftime("%Y-%m-%d")
        trnA_start_date = sensA_startObj.strftime("%Y-%m-%d")
        
        # convert from doy to yyyy-mm-dd
        trnB_date_obj = datetime.strptime(str(trainB_doy), "%Y%j")
        sensB_startObj = trnB_date_obj - timedelta(days=7)
        trnB_end_date= trnB_date_obj.strftime("%Y-%m-%d")
        trnB_start_date = sensB_startObj.strftime("%Y-%m-%d")
        
        merged_test = download_and_merge_snotel_data(
            id_list=id_list,
            state_list=state_list,
            start_date=test_start_date,
            end_date=test_end_date,
            output_dir=snotelWS
        )
        
        merged_trainA = download_and_merge_snotel_data(
            id_list=id_list,
            state_list=state_list,
            start_date=trnA_start_date,
            end_date=trnA_end_date,
            output_dir=snotelWS
        )
        
        merged_trainB = download_and_merge_snotel_data(
            id_list=id_list,
            state_list=state_list,
            start_date=trnB_start_date,
            end_date=trnB_end_date,
            output_dir=snotelWS
        )
        
        snotelSHP = safe_read_shapefile(snotel_shp)
        gdf_valid = get_snotel_raster_values(validASO, snotelSHP, value_column="aso_val")
        gdf_CNN = get_snotel_raster_values(CNN_out, snotelSHP, value_column="cnn_val")
        gdf_trainA = get_snotel_raster_values(trainA, snotelSHP, value_column="trainA")
        gdf_trainB = get_snotel_raster_values(trainB, snotelSHP, value_column="trainB")
        
        # arranging the sensor shapefile for the sensors
        filtered_df = merged_test[merged_test['Date'] == test_end_date]
        sensor_cols = [col for col in filtered_df.columns if col !="Date"]
        long_df = filtered_df.melt(id_vars=['Date'], value_vars=sensor_cols,
                                var_name="site_id", value_name="snotel_SWE")
        
        # arranging geodatabase for the scatter plot
        gdf_valid = gdf_valid[["site_id", "aso_val"]]
        gdf_valid['ASO_SWE'] = gdf_valid['aso_val']*39.3701
        gdf_valid['Date'] = test_end_date
        
        gdf_CNN = gdf_CNN[["site_id", "cnn_val"]]
        gdf_CNN['CNN_SWE'] = gdf_CNN['cnn_val']*39.3701
        gdf_CNN['Date'] = test_end_date
        
        gdf_valid['site_id'] = gdf_valid['site_id'].astype(str)
        long_df['site_id'] = long_df['site_id'].astype(str)
        gdf_CNN['site_id'] = gdf_CNN['site_id'].astype(str)
        scatter = pd.merge(long_df, gdf_valid[['site_id', 'ASO_SWE']], on='site_id', how='left')
        scatter = scatter.dropna()
        scatter = pd.merge(scatter, gdf_CNN[['site_id', 'CNN_SWE']], on='site_id', how='left')
        
        # make a scatter plot
        CNN_Output = scatter["CNN_SWE"]
        ASO_Output = scatter["ASO_SWE"]
        SNOTEL = scatter["snotel_SWE"]
        plt.figure(figsize=(6,6))

        has_cnn = scatter["CNN_SWE"].notna()
        no_cnn = scatter["CNN_SWE"].isna()
        
        # Always plot SNOTEL vs ASO (blue dots)
        plt.scatter(SNOTEL, ASO_Output, alpha=0.7, color='blue', label='ASO SWE', marker='o')
        
        if has_cnn.any():
            # Plot CNN vs SNOTEL where CNN is valid (green dots)
            plt.scatter(SNOTEL[has_cnn], CNN_Output[has_cnn], alpha=0.7, color='green', label='CNN SWE', marker='o')
            
            # Plot ASO vs SNOTEL where CNN is missing (gray X's)
            plt.scatter(SNOTEL[no_cnn], ASO_Output[no_cnn], alpha=0.7, color='gray', label='CNN is NA', marker='x')
        else:
            # If no CNN data at all, just plot ASO vs SNOTEL with blue dots (already done above)
            # And plot X markers for all points (because all CNN is missing)
            plt.scatter(SNOTEL, ASO_Output, alpha=0.7, color='gray', label='CNN NA', marker='x')

        # Plot 1:1 line
        min_val = np.nanmin([SNOTEL.min(), CNN_Output.min(), ASO_Output.min()])
        max_val = np.nanmax([SNOTEL.max(), CNN_Output.max(), ASO_Output.max()])
        padding = 0.05 * (max_val - min_val)
        lims = [min_val - padding, max_val + padding]
        plt.plot(lims, lims, 'r--', label='1:1 Line')
    
        # Remove NaNs from both arrays before calculating limits
        # valid_mask = ~np.isnan(snotel_values) & ~np.isnan(model_predictions)
        # snotel_valid = snotel_values[valid_mask]
        # model_valid = model_predictions[valid_mask]
        snotel_values = scatter["snotel_SWE"]
        aso_values = scatter["ASO_SWE"]
        cnn_values = scatter["CNN_SWE"]
        print("snotel_values:", snotel_values.shape, snotel_values.head())
        print("aso_values:", aso_values.shape, aso_values.head())
        print("cnn_values:", cnn_values.shape, cnn_values.head())

        lims = [
            np.min([snotel_values.min(), aso_values.min(), cnn_values.min()]),
            np.max([snotel_values.max(), aso_values.max(), cnn_values.max()])
        ]
        
        # Optionally add padding
        padding = 0.05 * (lims[1] - lims[0])
        lims = [lims[0] - padding, lims[1] + padding]
                
        # Labels and styling
        plt.xlabel('SNOTEL SWE (in)')
        plt.ylabel("Model SWE (in)")
        plt.title(f'{interation}: SNOTEL SWE vs CNN & ASO SWE | {test_basin}_{test_doy} | {group}')
        plt.legend()
        plt.grid(True)
        plt.axis('square')  
        # plt.xlim(lims)
        # plt.ylim(lims)
        plt.savefig(vettingWS + f"Fig07_ASOvCNN_scatterPlot_{interation}_{test_basin}_{test_doy}_{group}.png")
        plt.show()
        
        sensors = gdf_valid['site_id'].tolist()
        sensors.append("Date")
        sensors_sub = merged_test[sensors]
        
        cols = sensors_sub.columns.tolist()
        if "Date" in cols:
            cols.insert(0, cols.pop(cols.index("Date")))  # move Date to front
            sensors_sub = sensors_sub[cols]
        
        ## test data
        # Get sensor IDs from gdf_valid and subset data
        merged_test.columns = merged_test.columns.astype(str)
        sens_test = gdf_valid['site_id'].astype(str).tolist()
        sens_test.append("Date")
        sensors_test = merged_test[sens_test].copy()
        
        # Ensure 'Date' is first column
        cols = sensors_test.columns.tolist()
        if "Date" in cols:
            cols.insert(0, cols.pop(cols.index("Date")))
            sensors_test = sensors_test[cols]
        
        # Rename columns for clarity (optional)
        sensors_test = sensors_test.rename(columns={col: f"# {col}" for col in sensors_test.columns if col != "Date"})
        sensors_test.columns = sensors_test.columns.str.strip()
        
        ## TrainA
        # Get sensor IDs from gdf_valid and subset data
        merged_trainA.columns = merged_trainA.columns.astype(str)
        sens_trainA = gdf_trainA['site_id'].astype(str).tolist()
        sens_trainA.append("Date")
        sens_trainA = merged_trainA[sens_trainA].copy()
        
        # Ensure 'Date' is first column
        cols = sens_trainA.columns.tolist()
        if "Date" in cols:
            cols.insert(0, cols.pop(cols.index("Date")))
            sens_trainA = sens_trainA[cols]
        
        # Rename columns for clarity (optional)
        sens_trainA = sens_trainA.rename(columns={col: f"# {col}" for col in sens_trainA.columns if col != "Date"})
        sens_trainA.columns = sens_trainA.columns.str.strip()
        
        ## TrainB
        # Get sensor IDs from gdf_valid and subset data
        merged_trainB.columns = merged_trainB.columns.astype(str)
        sens_trainB = gdf_trainB['site_id'].astype(str).tolist()
        sens_trainB.append("Date")
        sens_trainB = merged_trainB[sens_trainB].copy()
        
        # Ensure 'Date' is first column
        cols = sens_trainB.columns.tolist()
        if "Date" in cols:
            cols.insert(0, cols.pop(cols.index("Date")))
            sens_trainB = sens_trainB[cols]
        
        # Rename columns for clarity (optional)
        sens_trainB = sens_trainB.rename(columns={col: f"# {col}" for col in sens_trainB.columns if col != "Date"})
        sens_trainB.columns = sens_trainB.columns.str.strip()
        
        # Optional: make sure dates are datetime
        for df in [sensors_test, sens_trainA, sens_trainB]:
            df["Date"] = pd.to_datetime(df["Date"])
        
        # Setup figure with 3 rows, 1 column
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Set of tuples: (dataframe, axis, title)
        groups = [
            (sensors_test, axes[0], f"Test_{test_basin}_{test_doy}"),
            (sens_trainA, axes[1], f"Train_{trainA_basin}_{trainA_doy}"),
            (sens_trainB, axes[2], f"Train_{trainB_basin}_{trainB_doy}"),
        ]
        
        # Plot each group
        for df, ax, title in groups:
            for col in df.columns:
                if col != "Date":
                    ax.plot(df["Date"], df[col], label=col, marker='o', alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel("SWE (in)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
            ax.grid(True)
        
        # Final formatting
        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  
        plt.suptitle(f"{interation} SNOTEL Timeseries Measurements", fontsize=12, y=1.02)
        plt.savefig(vettingWS + f"Fig08_SNOTELtimeseries_{interation}_{test_basin}_{test_doy}_{group}.png")
        plt.show()


# In[3]:


year


# In[ ]:




