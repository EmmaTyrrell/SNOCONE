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

def download_and_merge_snotel_data(id_list, state_list, start_date, end_date, output_dir):
    merged_csv_path = os.path.join(output_dir, f"merged_snotel_{end_date}.csv")
    
    # Skip if already downloaded
    if os.path.exists(merged_csv_path):
        print("Sensors already downloaded.")
        return pd.read_csv(merged_csv_path)
    
    print("Downloading SNOTEL data...")

    for ids, state in zip(id_list, state_list):
        if ids == 0:
            continue

        url = (
            f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/"
            f"start_of_period/{ids}:{state}:SNTL%257Cid=%2522%2522%257Cname/"
            f"{start_date},{end_date}/stationId,name,WTEQ::value?fitToScreen=false"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Filter lines
            lines = response.text.splitlines()
            filtered_lines = [line for line in lines if not line.lstrip().startswith('#')]
            filtered_csv = "\n".join(filtered_lines)
            
            df = pd.read_csv(io.StringIO(filtered_csv))

            # Find SWE column
            matching_cols = [col for col in df.columns if "Snow Water Equivalent" in col]
            if matching_cols:
                col = matching_cols[0]
                match = re.search(r'\(([^)]+)\)', col)
                new_col_name = match.group(1) if match else col
                df = df.rename(columns={col: new_col_name})

            # Save to temp CSV
            temp_csv_path = os.path.join(output_dir, f"snotel_{ids}_{state}_{end_date}.csv")
            df.to_csv(temp_csv_path, index=False)

        except Exception as ex:
            print(f"Error downloading {ids}, {state}: {ex}")
            continue

    # Merge downloaded CSVs
    csv_files = [f for f in os.listdir(output_dir) if f.startswith("snotel") and f.endswith(".csv")]
    merged_df = None

    for file in csv_files:
        file_path = os.path.join(output_dir, file)
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
            continue

        df.columns = [col.strip() for col in df.columns]
        if 'Date' not in df.columns:
            print(f"Skipping file without 'Date' column: {file}")
            continue

        sensor_name = os.path.splitext(file)[0].split("_")[1]
        data_cols = [col for col in df.columns if col != 'Date']
        if not data_cols:
            print(f"No SWE column in file: {file}")
            continue

        df = df[['Date', data_cols[0]]].rename(columns={data_cols[0]: sensor_name})

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')

    # Save and clean up
    if merged_df is not None:
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved to: {merged_csv_path}")
    else:
        print("No dataframes merged.")
        return pd.DataFrame()  # return empty df

    # Delete intermediate files
    for file in os.listdir(output_dir):
        if file.startswith("snotel") and not file.startswith("merged"):
            os.remove(os.path.join(output_dir, file))

    return merged_df

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
