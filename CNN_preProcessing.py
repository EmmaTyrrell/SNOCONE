import rasterio
import shap
import pandas as pd
from rasterio.mask import mask
from rasterio.windows import from_bounds
import psutil
from rasterio.transform import from_bounds 
import numpy as np
import sys
import os
import subprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.utils import register_keras_serializable
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Input, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from rasterio.transform import from_bounds 
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import Loss
import gc
import tensorflow.keras.backend as K

## function for min-max scaling
def min_max_scale(data, min_val=None, max_val=None, feature_range=(0, 1)):
    """Min-Max normalize a NumPy array to a target range."""
    data = data.astype(np.float32)
    mask = np.isnan(data)

    d_min = np.nanmin(data) if min_val is None else min_val
    d_max = np.nanmax(data) if max_val is None else max_val

    # if d_max == d_min:
    #     raise ValueError("Min and max are equal — can't scale.")
    if d_max == d_min:
        return np.full_like(data, feature_range[0], dtype=np.float32)

    a, b = feature_range
    scaled = (data - d_min) / (d_max - d_min)  # to [0, 1]
    scaled = scaled * (b - a) + a              # to [a, b]

    scaled[mask] = np.nan  # preserve NaNs
    return scaled

# reads and aligns raster
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

# split up the features and arrarys 
def target_feature_stacks(start_year, end_year, WorkspaceBase, ext, vegetation_path, landCover_path, phv_path, target_shape, shapeChecks, expected_channels=None):
        ## create empty arrays
        featureArray = []
        targetArray = []
        
        # loop through the years and feature data
        years = list(range(start_year, (end_year + 1)))
        for year in years:
            print(f"Processing year {year}")
            targetSplits = WorkspaceBase + f"{year}/SWE_processed_splits/"
            fSCAWorkspace = WorkspaceBase + f"{year}/fSCA/"
            DMFSCAWorkspace = WorkspaceBase + f"{year}/DMFSCA/"
            for sample in os.listdir(targetSplits):
                featureTuple = ()
                featureName = []
                # loop through each sample and get the corresponding features
                if sample.endswith(ext):
                    # read in data
                    with rasterio.open(targetSplits + sample) as samp_src:
                        samp_data = samp_src.read(1)
                        meta = samp_src.meta.copy()
                        samp_extent = samp_src.bounds
                        samp_transform = samp_src.transform
                        samp_crs = samp_src.crs
            
                        # apply a mask to all no data values. Reminder that nodata values is -9999
                        mask = samp_data >= 0
                        msked_target = np.where(mask, samp_data, -1)
                        target_shape = msked_target.shape
            
                        # flatted data
                        samp_flat = msked_target.flatten()
                        
        
                    # try to get the fsca variables 
                    sample_root = "_".join(sample.split("_")[:2])
                    for fSCA in os.listdir(fSCAWorkspace):
                        if fSCA.endswith(".tif") and fSCA.startswith(sample_root):
                            # featureName.append(f"fSCA")
                            featureName.append(f"fSCA")
                            fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                            fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                            featureTuple += (fsca_norm,)
                            # print(fsca_norm.shape)
                            if shapeChecks == "Y":
                                if fsca_norm.shape != target_shape:
                                    print(f"WRONG SHAPE FOR {sample}: FSCA")

                    # try to get the dmfsca variables 
                    sample_doy = sample.split("_")[1]
                    for DMFSCA in os.listdir(DMFSCAWorkspace):
                        if DMFSCA.endswith(".tif") and DMFSCA.startswith(sample_doy):
                            featureName.append(f"DMFSCA")
                            dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                            dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                            featureTuple += (dmfsca_norm,)
                            # print(dmfsca_norm.shape)
                            if shapeChecks == "Y":
                                if dmfsca_norm.shape != target_shape:
                                    print(f"WRONG SHAPE FOR {sample}: DMFSCA")


                    # get a DOY array into a feature 
                    date_string = sample.split("_")[1]
                    doy_str = date_string[-3:]
                    doy = float(doy_str)
                    DOY_array = np.full_like(msked_target, doy)
                    doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                    featureTuple += (doy_norm,)
                    featureName.append("DOY")
            
                    # get the vegetation array
                    for tree in os.listdir(vegetation_path):
                        if tree.endswith(".tif"):
                            if tree.startswith(f"{year}"):
                                # featureName.append(f"{tree[:-4]}")
                                featureName.append(f"Tree Density")
                                tree_norm = read_aligned_raster(
                                src_path=vegetation_path + tree,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                                featureTuple += (tree_norm,)
                                if shapeChecks == "Y":
                                    if tree_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: TREE")

                    # get the vegetation array
                    for land in os.listdir(landCover_path):
                        if land.endswith(".tif"):
                            if land.startswith(f"{year}"):
                                # featureName.append(f"{tree[:-4]}")
                                featureName.append(f"LandCover")
                                land_norm = read_aligned_raster(
                                src_path=landCover_path + land,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                                featureTuple += (land_norm,)
                                if shapeChecks == "Y":
                                    if land_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: Land")
   
                    
                    # # get all the features in the fodler 
                    for phv in os.listdir(phv_path):
                        if phv.endswith(".tif"):
                            featureName.append(f"{phv[:-4]}")
                            phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                            featureTuple += (phv_data,)
                            if shapeChecks == "Y":
                                if phv_data.shape != target_shape:
                                     print(f"WRONG SHAPE FOR {sample}: {phv}")
                                
                    feature_stack = np.dstack(featureTuple)
                    if expected_channels is not None and feature_stack.shape[2] != expected_channels:
                        print(f"{sample} has shape {feature_stack.shape} — missing or extra feature?")
                        print(featureName)
                        print(" ")
                    else:
                        featureArray.append(feature_stack)
                        targetArray.append(samp_flat)
        return  np.array(featureArray), np.array(targetArray), featureName

# testing out this function with test date
def target_feature_stacks_testGroups(year, target_splits_path, fSCA_path, DMFSCA_path, vegetation_path, landCover_path, phv_path, extension_filter, desired_shape, debug_output_folder, num_of_channels, shapeChecks):
        ## create empty arrays
        featureArray = []
        targetArray = []
        extent_list = []
        crs_list = []
        
        # loop through the years and feature data
        # print(f"Processing {group}")
        targetSplits = target_splits_path
        fSCAWorkspace = fSCA_path
        DMFSCAWorkspace = DMFSCA_path
        for sample in os.listdir(targetSplits):
            featureTuple = ()
            featureName = []
            # loop through each sample and get the corresponding features
            if sample.endswith(extension_filter):
                # read in data
                with rasterio.open(targetSplits + sample) as samp_src:
                    samp_data = samp_src.read(1)
                    meta = samp_src.meta.copy()
                    samp_extent = samp_src.bounds
                    samp_transform = samp_src.transform
                    samp_crs = samp_src.crs
                    # apply a no-data mask
                    mask = samp_data >= 0
                    msked_target = np.where(mask, samp_data, -1)
                    target_shape = msked_target.shape
        
                    # flatted data
                    samp_flat = msked_target.flatten()
                    
    
                # try to get the fsca variables 
                sample_root = "_".join(sample.split("_")[:2])
                for fSCA in os.listdir(fSCAWorkspace):
                    if fSCA.endswith(extension_filter) and fSCA.startswith(sample_root):
                        featureName.append(f"{fSCA[:-4]}")
                        fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                        fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                        featureTuple += (fsca_norm,)
                        # print(fsca_norm.shape)
                        if fsca_norm.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: FSCA")
                            output_debug_path = debug_output_folder + f"/{sample_root}_BAD_FSCA.tif"
                            save_array_as_raster(
                                output_path=output_debug_path,
                                array=fsca_norm,
                                extent=samp_extent,
                                crs=samp_crs,
                                nodata_val=-1
                            )
                            
                # try to get the dmfsca variables 
                sample_doy = sample.split("_")[1]
                for DMFSCA in os.listdir(DMFSCAWorkspace):
                    if DMFSCA.endswith(".tif") and DMFSCA.startswith(sample_doy):
                        featureName.append(f"DMFSCA")
                        dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                        dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                        featureTuple += (dmfsca_norm,)
                        # print(dmfsca_norm.shape)
                        if shapeChecks == "Y":
                            if dmfsca_norm.shape != target_shape:
                                print(f"WRONG SHAPE FOR {sample}: DMFSCA")
        
                # get a DOY array into a feature 
                date_string = sample.split("_")[1]
                doy_str = date_string[-3:]
                doy = float(doy_str)
                DOY_array = np.full_like(msked_target, doy)
                doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                featureTuple += (doy_norm,)
                featureName.append(doy)
        
                # get the vegetation array
                for tree in os.listdir(vegetation_path):
                    if tree.endswith(extension_filter):
                        if tree.startswith(f"{year}"):
                            featureName.append(f"{tree[:-4]}")
                            tree_norm = read_aligned_raster(
                            src_path=vegetation_path + tree,
                            extent=samp_extent,
                            target_shape=target_shape
                            )
                            tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                            featureTuple += (tree_norm,)
                            if tree_norm.shape != desired_shape:
                                print(f"WRONG SHAPE FOR {sample}: TREE")
                                output_debug_path = debug_output_folder + f"/{sample_root}_BAD_TREE.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                # get the vegetation array
                for land in os.listdir(landCover_path):
                    if land.endswith(".tif"):
                        if land.startswith(f"{year}"):
                            # featureName.append(f"{tree[:-4]}")
                            featureName.append(f"LandCover")
                            land_norm = read_aligned_raster(
                            src_path=landCover_path + land,
                            extent=samp_extent,
                            target_shape=target_shape
                            )
                            land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                            featureTuple += (land_norm,)
                            if land_norm.shape != (256, 256):
                                print(f"WRONG SHAPE FOR {sample}: Land")
                                # output_debug_path = f"./debug_output/{sample_root}_BAD_TREE.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                
                # # get all the features in the fodler 
                for phv in os.listdir(phv_path):
                    if phv.endswith(extension_filter):
                        print(phv)
                        featureName.append(f"{phv[:-4]}")
                        phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                        featureTuple += (phv_data,)
                        if phv_data.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: {phv}")
                            output_debug_path = debug_output_folder + f"/{sample_root}_BAD_{phv[:-4]}.tif"
                            save_array_as_raster(
                                output_path=output_debug_path,
                                array=fsca_norm,
                                extent=samp_extent,
                                crs=samp_crs,
                                nodata_val=-1
                            )
                feature_stack = np.dstack(featureTuple)
                if feature_stack.shape[2] != num_of_channels:
                    print(f"{sample} has shape {feature_stack.shape} — missing or extra feature?")
                    print(featureName)
                    print(" ")
                else:
                    featureArray.append(feature_stack)
                    # y_stack = np.stack([msked_target, fsca], axis=-1).astype(np.float32)
                    targetArray.append(samp_flat)
                    extent_list.append(samp_extent)
                    crs_list.append(samp_crs)
        return  np.array(featureArray), np.array(targetArray), extent_list, crs_list
  
# split up the features and arrarys 
def target_feature_stacks_basins(start_year, end_year, WorkspaceBase, ext, vegetation_path, landCover_path, phv_path, target_shape, shapeChecks, basin_name, expected_channels=None):
        ## create empty arrays
        featureArray = []
        targetArray = []
        
        # loop through the years and feature data
        years = list(range(start_year, (end_year + 1)))
        for year in years:
            print(f"Processing year {year}")
            targetSplits = WorkspaceBase + f"{year}/SWE_processed_splits/"
            fSCAWorkspace = WorkspaceBase + f"{year}/fSCA/"
            DMFSCAWorkspace = WorkspaceBase + f"{year}/DMFSCA/"
            for sample in os.listdir(targetSplits):
                featureTuple = ()
                featureName = []
                # loop through each sample and get the corresponding features
                if sample.endswith(ext):
                    if sample.startswith(basin_name):
                        # read in data
                        with rasterio.open(targetSplits + sample) as samp_src:
                            samp_data = samp_src.read(1)
                            meta = samp_src.meta.copy()
                            samp_extent = samp_src.bounds
                            samp_transform = samp_src.transform
                            samp_crs = samp_src.crs
                
                            # apply a mask to all no data values. Reminder that nodata values is -9999
                            mask = samp_data >= 0
                            msked_target = np.where(mask, samp_data, -1)
                            target_shape = msked_target.shape
                
                            # flatted data
                            samp_flat = msked_target.flatten()
                            
            
                        # try to get the fsca variables 
                        sample_root = "_".join(sample.split("_")[:2])
                        for fSCA in os.listdir(fSCAWorkspace):
                            if fSCA.endswith(".tif") and fSCA.startswith(sample_root):
                                # featureName.append(f"fSCA")
                                featureName.append(f"fSCA")
                                fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                                fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                                featureTuple += (fsca_norm,)
                                # print(fsca_norm.shape)
                                if shapeChecks == "Y":
                                    if fsca_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: FSCA")
    
                        # try to get the dmfsca variables 
                        sample_doy = sample.split("_")[1]
                        for DMFSCA in os.listdir(DMFSCAWorkspace):
                            if DMFSCA.endswith(".tif") and DMFSCA.startswith(sample_doy):
                                featureName.append(f"DMFSCA")
                                dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                                dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                                featureTuple += (dmfsca_norm,)
                                # print(dmfsca_norm.shape)
                                if shapeChecks == "Y":
                                    if dmfsca_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: DMFSCA")
    
    
                        # get a DOY array into a feature 
                        date_string = sample.split("_")[1]
                        doy_str = date_string[-3:]
                        doy = float(doy_str)
                        DOY_array = np.full_like(msked_target, doy)
                        doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                        featureTuple += (doy_norm,)
                        featureName.append("DOY")
                
                        # get the vegetation array
                        for tree in os.listdir(vegetation_path):
                            if tree.endswith(".tif"):
                                if tree.startswith(f"{year}"):
                                    # featureName.append(f"{tree[:-4]}")
                                    featureName.append(f"Tree Density")
                                    tree_norm = read_aligned_raster(
                                    src_path=vegetation_path + tree,
                                    extent=samp_extent,
                                    target_shape=target_shape
                                    )
                                    tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                                    featureTuple += (tree_norm,)
                                    if shapeChecks == "Y":
                                        if tree_norm.shape != target_shape:
                                            print(f"WRONG SHAPE FOR {sample}: TREE")
    
                        # get the vegetation array
                        for land in os.listdir(landCover_path):
                            if land.endswith(".tif"):
                                if land.startswith(f"{year}"):
                                    # featureName.append(f"{tree[:-4]}")
                                    featureName.append(f"LandCover")
                                    land_norm = read_aligned_raster(
                                    src_path=landCover_path + land,
                                    extent=samp_extent,
                                    target_shape=target_shape
                                    )
                                    land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                                    featureTuple += (land_norm,)
                                    if shapeChecks == "Y":
                                        if land_norm.shape != target_shape:
                                            print(f"WRONG SHAPE FOR {sample}: Land")
       
                        
                        # # get all the features in the fodler 
                        for phv in os.listdir(phv_path):
                            if phv.endswith(".tif"):
                                featureName.append(f"{phv[:-4]}")
                                phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                                featureTuple += (phv_data,)
                                if shapeChecks == "Y":
                                    if phv_data.shape != target_shape:
                                         print(f"WRONG SHAPE FOR {sample}: {phv}")
                                    
                        feature_stack = np.dstack(featureTuple)
                        if expected_channels is not None and feature_stack.shape[2] != expected_channels:
                            print(f"{sample} has shape {feature_stack.shape} — missing or extra feature?")
                            print(featureName)
                            print(" ")
                        else:
                            featureArray.append(feature_stack)
                            targetArray.append(samp_flat)
        return  np.array(featureArray), np.array(targetArray), featureName

# Modified function that uses the CSV order as ground truth
def target_feature_stacks_testGroups_scores(year, target_splits_path, fSCA_path, DMFSCA_path, vegetation_path, landCover_path, phv_path, extension_filter, desired_shape, debug_output_folder, num_of_channels, shapeChecks, timestamp, ModelOutputs, feature_Listcsv):
        
        # Get the CSV feature order for your specific timestamp
        
        # Load the exact feature order from CSV
        feat_df = pd.read_csv(feature_Listcsv)
        csv_feature_order = feat_df[timestamp].dropna().astype(str).tolist()
        
        print(f"Using CSV feature order: {len(csv_feature_order)} features")
        
        ## create empty arrays
        featureArray = []
        targetArray = []
        extent_list = []
        crs_list = []
        
        targetSplits = target_splits_path
        fSCAWorkspace = fSCA_path
        DMFSCAWorkspace = DMFSCA_path
        
        for sample in sorted(os.listdir(targetSplits)):  # SORTED for consistency
            if sample.endswith(extension_filter):
                
                # Dictionary to store features by name
                feature_dict = {}
                
                # read in data
                with rasterio.open(targetSplits + sample) as samp_src:
                    samp_data = samp_src.read(1)
                    meta = samp_src.meta.copy()
                    samp_extent = samp_src.bounds
                    samp_transform = samp_src.transform
                    samp_crs = samp_src.crs
                    # apply a no-data mask
                    mask = samp_data >= 0
                    msked_target = np.where(mask, samp_data, -1)
                    target_shape = msked_target.shape
                    samp_flat = msked_target.flatten()
                    
                sample_root = "_".join(sample.split("_")[:2])
                sample_doy = sample.split("_")[1]
    
                # 1. Process fSCA
                for fSCA in sorted(os.listdir(fSCAWorkspace)):
                    if fSCA.endswith(extension_filter) and fSCA.startswith(sample_root):
                        fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                        fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                        feature_dict["fSCA"] = fsca_norm
                        if fsca_norm.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: FSCA")
                        break
                            
                # 2. Process DMFSCA
                for DMFSCA in sorted(os.listdir(DMFSCAWorkspace)):
                    if DMFSCA.endswith(".tif") and DMFSCA.startswith(sample_doy):
                        dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                        dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                        feature_dict["DMFSCA"] = dmfsca_norm
                        if shapeChecks == "Y" and dmfsca_norm.shape != target_shape:
                            print(f"WRONG SHAPE FOR {sample}: DMFSCA")
                        break
        
                # 3. Process DOY
                date_string = sample.split("_")[1]
                doy_str = date_string[-3:]
                doy = float(doy_str)
                DOY_array = np.full_like(msked_target, doy)
                doy_norm = min_max_scale(DOY_array, min_val=0, max_val=366)
                feature_dict["DOY"] = doy_norm
        
                # 4. Process Tree Density
                for tree in sorted(os.listdir(vegetation_path)):
                    if tree.endswith(extension_filter) and tree.startswith(f"{year}"):
                        tree_norm = read_aligned_raster(
                            src_path=vegetation_path + tree,
                            extent=samp_extent,
                            target_shape=target_shape
                        )
                        tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                        feature_dict["Tree Density"] = tree_norm
                        if tree_norm.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: TREE")
                        break
                            
                # 5. Process LandCover
                for land in sorted(os.listdir(landCover_path)):
                    if land.endswith(".tif") and land.startswith(f"{year}"):
                        land_norm = read_aligned_raster(
                            src_path=landCover_path + land,
                            extent=samp_extent,
                            target_shape=target_shape
                        )
                        land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                        feature_dict["LandCover"] = land_norm
                        if land_norm.shape != (256, 256):
                            print(f"WRONG SHAPE FOR {sample}: Land")
                        break
                
                # 6. Process PHV features - store ALL of them by name
                for phv in sorted(os.listdir(phv_path)):
                    if phv.endswith(extension_filter):
                        feature_name = phv[:-4]  # Remove .tif extension
                        phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                        feature_dict[feature_name] = phv_data
                        if phv_data.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: {phv}")
                
                # Now build the feature stack in the EXACT CSV order
                featureTuple = ()
                featureName = []
                missing_features = []
                
                for feature_name in csv_feature_order:
                    if feature_name in feature_dict:
                        featureTuple += (feature_dict[feature_name],)
                        featureName.append(feature_name)
                    else:
                        # Create placeholder for missing feature
                        placeholder = np.full_like(msked_target, -1, dtype=np.float32)
                        featureTuple += (placeholder,)
                        featureName.append(feature_name)
                        missing_features.append(feature_name)
                        print(f"WARNING: Missing feature {feature_name} for sample {sample}")
                
                if missing_features:
                    print(f"Missing features for {sample}: {missing_features}")
                            
                feature_stack = np.dstack(featureTuple)
                
                # Verify the feature count matches
                if feature_stack.shape[2] != len(csv_feature_order):
                    print(f"{sample} has shape {feature_stack.shape} — expected {len(csv_feature_order)} channels")
                    print(f"Feature names: {featureName}")
                elif feature_stack.shape[2] != num_of_channels:
                    print(f"WARNING: {sample} has {feature_stack.shape[2]} channels, expected {num_of_channels}")
                    print("This might indicate a mismatch between CSV and expected channels")
                else:
                    featureArray.append(feature_stack)
                    targetArray.append(samp_flat)
                    extent_list.append(samp_extent)
                    crs_list.append(samp_crs)
                    
                    # Debug: Print feature order for first sample
                    if len(featureArray) == 1:
                        print(f"First sample feature order:")
                        for i, name in enumerate(featureName):
                            print(f"  {i:2d}: {name}")
                        
        print(f"Loaded {len(featureArray)} samples with features in CSV order")
        return np.array(featureArray), np.array(targetArray), extent_list, crs_list

# function
# testing out this function with test date
def target_feature_stacks_testScoreGroups(year, target_splits_path, fSCA_path, DMFSCA_path, vegetation_path, landCover_path, phv_path, 
                                          extension_filter, desired_shape, debug_output_folder, num_of_channels, shapeChecks, 
                                          trainA_root, trainB_root, test_root):
        ## create empty arrays
        featureArray = []
        targetArray = []
        extent_list = []
        crs_list = []
        testSampList = []
        # loop through the years and feature data
        # print(f"Processing {group}")
        targetSplits = target_splits_path
        fSCAWorkspace = fSCA_path
        DMFSCAWorkspace = DMFSCA_path
        for sample in os.listdir(targetSplits):
            featureTuple = ()
            featureName = []
            # loop through each sample and get the corresponding features
            if sample.endswith(extension_filter):
                if sample.startswith(test_root):
                    testSampList.append(sample)
                    # read in data
                    with rasterio.open(targetSplits + sample) as samp_src:
                        samp_data = samp_src.read(1)
                        meta = samp_src.meta.copy()
                        samp_extent = samp_src.bounds
                        samp_transform = samp_src.transform
                        samp_crs = samp_src.crs
                        # apply a no-data mask
                        mask = samp_data >= 0
                        msked_target = np.where(mask, samp_data, -1)
                        target_shape = msked_target.shape
            
                        # flatted data
                        samp_flat = msked_target.flatten()
                        
        
                    # try to get the fsca variables 
                    sample_root = "_".join(sample.split("_")[:2])
                    for fSCA in os.listdir(fSCAWorkspace):
                        if fSCA.endswith(extension_filter) and fSCA.startswith(sample_root):
                            featureName.append(f"{fSCA[:-4]}")
                            fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                            fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                            featureTuple += (fsca_norm,)
                            # print(fsca_norm.shape)
                            if fsca_norm.shape != desired_shape:
                                print(f"WRONG SHAPE FOR {sample}: FSCA")
                                output_debug_path = debug_output_folder + f"/{sample_root}_BAD_FSCA.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                                
                    # try to get the dmfsca variables 
                    sample_doy = sample.split("_")[1]
                    for DMFSCA in os.listdir(DMFSCAWorkspace):
                        if DMFSCA.endswith(".tif") and DMFSCA.startswith(sample_doy):
                            featureName.append(f"DMFSCA")
                            dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                            dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                            featureTuple += (dmfsca_norm,)
                            # print(dmfsca_norm.shape)
                            if shapeChecks == "Y":
                                if dmfsca_norm.shape != target_shape:
                                    print(f"WRONG SHAPE FOR {sample}: DMFSCA")
            
                    # get a DOY array into a feature 
                    date_string = sample.split("_")[1]
                    doy_str = date_string[-3:]
                    doy = float(doy_str)
                    DOY_array = np.full_like(msked_target, doy)
                    doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                    featureTuple += (doy_norm,)
                    featureName.append(doy)
            
                    # get the vegetation array
                    for tree in os.listdir(vegetation_path):
                        if tree.endswith(extension_filter):
                            if tree.startswith(f"{year}"):
                                featureName.append(f"{tree[:-4]}")
                                tree_norm = read_aligned_raster(
                                src_path=vegetation_path + tree,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                                featureTuple += (tree_norm,)
                                if tree_norm.shape != desired_shape:
                                    print(f"WRONG SHAPE FOR {sample}: TREE")
                                    output_debug_path = debug_output_folder + f"/{sample_root}_BAD_TREE.tif"
                                    save_array_as_raster(
                                        output_path=output_debug_path,
                                        array=fsca_norm,
                                        extent=samp_extent,
                                        crs=samp_crs,
                                        nodata_val=-1
                                    )
                    # get the vegetation array
                    for land in os.listdir(landCover_path):
                        if land.endswith(".tif"):
                            if land.startswith(f"{year}"):
                                # featureName.append(f"{tree[:-4]}")
                                featureName.append(f"LandCover")
                                land_norm = read_aligned_raster(
                                src_path=landCover_path + land,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                                featureTuple += (land_norm,)
                                if land_norm.shape != (256, 256):
                                    print(f"WRONG SHAPE FOR {sample}: Land")
                                    # output_debug_path = f"./debug_output/{sample_root}_BAD_TREE.tif"
                                    save_array_as_raster(
                                        output_path=output_debug_path,
                                        array=fsca_norm,
                                        extent=samp_extent,
                                        crs=samp_crs,
                                        nodata_val=-1
                                    )
                    
                    # # get all the features in the fodler 
                    for phv in os.listdir(phv_path):
                        if phv.endswith(extension_filter):
                            featureName.append(f"{phv[:-4]}")
                            phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                            featureTuple += (phv_data,)
                            if phv_data.shape != desired_shape:
                                print(f"WRONG SHAPE FOR {sample}: {phv}")
                                output_debug_path = debug_output_folder + f"/{sample_root}_BAD_{phv[:-4]}.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                    feature_stack = np.dstack(featureTuple)
                    if feature_stack.shape[2] != num_of_channels:
                        print(f"{sample} has shape {feature_stack.shape} — missing or extra feature?")
                        print(featureName)
                        print(" ")
                    else:
                        featureArray.append(feature_stack)
                        # y_stack = np.stack([msked_target, fsca], axis=-1).astype(np.float32)
                        targetArray.append(samp_flat)
                        extent_list.append(samp_extent)
                        crs_list.append(samp_crs)
        return  np.array(featureArray), np.array(targetArray), extent_list, crs_list, featureName, testSampList

# function
# testing out this function with test date
def target_feature_stacks_trainScoreGroups(year, target_splits_path, fSCA_path, DMFSCA_path, vegetation_path, landCover_path, phv_path, 
                                           extension_filter, desired_shape, debug_output_folder, num_of_channels, shapeChecks, 
                                           trainA_root, trainB_root, test_root):
        ## create empty arrays
        featureArray = []
        targetArray = []
        extent_list = []
        crs_list = []
        trainSampList = []
        # loop through the years and feature data
        # print(f"Processing {group}")
        targetSplits = target_splits_path
        fSCAWorkspace = fSCA_path
        DMFSCAWorkspace = DMFSCA_path
        for sample in os.listdir(targetSplits):
            featureTuple = ()
            featureName = []
            # loop through each sample and get the corresponding features
            if sample.endswith(extension_filter):
                if sample.startswith((trainA_root, trainB_root)) and not sample.startswith(test_root):
                    trainSampList.append(sample)
                    
                    # read in data
                    with rasterio.open(targetSplits + sample) as samp_src:
                        samp_data = samp_src.read(1)
                        meta = samp_src.meta.copy()
                        samp_extent = samp_src.bounds
                        samp_transform = samp_src.transform
                        samp_crs = samp_src.crs
                        # apply a no-data mask
                        mask = samp_data >= 0
                        msked_target = np.where(mask, samp_data, -1)
                        target_shape = msked_target.shape
            
                        # flatted data
                        samp_flat = msked_target.flatten()
                        
        
                    # try to get the fsca variables 
                    sample_root = "_".join(sample.split("_")[:2])
                    for fSCA in os.listdir(fSCAWorkspace):
                        if fSCA.endswith(extension_filter) and fSCA.startswith(sample_root):
                            featureName.append(f"{fSCA[:-4]}")
                            fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                            fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                            featureTuple += (fsca_norm,)
                            # print(fsca_norm.shape)
                            if fsca_norm.shape != desired_shape:
                                print(f"WRONG SHAPE FOR {sample}: FSCA")
                                output_debug_path = debug_output_folder + f"/{sample_root}_BAD_FSCA.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                                
                    # try to get the dmfsca variables 
                    sample_doy = sample.split("_")[1]
                    for DMFSCA in os.listdir(DMFSCAWorkspace):
                        if DMFSCA.endswith(".tif") and DMFSCA.startswith(sample_doy):
                            featureName.append(f"DMFSCA")
                            dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                            dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                            featureTuple += (dmfsca_norm,)
                            # print(dmfsca_norm.shape)
                            if shapeChecks == "Y":
                                if dmfsca_norm.shape != target_shape:
                                    print(f"WRONG SHAPE FOR {sample}: DMFSCA")
            
                    # get a DOY array into a feature 
                    date_string = sample.split("_")[1]
                    doy_str = date_string[-3:]
                    doy = float(doy_str)
                    DOY_array = np.full_like(msked_target, doy)
                    doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                    featureTuple += (doy_norm,)
                    featureName.append(doy)
            
                    # get the vegetation array
                    for tree in os.listdir(vegetation_path):
                        if tree.endswith(extension_filter):
                            if tree.startswith(f"{year}"):
                                featureName.append(f"{tree[:-4]}")
                                tree_norm = read_aligned_raster(
                                src_path=vegetation_path + tree,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                                featureTuple += (tree_norm,)
                                if tree_norm.shape != desired_shape:
                                    print(f"WRONG SHAPE FOR {sample}: TREE")
                                    output_debug_path = debug_output_folder + f"/{sample_root}_BAD_TREE.tif"
                                    save_array_as_raster(
                                        output_path=output_debug_path,
                                        array=fsca_norm,
                                        extent=samp_extent,
                                        crs=samp_crs,
                                        nodata_val=-1
                                    )
                    # get the vegetation array
                    for land in os.listdir(landCover_path):
                        if land.endswith(".tif"):
                            if land.startswith(f"{year}"):
                                # featureName.append(f"{tree[:-4]}")
                                featureName.append(f"LandCover")
                                land_norm = read_aligned_raster(
                                src_path=landCover_path + land,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                                featureTuple += (land_norm,)
                                if land_norm.shape != (256, 256):
                                    print(f"WRONG SHAPE FOR {sample}: Land")
                                    # output_debug_path = f"./debug_output/{sample_root}_BAD_TREE.tif"
                                    save_array_as_raster(
                                        output_path=output_debug_path,
                                        array=fsca_norm,
                                        extent=samp_extent,
                                        crs=samp_crs,
                                        nodata_val=-1
                                    )
                    
                    # # get all the features in the fodler 
                    for phv in os.listdir(phv_path):
                        if phv.endswith(extension_filter):
                            featureName.append(f"{phv[:-4]}")
                            phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                            featureTuple += (phv_data,)
                            if phv_data.shape != desired_shape:
                                print(f"WRONG SHAPE FOR {sample}: {phv}")
                                output_debug_path = debug_output_folder + f"/{sample_root}_BAD_{phv[:-4]}.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                    feature_stack = np.dstack(featureTuple)
                    if feature_stack.shape[2] != num_of_channels:
                        print(f"{sample} has shape {feature_stack.shape} — missing or extra feature?")
                        print(featureName)
                        print(" ")
                    else:
                        featureArray.append(feature_stack)
                        # y_stack = np.stack([msked_target, fsca], axis=-1).astype(np.float32)
                        targetArray.append(samp_flat)
                        extent_list.append(samp_extent)
                        crs_list.append(samp_crs)
        return  np.array(featureArray), np.array(targetArray), extent_list, crs_list, featureName, trainSampList
