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
from CNN_preProcessing import *
from CNN_modelArchitectureBlocks import*

_mae_metric = MeanAbsoluteError()
_mse_metric = MeanSquaredError()
_rmse_metric = MeanSquaredError()

@register_keras_serializable()
def swe_fsca_consistency_loss_fn(y_true, y_pred, base_loss_fn, penalty_weight=0.5, 
                                swe_threshold=0.01, fsca_threshold=0.01, mask_value=-1):
    """
    Core loss function that penalizes SWE-fSCA inconsistencies with masking support.
    
    Args:
        y_true: True values with shape (batch, n_pixels, 2) where [:,:,0] is SWE and [:,:,1] is fSCA
        y_pred: Predicted SWE values with shape (batch, n_pixels)
        base_loss_fn: Base loss function (e.g., MeanSquaredError())
        penalty_weight: Weight for consistency penalties
        swe_threshold: Minimum SWE to consider "snow present"
        fsca_threshold: Minimum fSCA to consider "snow cover present"  
        mask_value: Value to mask out (e.g., -1 for nodata)
    """
    
    # Handle different target shapes
    y_true_shape = tf.shape(y_true)
    
    if len(y_true.shape) == 3 and y_true.shape[-1] == 2:
        # Combined targets (batch, n_pixels, 2)
        swe_true = y_true[:, :, 0]
        fsca_true = y_true[:, :, 1]
    else:
        # Single SWE targets only - can't apply consistency penalty
        swe_true = y_true
        # Return base loss only without consistency penalty
        mask = tf.not_equal(swe_true, mask_value)
        swe_true_masked = tf.boolean_mask(swe_true, mask)
        swe_pred_masked = tf.boolean_mask(y_pred, mask)
        return base_loss_fn(swe_true_masked, swe_pred_masked)
    
    # Apply mask for valid pixels only
    mask = tf.not_equal(swe_true, mask_value)
    swe_true_masked = tf.boolean_mask(swe_true, mask)
    swe_pred_masked = tf.boolean_mask(y_pred, mask)
    fsca_true_masked = tf.boolean_mask(fsca_true, mask)
    
    # Base loss (MSE, MAE, etc.) on masked data
    base_loss = base_loss_fn(swe_true_masked, swe_pred_masked)
    
    # Consistency penalties on masked data
    # Penalty 1: Predicted SWE > threshold but observed fSCA ≈ 0
    swe_present = tf.cast(swe_pred_masked > swe_threshold, tf.float32)
    fsca_absent = tf.cast(fsca_true_masked <= fsca_threshold, tf.float32)
    penalty_1_mask = swe_present * fsca_absent
    penalty_1 = tf.reduce_mean(penalty_1_mask * tf.square(swe_pred_masked))
    
    # Penalty 2: Predicted SWE ≈ 0 but observed fSCA > threshold
    swe_absent = tf.cast(swe_pred_masked <= swe_threshold, tf.float32)
    fsca_present = tf.cast(fsca_true_masked > fsca_threshold, tf.float32)
    penalty_2_mask = swe_absent * fsca_present
    penalty_2 = tf.reduce_mean(penalty_2_mask * tf.square(fsca_true_masked))
    
    # Total loss with penalties
    consistency_penalty = penalty_1 + penalty_2
    total_loss = base_loss + penalty_weight * consistency_penalty
    
    return total_loss

@register_keras_serializable(name="swe_fsca_loss")
def make_swe_fsca_loss(base_loss_fn=MeanSquaredError(), penalty_weight=0.5,
                       swe_threshold=0.01, fsca_threshold=0.01, mask_value=-1):
    def loss(y_true, y_pred):
        return swe_fsca_consistency_loss_fn(
            y_true, y_pred,
            base_loss_fn=base_loss_fn,
            penalty_weight=penalty_weight,
            swe_threshold=swe_threshold,
            fsca_threshold=fsca_threshold,
            mask_value=mask_value
        )
    return loss
                         
@register_keras_serializable()
def low_snow_sensitivity_penalty(y_true, y_pred, mask_value=-1, 
                                low_threshold=0.05, penalty_scale=5.0):
    """
    Heavily penalizes underestimating low snow values.
    """
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    
    # Identify low snow pixels
    low_snow_mask = tf.cast(y_true_masked <= low_threshold, tf.float32)
    
    # Calculate errors
    error = y_true_masked - y_pred_masked
    
    # Penalty for underestimating low snow (negative error when true > pred)
    underestimate_penalty = tf.where(
        tf.logical_and(low_snow_mask > 0, error > 0),  # Low snow AND underestimated
        penalty_scale * tf.square(error),               # Heavy penalty
        tf.square(error)                                # Normal penalty
    )
    
    return tf.reduce_mean(underestimate_penalty)
                                  
@register_keras_serializable()
def masked_loss_fn(y_true, y_pred, loss_fn, mask_value=-1):
    mask = tf.not_equal(y_true, mask_value)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return loss_fn(y_true_masked, y_pred_masked)

@register_keras_serializable(name="masked_mse")
def masked_mse(y_true, y_pred):
    return masked_loss_fn(y_true, y_pred, _mse_metric)

@register_keras_serializable(name="masked_mae")
def masked_mae(y_true, y_pred):
    return masked_loss_fn(y_true, y_pred, _mae_metric)

@register_keras_serializable(name="masked_rmse")
def masked_rmse(y_true, y_pred):
    mse = masked_loss_fn(y_true, y_pred, _mse_metric)
    return tf.sqrt(mse)

# split up the features and arrarys 
def target_feature_stacks_SHAP(start_year, end_year, WorkspaceBase, ext, vegetation_path, landCover_path, phv_path, target_shape, shapeChecks, desired_features=None, expected_channels=None):
        
        import os
        import numpy as np
        import rasterio
    
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
                            if desired_features is None or "fSCA" in desired_features:
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
                            if desired_features is None or "DMFSCA" in desired_features:
                                featureName.append(f"DMFSCA")
                                dmfsca_norm = read_aligned_raster(src_path=DMFSCAWorkspace + DMFSCA, extent=samp_extent, target_shape=target_shape)
                                dmfsca_norm = min_max_scale(dmfsca_norm, min_val=0, max_val=100)
                                featureTuple += (dmfsca_norm,)
                                # print(dmfsca_norm.shape)
                                if shapeChecks == "Y":
                                    if dmfsca_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: DMFSCA")

                    # get a DOY array into a feature 
                    if desired_features is None or "DOY" in desired_features:
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
                                if desired_features is None or "Tree Density" in desired_features:
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
                                if desired_features is None or "LandCover" in desired_features:
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
                            feat_label = phv[:-4]
                            if desired_features is None or feat_label in desired_features:
                                featureName.append(feat_label)
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
                      # featureArray.append(feature_stack)
                      # targetArray.append(samp_flat)
        return  np.array(featureArray), np.array(targetArray), featureName



import shap
import numpy as np
import pandas as pd
import os

def run_shap(weights_path, X_sample, feature_names, featNo, architecture, final_activation, custom_loss_fn, output_dir=None):
    """
    SHAP feature importance analysis
    
    Parameters:
    -----------
    weights_path : str - Path to .h5 weights file
    X_sample : numpy.ndarray - Sample data (shape: samples, height, width, features)
    feature_names : list - List of feature names
    featNo : int - Number of features
    architecture : str - Model architecture name
    final_activation : str - Final activation function
    custom_loss_fn : function - Custom loss function
    output_dir : str, optional - Directory to save results (CSV + plots)
    
    Returns: DataFrame with feature importance rankings
    """
    
    # Load model
    print("Loading model...")
    model = model_implementation(featNo, architecture, final_activation)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss=custom_loss_fn, metrics=[masked_rmse, masked_mae, masked_mse])
    
    # SHAP analysis
    print("Creating SHAP explainer...")
    background = X_sample[:20]
    explainer = shap.GradientExplainer(model, background)
    
    print("Calculating SHAP values...")
    X_explain = X_sample[:10]
    shap_values = explainer.shap_values(X_explain)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    print(f"SHAP values shape: {shap_values.shape}")
    
    # Calculate feature importance
    if len(shap_values.shape) == 4:  # (samples, height, width, features)
        feature_importance = np.mean(np.abs(shap_values), axis=(0, 1, 2))
    else:
        feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    # Create results
    results = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': feature_importance,
        'Normalized_Importance': feature_importance / np.max(feature_importance)
    }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
    
    results['Rank'] = range(1, len(results) + 1)
    
    # Print results
    print("\nFeature Importance Rankings:")
    print(results[['Rank', 'Feature', 'SHAP_Importance']].to_string(index=False))
    
    # Save files if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'feature_importance.csv')
        results.to_csv(csv_path, index=False)
        print(f"\nCSV saved: {csv_path}")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bar chart
        colors = plt.cm.viridis(results['Normalized_Importance'])
        ax1.barh(range(len(results)), results['SHAP_Importance'], color=colors)
        ax1.set_yticks(range(len(results)))
        ax1.set_yticklabels(results['Feature'])
        ax1.set_xlabel('SHAP Importance')
        ax1.set_title('SWE Feature Importance')
        ax1.invert_yaxis()
        
        # Line plot
        ax2.plot(range(1, len(results)+1), results['SHAP_Importance'], 'o-', 
                linewidth=2, markersize=8, color='steelblue')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('SHAP Importance')
        ax2.set_title('Feature Importance by Rank')
        ax2.grid(True, alpha=0.3)
        
        # Annotate top 5
        for i in range(min(5, len(results))):
            ax2.annotate(results.iloc[i]['Feature'], 
                        (i+1, results.iloc[i]['SHAP_Importance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'feature_importance_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")
        plt.show()
    
    return results

@register_keras_serializable(name="combined_swe_fsca_lowsnow_loss")
def make_combined_swe_fsca_lowsnow_loss(base_loss_fn=MeanSquaredError(),
                                        penalty_weight=0.5,
                                        low_snow_weight=1.0,
                                        swe_threshold=0.01,
                                        fsca_threshold=0.01,
                                        low_threshold=0.05,
                                        penalty_scale=5.0,
                                        mask_value=-1):
    """
    Combined loss that enforces SWE-fSCA consistency and low snow sensitivity.
    """
    def loss(y_true, y_pred):
        # SWE-fSCA consistency loss
        consistency_loss = swe_fsca_consistency_loss_fn(
            y_true, y_pred,
            base_loss_fn=base_loss_fn,
            penalty_weight=penalty_weight,
            swe_threshold=swe_threshold,
            fsca_threshold=fsca_threshold,
            mask_value=mask_value
        )

        # Extract SWE component (second channel)
        tensor_rank = tf.rank(y_true)
        original_shape = tf.shape(y_true)
        
        swe_true = tf.cond(
            tf.logical_and(
                tf.equal(tensor_rank, 2),
                tf.equal(original_shape[-1] % 2, 0)  # Even number of elements
            ),
            lambda: y_true[:, 1::2],  # Take every other element starting from index 1 (SWE)
            lambda: y_true  # Use full tensor if not the expected format
        )

        low_snow_loss = low_snow_sensitivity_penalty(
            swe_true, y_pred,
            mask_value=mask_value,
            low_threshold=low_threshold,
            penalty_scale=penalty_scale
        )

        return consistency_loss + low_snow_weight * low_snow_loss
    return loss
