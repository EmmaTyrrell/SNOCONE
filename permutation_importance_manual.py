import rasterio
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance
import seaborn as sns
from rasterio.mask import mask
from rasterio.windows import from_bounds
import psutil
from rasterio.transform import from_bounds 
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import Loss

def permutation_importance_manual(model, X_test, y_test, feature_names, 
                                 metric='accuracy', n_repeats=10, random_state=42):
    """
    Manual implementation of permutation importance for Keras models.
    Handles both 2D (samples, features) and 4D (samples, height, width, channels) data.
    """
    np.random.seed(random_state)
    
    # Get baseline score
    y_pred = model.predict(X_test)
    
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_pred sample stats - min: {y_pred.min():.4f}, max: {y_pred.max():.4f}, mean: {y_pred.mean():.4f}")
    print(f"y_test sample stats - min: {y_test.min():.4f}, max: {y_test.max():.4f}, mean: {y_test.mean():.4f}")
    
    if metric == 'mse':
        # For regression, use MSE (negative for "higher is better" interpretation)
        baseline_score = -mean_squared_error(y_test.flatten(), y_pred.flatten())
    elif metric == 'mae':
        # Alternative: Mean Absolute Error
        from sklearn.metrics import mean_absolute_error
        baseline_score = -mean_absolute_error(y_test.flatten(), y_pred.flatten())
    elif metric == 'r2':
        # Alternative: R² score (already "higher is better")
        from sklearn.metrics import r2_score
        baseline_score = r2_score(y_test.flatten(), y_pred.flatten())
    elif metric == 'rmse':
        # Root Mean Squared Error
        baseline_score = -np.sqrt(mean_squared_error(y_test.flatten(), y_pred.flatten()))
    
    importances = []
    
    # Determine if we have image data (4D) or regular data (2D)
    if len(X_test.shape) == 4:  # Image data: (samples, height, width, channels)
        n_channels = X_test.shape[-1]
        print(f"Processing {n_channels} image channels...")
    else:  # Regular data: (samples, features)
        n_channels = X_test.shape[1]
        print(f"Processing {n_channels} features...")
    
    for feature_idx in range(n_channels):
        feature_importances = []
        
        for repeat in range(n_repeats):
            # Create a copy of the test data
            X_permuted = X_test.copy()
            
            if len(X_test.shape) == 4:  # Image data
                # Shuffle the entire channel across all samples
                channel_data = X_permuted[:, :, :, feature_idx].copy()
                # Flatten, shuffle, and reshape back
                flattened = channel_data.reshape(-1)
                np.random.shuffle(flattened)
                X_permuted[:, :, :, feature_idx] = flattened.reshape(channel_data.shape)
            else:  # Regular 2D data
                # Shuffle the feature column
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
            
            # Get predictions with permuted feature
            y_pred_perm = model.predict(X_permuted)
            
            # Calculate permuted score for regression
            if metric == 'mse':
                permuted_score = -mean_squared_error(y_test.flatten(), y_pred_perm.flatten())
            elif metric == 'mae':
                permuted_score = -mean_absolute_error(y_test.flatten(), y_pred_perm.flatten())
            elif metric == 'r2':
                permuted_score = r2_score(y_test.flatten(), y_pred_perm.flatten())
            elif metric == 'rmse':
                permuted_score = -np.sqrt(mean_squared_error(y_test.flatten(), y_pred_perm.flatten()))
            
            # Calculate importance as decrease in performance
            importance = baseline_score - permuted_score
            feature_importances.append(importance)
        
        print(f"Completed channel {feature_idx + 1}/{n_channels}")
        
        importances.append({
            'feature': feature_names[feature_idx],
            'importance_mean': np.mean(feature_importances),
            'importance_std': np.std(feature_importances)
        })
    
    return pd.DataFrame(importances), baseline_score
