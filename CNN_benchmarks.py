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

