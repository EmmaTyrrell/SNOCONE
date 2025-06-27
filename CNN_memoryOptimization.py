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

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Get batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Fetch batch data
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # Optionally, do preprocessing or masking here
        # (e.g., ensure y_batch is ready for custom loss with -1 masking)
        y_batch[y_batch == -1] = 0

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def clear_memory():
    """Comprehensive memory clearing function"""
    # Clear TensorFlow/Keras session
    K.clear_session()
    
    # Clear TensorFlow's default graph
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Optional: Print memory usage for monitoring
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")

def memory_efficient_prediction(model, X_data, batch_size=5):
    """Make predictions in smaller batches to reduce memory usage"""
    predictions = []
    n_samples = len(X_data)
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_X = X_data[i:batch_end]
        
        # Make prediction for this batch
        batch_pred = model.predict(batch_X, batch_size=len(batch_X), verbose=0)
        predictions.append(batch_pred)
        
        # Clear intermediate variables
        del batch_X, batch_pred
        
    # Concatenate all predictions
    if len(predictions) > 0:
        all_predictions = np.concatenate(predictions, axis=0)
    else:
        all_predictions = np.array([])
    
    # Clean up
    del predictions
    gc.collect()
    
    return all_predictions
