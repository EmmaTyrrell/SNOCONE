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
from tensorflow.keras.layers import UpSampling2D, Reshape, Concatenate
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
from CNN_benchmarks import*
from CNN_memoryOptimization import*
from CNN_preProcessing import*
print("modules imported")

# ResNet Building Blocks
def conv_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block with a convolution shortcut."""
    bn_axis = 3  # Channel axis for 'channels_last'
    
    if conv_shortcut:
        shortcut = Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x

def identity_block(x, filters, kernel_size=3, name=None):
    """A residual block without a convolution shortcut."""
    bn_axis = 3
    
    shortcut = x

    x = Conv2D(filters, 1, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x

def basic_block(x, filters, stride=1, conv_shortcut=False, name=None):
    """Basic residual block for ResNet-18/34."""
    bn_axis = 3
    
    if conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x if stride == 1 else Conv2D(filters, 1, strides=stride, name=name + '_0_conv')(x)
        if stride != 1:
            shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)

    x = Conv2D(filters, 3, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = Conv2D(filters, 3, padding='same', name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x

def Baseline_CNN(input_shape, output_size=65536, final_activation='linear'):
    """
    Your original baseline CNN model.
    """
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape, padding='valid'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2), padding='valid'))

    # Add stacked convolutions
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2), padding='valid'))

    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dropout(0.1))

    # Dense layer
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    # Output layer
    model.add(Dense(output_size, activation=final_activation))

    return model

# this section is the different ResNet architectures
def ResNet18(input_shape, num_classes, final_activation):
    """ResNet-18 implementation."""
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, 7, strides=2, padding='same', name='conv1_conv')(inputs)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # Residual blocks
    x = basic_block(x, 64, name='conv2_block1', conv_shortcut=True)
    x = basic_block(x, 64, name='conv2_block2')

    x = basic_block(x, 128, stride=2, name='conv3_block1', conv_shortcut=True)
    x = basic_block(x, 128, name='conv3_block2')

    x = basic_block(x, 256, stride=2, name='conv4_block1', conv_shortcut=True)
    x = basic_block(x, 256, name='conv4_block2')

    x = basic_block(x, 512, stride=2, name='conv5_block1', conv_shortcut=True)
    x = basic_block(x, 512, name='conv5_block2')

    # Global average pooling and final layers
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.1, name='dropout1')(x)
    outputs = Dense(num_classes, activation=final_activation, name='predictions')(x)

    model = Model(inputs, outputs, name='resnet18')
    return model

def ResNet34(input_shape, num_classes, final_activation):
    """ResNet-34 implementation."""
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, 7, strides=2, padding='same', name='conv1_conv')(inputs)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # Residual blocks
    x = basic_block(x, 64, name='conv2_block1', conv_shortcut=True)
    x = basic_block(x, 64, name='conv2_block2')
    x = basic_block(x, 64, name='conv2_block3')

    x = basic_block(x, 128, stride=2, name='conv3_block1', conv_shortcut=True)
    for i in range(2, 5):
        x = basic_block(x, 128, name=f'conv3_block{i}')

    x = basic_block(x, 256, stride=2, name='conv4_block1', conv_shortcut=True)
    for i in range(2, 7):
        x = basic_block(x, 256, name=f'conv4_block{i}')

    x = basic_block(x, 512, stride=2, name='conv5_block1', conv_shortcut=True)
    x = basic_block(x, 512, name='conv5_block2')
    x = basic_block(x, 512, name='conv5_block3')

    # Global average pooling and final layers
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.1, name='dropout1')(x)
    outputs = Dense(num_classes, activation=final_activation, name='predictions')(x)

    model = Model(inputs, outputs, name='resnet34')
    return model

def ResNet50(input_shape, num_classes, final_activation):
    """ResNet-50 implementation using bottleneck blocks."""
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, 7, strides=2, padding='same', name='conv1_conv')(inputs)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    # Residual blocks with bottleneck design
    x = conv_block(x, 64, name='conv2_block1')
    x = identity_block(x, 64, name='conv2_block2')
    x = identity_block(x, 64, name='conv2_block3')

    x = conv_block(x, 128, stride=2, name='conv3_block1')
    for i in range(2, 5):
        x = identity_block(x, 128, name=f'conv3_block{i}')

    x = conv_block(x, 256, stride=2, name='conv4_block1')
    for i in range(2, 7):
        x = identity_block(x, 256, name=f'conv4_block{i}')

    x = conv_block(x, 512, stride=2, name='conv5_block1')
    x = identity_block(x, 512, name='conv5_block2')
    x = identity_block(x, 512, name='conv5_block3')

    # Global average pooling and final layers
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.1, name='dropout1')(x)
    outputs = Dense(num_classes, activation=final_activation, name='predictions')(x)

    model = Model(inputs, outputs, name='resnet50')
    return model

# Custom ResNet for your specific use case
def CustomResNet_SWE(input_shape, output_size=65536, final_activation='linear'):
    """
    Custom ResNet adapted for your SWE-fSCA prediction task.
    Maintains spatial resolution better than standard ResNet.
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution (smaller stride to preserve resolution)
    x = Conv2D(64, 7, strides=1, padding='same', name='conv1_conv')(inputs)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding='same', name='pool1_pool')(x)

    # Residual blocks with modified strides to preserve more spatial information
    x = basic_block(x, 64, name='conv2_block1', conv_shortcut=True)
    x = basic_block(x, 64, name='conv2_block2')

    x = basic_block(x, 128, stride=2, name='conv3_block1', conv_shortcut=True)
    x = basic_block(x, 128, name='conv3_block2')

    x = basic_block(x, 256, stride=2, name='conv4_block1', conv_shortcut=True)
    x = basic_block(x, 256, name='conv4_block2')

    # Additional residual blocks for better feature learning
    x = basic_block(x, 512, stride=2, name='conv5_block1', conv_shortcut=True)
    x = basic_block(x, 512, name='conv5_block2')

    # Flatten for dense layers (similar to your original approach)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.1, name='dropout1')(x)
    
    # Dense layers
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Dropout(0.1, name='dropout2')(x)
    
    x = Dense(512, activation='relu', name='fc2')(x)
    x = BatchNormalization(name='fc2_bn')(x)
    x = Dropout(0.1, name='dropout3')(x)
    
    # Output layer
    outputs = Dense(output_size, activation=final_activation, name='predictions')(x)

    model = Model(inputs, outputs, name='custom_resnet_swe')
    return model

def model_implementation(featNo, architecture, final_activation='linear'):
    """
    Create ResNet model based on the architecture specified at the top.
    """
    input_shape = (256, 256, featNo)
    
    # Select architecture based on top-level configuration
    if architecture == "Baseline":
        model = Baseline_CNN(input_shape, 65536, final_activation)
    elif architecture == "ResNet18":
        model = ResNet18(input_shape, 65536, final_activation)
    elif architecture == "ResNet34":
        model = ResNet34(input_shape, 65536, final_activation)
    elif architecture == "ResNet50":
        model = ResNet50(input_shape, 65536, final_activation)
    elif architecture == "CustomSWE":
        model = CustomResNet_SWE(input_shape, 65536, final_activation)
    elif architecture == "AdvancedBaseline":
        model = AdvancedBaseline(input_shape, 65536, final_activation)
    elif architecture == "FCN_SWE":
        model = FCN_SWE(input_shape, 65536, final_activation)
    elif architecture == "FCN_SWE_Skipped":
        model = FCN_SWE_Skipped(input_shape, 65536, final_activation)
    elif architecture == "FCN_SWE_AvgPool":
        model = FCN_SWE_AvgPool(input_shape, 65536, final_activation)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Options are: ResNet18, ResNet34, ResNet50, CustomSWE")
    return model 

def model_predict(X):
    """
    Wrapper function to get predictions from the model.
    For CNNs with spatial outputs, you might want to either:
    1. Focus on one specific pixel location or
    2. Average across all spatial dimensions
    """
    preds = model.predict(X)
    # For a model with many output pixels (65536 in your case), you might want to:
    # - Either focus on specific pixels
    # - Or aggregate across all pixels (e.g., mean)
    return preds.reshape(X.shape[0], -1)  # Reshape to (batch_size, all_pixels)
    
    print(f"Using {architecture} architecture")
    return model

def load_and_prepare_model(model_path):
    """Load a saved Keras model"""
    model = tf.keras.models.load_model(model_path)
    return model

import os
import tempfile
import shutil
import h5py

def safe_model_save(model, filepath, backup_formats=['h5', 'weights']):
    """
    Safely save model with multiple formats and verification
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to temporary file first
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_model.keras')
    
    try:
        print(f"Saving model to temporary location: {temp_path}")
        model.save(temp_path)
        
        # Verify the saved file
        print("Verifying saved model...")
        with h5py.File(temp_path, 'r') as f:
            print(f"Model keys: {list(f.keys())}")
        
        # Test loading
        test_model = tf.keras.models.load_model(temp_path)
        print("Model loads successfully")
        del test_model
        
        # Move to final location
        shutil.move(temp_path, filepath)
        print(f"Model successfully saved to: {filepath}")
        
        # Save backup formats
        base_path = filepath.rsplit('.', 1)[0]
        
        if 'h5' in backup_formats:
            h5_path = f"{base_path}.h5"
            print(f"Saving H5 backup: {h5_path}")
            model.save(h5_path, save_format='h5')
            
        if 'weights' in backup_formats:
            weights_path = f"{base_path}_weights.h5"
            print(f"Saving weights backup: {weights_path}")
            model.save_weights(weights_path)
            
            # Save architecture separately
            arch_path = f"{base_path}_architecture.json"
            with open(arch_path, 'w') as f:
                f.write(model.to_json())
        
        return True
        
    except Exception as e:
        print(f"Save failed: {e}")
        return False
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def load_model_with_weights(weights_path, featNo, architecture, final_activation):
    """
    Recreate model using your resnet_model_implementation() and load weights
    """
    print(f"Recreating {architecture} model...")
    
    # Recreate the exact same model using your function
    model = resnet_model_implementation(featNo, architecture, final_activation)
    print(model.summary())
    # Load the saved weights
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    
    # Recompile with the same loss and metrics
    custom_loss_fn = make_swe_fsca_loss(
        base_loss_fn=MeanSquaredError(),
        penalty_weight=0.3,
        swe_threshold=0.01,
        fsca_threshold=0.01,
        mask_value=-1
    )
    
    model.compile(
        optimizer='adam',
        loss=custom_loss_fn,
        metrics=[masked_rmse, masked_mae, masked_mse]
    )
    
    print("Model recreated and weights loaded successfully!")
    return model

def attention_block(x, filters):
    """Spatial attention block to focus on snow regions."""
    # Generate attention map
    attention = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)
    
    # Apply attention
    attended = tf.keras.layers.Multiply()([x, attention])
    
    # Residual connection
    return tf.keras.layers.Add()([x, attended])

def AdvancedBaseline(input_shape, output_size=65536, final_activation='linear'):
    """
    Enhanced CNN with multi-scale feature extraction for better low snow detection.
    """
    inputs = Input(shape=input_shape)
    
    # Multi-scale branches
    # Branch 1: Fine details (3x3)
    branch1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    branch1 = BatchNormalization()(branch1)
    
    # Branch 2: Medium patterns (5x5)
    branch2 = Conv2D(32, (5,5), activation='relu', padding='same')(inputs)
    branch2 = BatchNormalization()(branch2)
    
    # Branch 3: Large patterns (7x7)
    branch3 = Conv2D(32, (7,7), activation='relu', padding='same')(inputs)
    branch3 = BatchNormalization()(branch3)
    
    # Concatenate multi-scale features
    merged = tf.keras.layers.Concatenate()([branch1, branch2, branch3])
    
    # Continue with your existing architecture pattern
    x = AveragePooling2D(pool_size=(2,2), padding='valid')(merged)
    
    # Add dilated convolutions for larger receptive field
    x = Conv2D(128, (3,3), activation='relu', padding='same', dilation_rate=2)(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2,2), padding='valid')(x)
    
    # Rest of your architecture...
    x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = attention_block(x, 128)
    x = AveragePooling2D(pool_size=(2,2), padding='valid')(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = attention_block(x, 128)
    x = AveragePooling2D(pool_size=(2,2), padding='valid')(x)
    
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    outputs = Dense(output_size, activation=final_activation)(x)
    
    return Model(inputs, outputs)

def FCN_SWE(input_shape, output_size=None, final_activation='linear'):
    model = Sequential()
    
    # Encoder (downsampling path)
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))  # 128x128
    
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))  # 64x64
    
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2)))  # 32x32
    
    # Decoder (upsampling path)
    model.add(UpSampling2D((2,2)))  # 64x64
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))  # 128x128
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))  # 256x256
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    # Final prediction layer
    model.add(Conv2D(1, (1,1), activation=final_activation))  # 256x256x1
    model.add(Reshape((65536,)))
    return model

# skipped connections
def FCN_SWE_Skipped(input_shape, output_size=None, final_activation='linear'):
    inputs = Input(shape=input_shape)
    
    # Encoder with skip connections stored
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    
    # Decoder with skip connections (U-Net style)
    u3 = UpSampling2D((2,2))(c4)
    u3 = Concatenate()([u3, c3])  # Skip connection
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(u3)
    c5 = BatchNormalization()(c5)
    
    u2 = UpSampling2D((2,2))(c5)
    u2 = Concatenate()([u2, c2])  # Skip connection
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(u2)
    c6 = BatchNormalization()(c6)
    
    u1 = UpSampling2D((2,2))(c6)
    u1 = Concatenate()([u1, c1])  # Skip connection
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(u1)
    c7 = BatchNormalization()(c7)
    
    # Output
    outputs = Conv2D(1, (1,1), activation=final_activation)(c7)
    outputs = Reshape((65536,))(outputs)
    
    return Model(inputs, outputs)

# learning rate logger
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []
        
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()
        self.lrs.append(lr)
        
def FCN_SWE_AvgPool(input_shape, output_size=None, final_activation='linear'):
    model = Sequential()
    
    # Encoder (downsampling path)
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(AveragePooling2D((2,2)))  # 128x128
    
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(AveragePooling2D((2,2)))  # 64x64
    
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(AveragePooling2D((2,2)))  # 32x32
    
    # Decoder (upsampling path)
    model.add(UpSampling2D((2,2)))  # 64x64
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))  # 128x128
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))  # 256x256
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    
    # Final prediction layer
    model.add(Conv2D(1, (1,1), activation=final_activation))  # 256x256x1
    model.add(Reshape((65536,)))
    return model
