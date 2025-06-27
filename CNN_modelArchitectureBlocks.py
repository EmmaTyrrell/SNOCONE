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

def resnet_model_implementation(featNo, final_activation='linear'):
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
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Options are: ResNet18, ResNet34, ResNet50, CustomSWE")

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
