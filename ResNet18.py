
import numpy as np
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
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import Loss
import gc
import tensorflow.keras.backend as K


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
