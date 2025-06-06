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
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, Flatten, Input, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.losses import Loss

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
    
    print(f"Using {architecture} architecture")
    return model
