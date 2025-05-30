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
