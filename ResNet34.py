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
