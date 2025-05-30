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
