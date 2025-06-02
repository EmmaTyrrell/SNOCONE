def load_and_prepare_model(model_path):
    """Load a saved Keras model"""
    model = tf.keras.models.load_model(model_path)
    return model
