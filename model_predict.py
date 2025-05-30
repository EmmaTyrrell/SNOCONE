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
