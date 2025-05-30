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
