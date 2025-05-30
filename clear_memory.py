def clear_memory():
    """Comprehensive memory clearing function"""
    # Clear TensorFlow/Keras session
    K.clear_session()
    
    # Clear TensorFlow's default graph
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Optional: Print memory usage for monitoring
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
