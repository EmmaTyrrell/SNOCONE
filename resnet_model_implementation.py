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
