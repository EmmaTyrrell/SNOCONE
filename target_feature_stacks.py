def target_feature_stacks(start_year, end_year, WorkspaceBase, ext, vegetation_path, landCover_path, phv_path, target_shape):
        ## create empty arrays
        featureArray = []
        targetArray = []
        
        # loop through the years and feature data
        years = list(range(start_year, (end_year + 1)))
        for year in years:
            print(f"Processing year {year}")
            targetSplits = WorkspaceBase + f"{year}/SWE_processed_splits/"
            fSCAWorkspace = WorkspaceBase + f"{year}/fSCA/"
            for sample in os.listdir(targetSplits):
                featureTuple = ()
                featureName = []
                # loop through each sample and get the corresponding features
                if sample.endswith(ext):
                    # read in data
                    with rasterio.open(targetSplits + sample) as samp_src:
                        samp_data = samp_src.read(1)
                        meta = samp_src.meta.copy()
                        samp_extent = samp_src.bounds
                        samp_transform = samp_src.transform
                        samp_crs = samp_src.crs
            
                        # apply a mask to all no data values. Reminder that nodata values is -9999
                        mask = samp_data >= 0
                        msked_target = np.where(mask, samp_data, -1)
                        target_shape = msked_target.shape
            
                        # flatted data
                        samp_flat = msked_target.flatten()
                        
        
                    # try to get the fsca variables 
                    sample_root = "_".join(sample.split("_")[:2])
                    for fSCA in os.listdir(fSCAWorkspace):
                        if fSCA.endswith(".tif") and fSCA.startswith(sample_root):
                            # featureName.append(f"fSCA")
                            featureName.append(f"fSCA")
                            fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                            fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                            featureTuple += (fsca_norm,)
                            # print(fsca_norm.shape)
                            if shapeChecks == "Y":
                                if fsca_norm.shape != target_shape:
                                    print(f"WRONG SHAPE FOR {sample}: FSCA")

                    # get a DOY array into a feature 
                    date_string = sample.split("_")[1]
                    doy_str = date_string[-3:]
                    doy = float(doy_str)
                    DOY_array = np.full_like(msked_target, doy)
                    doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                    featureTuple += (doy_norm,)
                    featureName.append("DOY")
            
                    # get the vegetation array
                    for tree in os.listdir(vegetation_path):
                        if tree.endswith(".tif"):
                            if tree.startswith(f"{year}"):
                                # featureName.append(f"{tree[:-4]}")
                                featureName.append(f"Tree Density")
                                tree_norm = read_aligned_raster(
                                src_path=vegetation_path + tree,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                                featureTuple += (tree_norm,)
                                if shapeChecks == "Y":
                                    if tree_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: TREE")

                    # get the vegetation array
                    for land in os.listdir(landCover_path):
                        if land.endswith(".tif"):
                            if land.startswith(f"{year}"):
                                # featureName.append(f"{tree[:-4]}")
                                featureName.append(f"LandCover")
                                land_norm = read_aligned_raster(
                                src_path=landCover_path + land,
                                extent=samp_extent,
                                target_shape=target_shape
                                )
                                land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                                featureTuple += (land_norm,)
                                if shapeChecks == "Y":
                                    if land_norm.shape != target_shape:
                                        print(f"WRONG SHAPE FOR {sample}: Land")
   
                    
                    # # get all the features in the fodler 
                    for phv in os.listdir(phv_path):
                        if phv.endswith(".tif"):
                            featureName.append(f"{phv[:-4]}")
                            phv_data = read_aligned_raster(src_path=phv_path + phv, extent=samp_extent, target_shape=target_shape)
                            featureTuple += (phv_data,)
                            if shapeChecks == "Y":
                                if phv_data.shape != target_shape:
                                     print(f"WRONG SHAPE FOR {sample}: {phv}")
                                
                    feature_stack = np.dstack(featureTuple)
                    featureArray.append(feature_stack)
                    
                    # combined_target = np.stack([samp_flat, fsca_norm.flatten()], axis=-1)
                    targetArray.append(samp_flat)
                    # targetArray.append(samp_flat)
        return  np.array(featureArray), np.array(targetArray), featureName
