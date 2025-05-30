def target_feature_stacks_testGroups(year, target_splits_path, fSCA_path, vegetation_path, landCover_path, phv_path, extension_filter, desired_shape, debug_output_folder, num_of_channels):
        ## create empty arrays
        featureArray = []
        targetArray = []
        extent_list = []
        crs_list = []
        
        # loop through the years and feature data
        # print(f"Processing {group}")
        targetSplits = target_splits_path
        fSCAWorkspace = fSCA_path
        for sample in os.listdir(targetSplits):
            featureTuple = ()
            featureName = []
            # loop through each sample and get the corresponding features
            if sample.endswith(extension_filter):
                # read in data
                with rasterio.open(targetSplits + sample) as samp_src:
                    samp_data = samp_src.read(1)
                    meta = samp_src.meta.copy()
                    samp_extent = samp_src.bounds
                    samp_transform = samp_src.transform
                    samp_crs = samp_src.crs
                    # apply a no-data mask
                    mask = samp_data >= 0
                    msked_target = np.where(mask, samp_data, -1)
                    target_shape = msked_target.shape
        
                    # flatted data
                    samp_flat = msked_target.flatten()
                    
    
                # try to get the fsca variables 
                sample_root = "_".join(sample.split("_")[:2])
                for fSCA in os.listdir(fSCAWorkspace):
                    if fSCA.endswith(extension_filter) and fSCA.startswith(sample_root):
                        featureName.append(f"{fSCA[:-4]}")
                        fsca_norm = read_aligned_raster(src_path=fSCAWorkspace + fSCA, extent=samp_extent, target_shape=target_shape)
                        fsca_norm = min_max_scale(fsca_norm, min_val=0, max_val=100)
                        featureTuple += (fsca_norm,)
                        # print(fsca_norm.shape)
                        if fsca_norm.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: FSCA")
                            output_debug_path = debug_output_folder + f"/{sample_root}_BAD_FSCA.tif"
                            save_array_as_raster(
                                output_path=output_debug_path,
                                array=fsca_norm,
                                extent=samp_extent,
                                crs=samp_crs,
                                nodata_val=-1
                            )
        
                # get a DOY array into a feature 
                date_string = sample.split("_")[1]
                doy_str = date_string[-3:]
                doy = float(doy_str)
                DOY_array = np.full_like(msked_target, doy)
                doy_norm = min_max_scale(DOY_array,  min_val=0, max_val=366)
                featureTuple += (doy_norm,)
                featureName.append(doy)
        
                # get the vegetation array
                for tree in os.listdir(vegetation_path):
                    if tree.endswith(extension_filter):
                        if tree.startswith(f"{year}"):
                            featureName.append(f"{tree[:-4]}")
                            tree_norm = read_aligned_raster(
                            src_path=tree_workspace + tree,
                            extent=samp_extent,
                            target_shape=target_shape
                            )
                            tree_norm = min_max_scale(tree_norm, min_val=0, max_val=100)
                            featureTuple += (tree_norm,)
                            if tree_norm.shape != desired_shape:
                                print(f"WRONG SHAPE FOR {sample}: TREE")
                                output_debug_path = debug_output_folder + f"/{sample_root}_BAD_TREE.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                # get the vegetation array
                for land in os.listdir(land_workspace):
                    if land.endswith(".tif"):
                        if land.startswith(f"{year}"):
                            # featureName.append(f"{tree[:-4]}")
                            featureName.append(f"LandCover")
                            land_norm = read_aligned_raster(
                            src_path=land_workspace + land,
                            extent=samp_extent,
                            target_shape=target_shape
                            )
                            land_norm = min_max_scale(land_norm, min_val=11, max_val=95)
                            featureTuple += (land_norm,)
                            if land_norm.shape != (256, 256):
                                print(f"WRONG SHAPE FOR {sample}: Land")
                                # output_debug_path = f"./debug_output/{sample_root}_BAD_TREE.tif"
                                save_array_as_raster(
                                    output_path=output_debug_path,
                                    array=fsca_norm,
                                    extent=samp_extent,
                                    crs=samp_crs,
                                    nodata_val=-1
                                )
                
                # # get all the features in the fodler 
                for phv in os.listdir(phv_path):
                    if phv.endswith(extension_filter):
                        featureName.append(f"{phv[:-4]}")
                        phv_data = read_aligned_raster(src_path=phv_features + phv, extent=samp_extent, target_shape=target_shape)
                        featureTuple += (phv_data,)
                        if phv_data.shape != desired_shape:
                            print(f"WRONG SHAPE FOR {sample}: {phv}")
                            output_debug_path = debug_output_folder + f"/{sample_root}_BAD_{phv[:-4]}.tif"
                            save_array_as_raster(
                                output_path=output_debug_path,
                                array=fsca_norm,
                                extent=samp_extent,
                                crs=samp_crs,
                                nodata_val=-1
                            )
                feature_stack = np.dstack(featureTuple)
                if feature_stack.shape[2] != num_of_channels:
                    print(f"{sample} has shape {feature_stack.shape} — missing or extra feature?")
                    print(featureName)
                    print(" ")
                else:
                    featureArray.append(feature_stack)
                    # y_stack = np.stack([msked_target, fsca], axis=-1).astype(np.float32)
                    targetArray.append(samp_flat)
                    extent_list.append(samp_extent)
                    crs_list.append(samp_crs)
        return  np.array(featureArray), np.array(targetArray), extent_list, crs_list
