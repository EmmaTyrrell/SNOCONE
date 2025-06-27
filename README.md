Starting here with a dictionary of all the functions used and where they are stored.
<br>
### Function Descriptions and Locations ##
**CNN_pre-processing.py**
<br>1) *min_max_scale*: This function conducts a min-max scaler on the features. You can either have pre-set minimum and maximum values or you can have it be dynamic
<br>2) *read_aligned_raster*: This reads in a raster file and clips it to the extent of the bounds of another raster. This is used when stacking featuers on a training sample. It's important that all samples and features are on the same coordinate system and grid alignment.
<br>3) *save_array_as_raster*: This saves a numpy array as a geoTIF file. This is used for QC'ing files as well as saving the final model predictions.
<br>4) *target_feature_stacks*: This is a critical function that assembles the arrays used as inputs to the model training. It starts with reading in a sample and then gathering all the features for the same location and extent of the sample. This is not a fully dynamic code and will need to be updated if a feature is included that is temporally restrictive, as in is not static. This includes vegetation cover and land cover (vary annually) and DOY, fSCA, and DMFSCA (vary daily). This is designed to read in the training samples per year. 
<br>5) *target_feature_stacks_testGroups*: This is a similar function as the one before yet is specifically designed for the test groups. Test groups are separated into folders "Group1", "Group2", etc. and within each have a "test" and "train" folder. Since these are not varied by year, a year parameter is required to selecte the correct temporal variables.
<br>
<br>**CNN_SNOTELComparisons.py**
<br>1) *download_and_merge_snotel_data*: This function is designed to download and process SNOTEL data from a start date and an end date given a list of station ID's and state abrev.
<br>2) *get_snotel_raster_values*: This function extracts the raster cell value at the same location as a snotel value. 
<br>
<br>**CNN_errorVisualization.py**
<br>1) *safe_read_shapefile*: Reads in a shapefile since there as a version compatiability issue. 
<br>2) *get_swe_custom_cmap*: Blue color scale for SWE values.
<br>3) *get_red_blue_error_cmap*: Percent error diverging colormap where negative percent errors are red and positive percent errors are blue.
<br>4) *improved_mosaic_blending_rasterio*: Cosine mosaicing methods for stitching and blending the square output samples from the model predictions.
<br>5) *read_raster_info*: Streamlined way to read in raster data. 
<br>6) *align_raster*: Aligns a raster to the coordinate system, cell size, extent, and grid locations as another reference raster. 
<br>7) *compute_percent_difference_map*: Computes the percent error for a raster against the validation raster. 
<br>
<br>**CNN_benchmarks.py**
<br>1) *swe_fsca_consistency_loss_fn*: This is a penalty to the CNN that punishes the model if it predicts SWE where there is zero fSCA and predicts zero SWE where fSCA is greater than zero. This is read in with a tensorflow serializable method. 
<br>2) *make_swe_fsca_loss*: This initalizes the fSCA penality into the model's loss function.
<br>3) *masked_loss_fn*: A custom loss function that only records the loss function over non-null data cells.
<br>4) *masked_mse*: A custom loss function that only records the mean squared error (MSE) over non-null data cells.
<br>5) *masked_mae*: A custom loss function that only records the mean absolute error (MAE) over non-null data cells.
<br>5) *masked_rmse*: A custom loss function that only records the root mean squared error (RMSE) over non-null data cells.
<br>
<br>**CNN_memoryOptimization.py**
<br>1) *class DataGenerator(Sequence)*: This is a data generator that only feeds in data in batches into the model to prevent memory overloads.
<br>2) *clear_memory*: Clears the memory in the code to prevent memory overload issues.
<br>3) *memory_efficient_prediction*: This functions conducts memory predictions in smaller batches to reduce memory usage.
