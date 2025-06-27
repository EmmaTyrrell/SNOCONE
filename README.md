Starting here with a dictionary of all the functions used and where they are stored.
<br>
# Function Descriptions and Locations ##
<br>**CNN_pre-processing.py**
<br>1) *min_max_scale*: This function conducts a min-max scaler on the features. You can either have pre-set minimum and maximum values or you can have it be dynamic
<br>2) *read_aligned_raster*: This reads in a raster file and clips it to the extent of the bounds of another raster. This is used when stacking featuers on a training sample. It's important that all samples and features are on the same coordinate system and grid alignment.
<br>3) *save_array_as_raster*: This saves a numpy array as a geoTIF file. This is used for QC'ing files as well as saving the final model predictions.
<br>4) *target_feature_stacks*: This is a critical function that assembles the arrays used as inputs to the model training. It starts with reading in a sample and then gathering all the features for the same location and extent of the sample. This is not a fully dynamic code and will need to be updated if a feature is included that is temporally restrictive, as in is not static. This includes vegetation cover and land cover (vary annually) and DOY, fSCA, and DMFSCA (vary daily). This is designed to read in the training samples per year. 
<br>5) *target_feature_stacks_testGroups*: This is a similar function as the one before yet is specifically designed for the test groups. Test groups are separated into folders "Group1", "Group2", etc. and within each have a "test" and "train" folder. Since these are not varied by year, a year parameter is required to selecte the correct temporal variables.
<br>
<br>**CNN_SNOTELComparisons.py**
<br>1) *download_and_merge_snotel_data*: This function is designed to download and process SNOTEL data from a start date and an end date given a list of station ID's and state abrev.
<br>2) *get_snotel_raster_values*: This function extracts the raster cell value at the same location as a snotel value. 
