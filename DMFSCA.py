#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import modules
import rasterio
import numpy as np
import os 
from datetime import datetime, timedelta
print("modules imported")

# establish paths
dmfscaWS = "D:/ASOML/DMFSCA/"
originalFiles = dmfscaWS + "totalFiles/"
output = dmfscaWS + "outputDMFSCA/"

# getting average
start_date = datetime(2024, 10, 1)
end_date = datetime(2025, 5, 26)

# sort files
raster_files = sorted([f for f in os.listdir(originalFiles) if f.endswith(".tif")])

# maps dates to file paths
raster_dict = {}
for f in raster_files:
    try:
        date_str = os.path.splitext(f)[0]  # assumes filename like "20241001.tif"
        file_date = datetime.strptime(date_str, "%Y%m%d")
        raster_dict[file_date] = os.path.join(originalFiles, f)
    except ValueError:
        continue

# Sort the dictionary by date
sorted_dates = sorted(raster_dict.keys())

for current_date in sorted_dates:
    if current_date < start_date or current_date > end_date:
        continue

    # Get all dates up to and including the current date
    date_subset = [d for d in sorted_dates if d <= current_date]

    # Initialize accumulator
    sum_array = None
    count = 0

    for d in date_subset:
        with rasterio.open(raster_dict[d]) as src:
            data = src.read(1).astype(np.float32)
            mask = data == src.nodata
            data[mask] = 0  # exclude NoData from sum
            if sum_array is None:
                sum_array = np.zeros_like(data)
                valid_mask = np.zeros_like(data, dtype=np.int32)
            sum_array += data
            valid_mask += ~mask  # count valid pixels
            profile = src.profile

    # Compute average
    with np.errstate(invalid='ignore'):
        avg_array = np.divide(sum_array, valid_mask, where=valid_mask != 0)
        avg_array[valid_mask == 0] = profile['nodata']  # restore NoData

    # Save output
    out_filename = os.path.join(output, f"{current_date.strftime('%Y%m%d')}_dmfsca.tif")
    with rasterio.open(out_filename, "w", **profile) as dst:
        dst.write(avg_array, 1)

    print(f"Wrote: {out_filename}")

