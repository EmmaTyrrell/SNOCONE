import geopandas as gpd
import os
import pandas as pd

def safe_read_shapefile(path):
        with fiona.open(path, 'r') as src:
            return gpd.GeoDataFrame.from_features(src, crs=src.crs)
