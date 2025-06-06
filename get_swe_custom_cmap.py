import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib_scalebar.scalebar import ScaleBar

def get_swe_custom_cmap(vmin=0.0001, vmax=3):
        """
        Custom SWE colormap:
          -1 → transparent
           0 → transparent
          0.0001–vmax → blue gradient
          >vmax → clipped to darkest blue
        """
        n_blue = 126
        blues = colormaps["Blues"](np.linspace(0.3, 1.0, n_blue))
    
        # Full colormap: transparent for -1 and 0, then blue for >0
        color_list = np.vstack([
            [1, 1, 1, 0],  # -1 = transparent
            [1, 1, 1, 0],  #  0 = transparent
            blues         # >0.0001 = blue gradient
        ])
        cmap = ListedColormap(color_list)
    
        # Bin edges for -1, 0, and >0
        edges = np.concatenate([
            [-1.5, -0.5],                                      # bin for -1
            np.linspace(-0.5 + vmin, vmax, n_blue + 1)         # bins for >0.0001
        ])
    
        norm = BoundaryNorm(edges, ncolors=cmap.N, clip=True)
        return cmap, norm
