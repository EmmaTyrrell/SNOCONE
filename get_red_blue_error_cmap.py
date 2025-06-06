import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib_scalebar.scalebar import ScaleBar

def get_red_blue_error_cmap(vmin=-100, vcenter=0, vmax=1000, steps=256):
        """
        Custom diverging colormap:
          - Red for underestimates (negative),
          - White at 0,
          - Blue for overestimates (positive).
        Ensures 0 is centered using BoundaryNorm.
        """
        assert vmin < vcenter < vmax, "vcenter must lie between vmin and vmax"
    
        # Determine how many colors to allocate left and right of center
        total_range = vmax - vmin
        neg_frac = abs(vcenter - vmin) / total_range
        pos_frac = abs(vmax - vcenter) / total_range
        n_neg = int(steps * neg_frac)
        n_pos = int(steps * pos_frac)
    
        # Sample colormaps
        reds = colormaps["Reds_r"](np.linspace(0.2, 1.0, n_neg))  # from light pink to red
        blues = colormaps["Blues"](np.linspace(0.2, 1.0, n_pos))  # from light blue to dark blue
    
        # Combine red + white + blue
        white = np.array([[1.0, 1.0, 1.0, 1.0]])
        full_colors = np.vstack([reds, white, blues])
        cmap = ListedColormap(full_colors)
    
        # Build bin edges so each color bin has equal step size (important for BoundaryNorm)
        boundaries = np.concatenate([
            np.linspace(vmin, vcenter, n_neg, endpoint=False),  # red bins
            [vcenter],
            np.linspace(vcenter, vmax, n_pos + 1)               # blue bins
        ])
    
        norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    
        return cmap, norm
