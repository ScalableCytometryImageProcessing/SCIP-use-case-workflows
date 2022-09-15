# AUTOGENERATED! DO NOT EDIT! File to edit: ../workflow/notebooks/core/00_threshold_masking.ipynb.

# %% auto 0
__all__ = ['plot_scip_segmentation']

# %% ../workflow/notebooks/core/00_threshold_masking.ipynb 2
from .common import *

# %% ../workflow/notebooks/core/00_threshold_masking.ipynb 4
import zarr
from scip.masking import threshold, remove_regions_touching_border, get_bounding_box

# %% ../workflow/notebooks/core/00_threshold_masking.ipynb 5
def plot_scip_segmentation(r, bbox_channel_index=0, smooth=1, border=True):
    z = zarr.open(r.meta_path)
    pixels = z[r.meta_zarr_idx].reshape(z.attrs["shape"][r.meta_zarr_idx])
    pixels = numpy.clip(pixels, a_min=0, a_max=4096)

    m = threshold.get_mask(dict(pixels=pixels), main_channel=bbox_channel_index, smooth=smooth)
    m = get_bounding_box(m)
    if border:
        m = remove_regions_touching_border(m, bbox_channel_index=bbox_channel_index)

    fig, axes = plt.subplots(2, len(pixels), dpi=150, squeeze=False)
    for i, (a, p) in enumerate(zip(m["mask"], pixels)):
        axes[0, i].imshow(a)
        axes[0, i].set_axis_off()
        axes[1, i].imshow(p)
        axes[1, i].set_axis_off()

    return m
