import numpy as np
import argparse

from pathlib import Path
from skimage.filters import threshold_otsu


parser = argparse.ArgumentParser(
    description="Create two-class attenuation image from TOF backprojection.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "tof_backproj_file", type=Path, help="Path to the TOF backprojection .npy file"
)
parser.add_argument(
    "--mu_water",
    type=float,
    default=0.0096,
    help="Attenuation coefficient for water (mm^-1)",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=None,
    help="Threshold value for segmentation (if not set, uses Otsu's method)",
)

args = parser.parse_args()

tof_backproj_file = args.tof_backproj_file
mu_water = args.mu_water
threshold = args.threshold

# %%

sens_file = tof_backproj_file.parent / tof_backproj_file.name.replace(
    "_tof_backproj_", "_sens_"
)

# Load the TOF backprojection file
tof_backproj = np.load(tof_backproj_file)

# Load the sensitivity file
sens_img = np.load(sens_file)

# Check that shapes match
if tof_backproj.shape != sens_img.shape:
    raise ValueError(
        f"Shape mismatch: TOF backproj {tof_backproj.shape} vs sensitivity {sens_img.shape}"
    )

# Create sensitivity-corrected backprojection by element-wise division
# Avoid division by zero by setting a small threshold
sens_img_safe = np.clip(sens_img, 1e-10, None)

corrected_backproj = tof_backproj / sens_img_safe

if threshold is None:
    threshold = threshold_otsu(corrected_backproj)

# Create two-class attenuation image
# Values <= otsu_threshold get 0 (air/background)
# Values > otsu_threshold get water_attenuation (tissue/water)
att_img = np.zeros_like(corrected_backproj, dtype=np.float32)
att_img[corrected_backproj > threshold] = mu_water

# safe to save the attenuation image
att_img_file = tof_backproj_file.parent / tof_backproj_file.name.replace(
    "_tof_backproj_", "_att_img_"
)

np.save(att_img_file, att_img)

print(f"Attenuation image saved to {att_img_file}")

# %%

import pymirc.viewer as pv

pv.ThreeAxisViewer([corrected_backproj, att_img])
