import numpy as np
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

from utils import lm_file_fingerprint
from viewer import show_vol_cuts


def threshold_otsu(image: np.ndarray, nbins: int = 256) -> float:
    """Otsu threshold of an image (numpy re-implementation of skimage.filters.threshold_otsu)

    The threshold is the center of the histogram bin that maximizes the
    between-class variance of the two classes "<= threshold" and "> threshold".
    """
    image = np.asarray(image, dtype=np.float64).ravel()
    vmin, vmax = float(image.min()), float(image.max())

    if vmin == vmax:
        return vmin

    hist, bin_edges = np.histogram(image, bins=nbins, range=(vmin, vmax))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist = hist.astype(np.float64)

    # class probabilities and means for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # between-class variance for a threshold between bin i and i+1
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    return float(bin_centers[np.nanargmax(variance12)])



parser = argparse.ArgumentParser(
    description="Create two-class attenuation image from TOF backprojection.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "input_file",
    type=Path,
    help="TOF backprojection .npy file written by 02_reconstruct_petsird.py "
    "(<stem>_tof_backproj_<hash>.npy), or the PETSIRD listmode file it was created from",
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

input_file: Path = args.input_file
mu_water = args.mu_water
threshold = args.threshold

# %%
# find the TOF backprojection file

if not input_file.exists():
    raise SystemExit(f"File {input_file} does not exist")

if input_file.suffix == ".npy":
    tof_backproj_file = input_file
else:
    # PETSIRD listmode file: look for the TOF backprojection(s) written by
    # 02_reconstruct_petsird.py for this file (tagged with its fingerprint)
    candidates = sorted(
        input_file.parent.glob(
            f"{input_file.stem}_tof_backproj_{lm_file_fingerprint(input_file)}_*.npy"
        )
    )
    if len(candidates) == 0:
        raise SystemExit(
            f"No TOF backprojection found for {input_file}.\n"
            f"Run a TOF reconstruction first: python 02_reconstruct_petsird.py {input_file}"
        )
    if len(candidates) > 1:
        raise SystemExit(
            f"Found {len(candidates)} TOF backprojections for {input_file} "
            "(from reconstructions with different settings).\n"
            "Please pass one of them explicitly:\n"
            + "\n".join(f"  {c}" for c in candidates)
        )
    tof_backproj_file = candidates[0]
    print(f"Using TOF backprojection {tof_backproj_file}")

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

# keep references to the viewers (widgets) to keep them interactive
viewers = [
    show_vol_cuts(corrected_backproj, fig_title="sens. corrected TOF backprojection"),
    show_vol_cuts(att_img, fig_title="attenuation image"),
]
plt.show()
