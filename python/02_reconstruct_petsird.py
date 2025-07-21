import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import parallelproj
import petsird

# for the parallelproj recon we can use cupy as array backend if available
# otherwise we fall back to numpy
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp
import argparse

print(f"Using {xp.__name__} for parallelproj reconstructions")

from utils import (
    get_all_detector_centers,
    read_listmode_prompt_events,
    backproject_efficiencies,
)


# %%
################################################################################
#### PARSE THE COMMAND LINE ####################################################
################################################################################

parser = argparse.ArgumentParser(
    description="PETSIRD analytic simulator reconstruction"
)
parser.add_argument("fname", type=str, help="Path to the PETSIRD listmode file")
parser.add_argument(
    "--img_shape",
    type=int,
    nargs=3,
    default=[55, 55, 19],
    help="Shape of the image to be reconstructed",
)
parser.add_argument(
    "--voxel_size",
    type=float,
    nargs=3,
    default=[2.0, 2.0, 2.0],
    help="Voxel size in mm",
)
parser.add_argument(
    "--fwhm_mm",
    type=float,
    default=2.5,
    help="FWHM in mm for Gaussian filter for resolution model",
)
parser.add_argument(
    "--store_energy_bins", action="store_true", help="Whether to store energy bins"
)
parser.add_argument("--num_epochs", type=int, default=5, help="Number of OSEM epochs")
parser.add_argument(
    "--num_subsets", type=int, default=20, help="Number of OSEM subsets"
)
parser.add_argument(
    "--verbose", action="store_true", help="Whether to print verbose output"
)
parser.add_argument(
    "--unity_sens_img",
    action="store_true",
    help="Whether to skip sensitivity image calculation and use unity image",
)
parser.add_argument(
    "--unity_effs",
    action="store_true",
    help="Whether to use unity efficiencies for LORs",
)
parser.add_argument(
    "--non-tof", action="store_true", help="Whether to disable TOF in recon"
)

args = parser.parse_args()

fname = args.fname
img_shape = tuple(args.img_shape)
voxel_size = tuple(args.voxel_size)
fwhm_mm = args.fwhm_mm
store_energy_bins = args.store_energy_bins
num_epochs = args.num_epochs
num_subsets = args.num_subsets
verbose = args.verbose
unity_sens_img = args.unity_sens_img
unity_effs = args.unity_effs
tof = not args.non_tof

# %%
################################################################################
#### READ PETSIRD HEADER #######################################################
################################################################################

reader = petsird.BinaryPETSIRDReader(fname)
header: petsird.Header = reader.read_header()
scanner_info: petsird.ScannerInformation = header.scanner
scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry

num_replicated_modules = scanner_geom.number_of_replicated_modules()
print(f"Scanner with {num_replicated_modules} types of replicated modules.")

# %%
################################################################################
### READ DETECTOR CENTERS AND VISUALIZE ########################################
################################################################################

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

print("Calculating all detector centers ...")
all_detector_centers = get_all_detector_centers(scanner_geom, ax=ax)

# calculate the scanner iso center to set the image origin that we need for the projectors
scanner_iso_center = xp.asarray(all_detector_centers[0].reshape(-1, 3).mean(0))

img_origin = scanner_iso_center - 0.5 * (xp.asarray(img_shape) - 1) * xp.asarray(
    voxel_size
)

if not ax is None:
    min_coords = all_detector_centers[0].reshape(-1, 3).min(0)
    max_coords = all_detector_centers[0].reshape(-1, 3).max(0)

    ax.set_xlim3d([min_coords.min(), max_coords.max()])
    ax.set_ylim3d([min_coords.min(), max_coords.max()])
    ax.set_zlim3d([min_coords.min(), max_coords.max()])

    for detector_centers in all_detector_centers:
        ax.scatter(
            detector_centers[:, :, 0].ravel(),
            detector_centers[:, :, 1].ravel(),
            detector_centers[:, :, 2].ravel(),
            s=0.5,
            c="k",
            alpha=0.3,
        )
    fig.show()

# %%
################################################################################
### CALCULATE THE SENSITIVTY IMAGE #############################################
################################################################################

sig = fwhm_mm / (2.35 * np.asarray(voxel_size))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

if unity_sens_img:
    print("Using ones as sensitivity image ...")
    sens_img = np.ones(img_shape, dtype="float32")
else:
    print("Calculating sensitivity image ...")
    sens_img: np.ndarray = backproject_efficiencies(
        scanner_info,
        all_detector_centers,
        img_shape,
        voxel_size,
        verbose=verbose,
        tof=tof,
    )

    # apply adjoint of image-based resolution model
    sens_img = res_model.adjoint(sens_img)

# %%
################################################################################
### CHECK WHETHER THE CALC SENS IMAGE MATCHES THE REF. SENSE IMAGE #############
################################################################################

ref_sens_img_path = Path(fname).parent / "reference_sensitivity_image.npy"

if ref_sens_img_path.exists():
    print(f"loading reference sensitivity image from {ref_sens_img_path}")
    ref_sens_img = np.load(ref_sens_img_path)
    if ref_sens_img.shape == sens_img.shape:
        if np.allclose(sens_img, ref_sens_img):
            print(
                f"calculated sensitivity image matches reference image {ref_sens_img_path}"
            )
        else:
            print(
                f"calculated sensitivity image does NOT match reference image {ref_sens_img_path}"
            )

# %%
################################################################################
### READ THE LM PROMPT EVENTS AND CONVERT TO COODINATES ########################
################################################################################

print("Reading prompt events from listmode file ...")

coords0, coords1, signed_tof_bins, effs, energy_idx0, energy_idx1 = (
    read_listmode_prompt_events(
        reader,
        header,
        all_detector_centers,
        store_energy_bins=True,
        verbose=verbose,
        unity_effs=unity_effs,
    )
)

print(signed_tof_bins.min(), signed_tof_bins.max())

# %%
################################################################################
### VISUALIZE GEOMETRY AND FIRST 5 EVENTS ######################################
################################################################################
if not ax is None:
    for i in range(5):
        ax.plot(
            [coords0[i, 0], coords1[i, 0]],
            [coords0[i, 1], coords1[i, 1]],
            [coords0[i, 2], coords1[i, 2]],
        )
    fig.show()

# %%
################################################################################
### PARALLELRPOJ BACKPROJECTIONS ###############################################
################################################################################

proj = parallelproj.ListmodePETProjector(
    xp.asarray(coords0).copy(),
    xp.asarray(coords1).copy(),
    img_shape,
    voxel_size,
    img_origin=img_origin,
)

non_tof_backproj = proj.adjoint(xp.ones(coords0.shape[0], dtype="float32"))

#### HACK assumes same TOF parameters for all module type pairs
sigma_tof = scanner_info.tof_resolution[0][0] / 2.35
tof_bin_edges = scanner_info.tof_bin_edges[0][0].edges
num_tofbins = tof_bin_edges.size - 1
tofbin_width = float(tof_bin_edges[1] - tof_bin_edges[0])

tof_params = parallelproj.TOFParameters(
    num_tofbins=num_tofbins, tofbin_width=tofbin_width, sigma_tof=sigma_tof
)

proj.tof_parameters = tof_params
proj.event_tofbins = xp.asarray(signed_tof_bins).copy()
proj.tof = True

tof_backproj = proj.adjoint(xp.ones(coords0.shape[0], dtype="float32"))

del proj
# %%
################################################################################
### PARALLELRPOJ LM OSEM RECONSTRUCTION ########################################
################################################################################

print("Starting parallelproj LM OSEM reconstruction ...")


lm_subset_projs = []
subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

# init recon, sens and eff arrays and covert to xp (numpy or cupy) arrays
recon = xp.ones(img_shape, dtype="float32")
sens_img = xp.asarray(sens_img, dtype="float32")
effs = xp.asarray(effs, dtype="float32")

for i_subset, sl in enumerate(subset_slices):
    lm_subset_projs.append(
        parallelproj.ListmodePETProjector(
            xp.asarray(coords0[sl, :]).copy(),
            xp.asarray(coords1[sl, :]).copy(),
            img_shape,
            voxel_size,
            img_origin=img_origin,
        )
    )

    #### HACK assumes same TOF parameters for all module type pairs
    if tof:
        lm_subset_projs[i_subset].tof_parameters = tof_params
        lm_subset_projs[i_subset].event_tofbins = xp.asarray(signed_tof_bins[sl]).copy()
        lm_subset_projs[i_subset].tof = True

for i_epoch in range(num_epochs):
    for i_subset, sl in enumerate(subset_slices):
        print(
            f"it {(i_epoch +1):03} / {num_epochs:03}, ss {(i_subset+1):03} / {num_subsets:03}",
            end="\r",
        )
        lm_exp = effs[sl] * lm_subset_projs[i_subset](res_model(recon))
        tmp = num_subsets * res_model.adjoint(
            lm_subset_projs[i_subset].adjoint(effs[sl] / lm_exp)
        )
        recon *= tmp / sens_img

opath = Path(fname).parent / f"lm_osem_{num_epochs}_{num_subsets}.npy"
xp.save(opath, recon)
print(f"LM OSEM recon saved to {opath}")

# %%
# SHOW RECON
import pymirc.viewer as pv

vi = pv.ThreeAxisViewer([parallelproj.to_numpy_array(x) for x in [recon, sens_img]])
