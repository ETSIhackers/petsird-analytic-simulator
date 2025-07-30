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
    default=None,
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
parser.add_argument(
    "--attenuation_image", type=str, default=None, help="Path to attenuation image"
)

args = parser.parse_args()

fname = args.fname
img_shape: list[int] | None = args.img_shape
voxel_size = tuple(args.voxel_size)
fwhm_mm = args.fwhm_mm
store_energy_bins = args.store_energy_bins
num_epochs = args.num_epochs
num_subsets = args.num_subsets
verbose = args.verbose
unity_sens_img = args.unity_sens_img
unity_effs = args.unity_effs
tof = not args.non_tof
attenuation_image_fname: str | None = args.attenuation_image

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

all_detector_centers_file = Path(fname).with_suffix(".detector_centers.npz")

if not all_detector_centers_file.exists():
    print("Calculating all detector centers ...")
    all_detector_centers = get_all_detector_centers(scanner_geom, ax=ax)
    np.savez_compressed(all_detector_centers_file, *all_detector_centers)
    print(f"Detector centers saved to {all_detector_centers_file}")
else:
    print(f"Loading all detector centers from {all_detector_centers_file}")
    with np.load(all_detector_centers_file) as data:
        all_detector_centers = [data[name] for name in data.files]

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
###  DETERMINE IMAGE ORIGIN AND IMAGE SHAPE IF NOT GIVEN #######################
################################################################################


if img_shape is not None:
    img_shape = tuple(img_shape)
else:
    # get the bounding box of the scanner detection elements
    scanner_bbox = all_detector_centers[0].reshape(-1, 3).max(0) - all_detector_centers[
        0
    ].reshape(-1, 3).min(0)

    i_ax = int(
        np.argmin(
            np.array(
                [
                    np.abs(scanner_bbox[1] - scanner_bbox[2]),
                    np.abs(scanner_bbox[0] - scanner_bbox[2]),
                    np.abs(scanner_bbox[0] - scanner_bbox[1]),
                ]
            )
        )
    )

    img_shape = (0.53 * scanner_bbox / np.array(voxel_size)).astype(int)
    img_shape[i_ax] = int(scanner_bbox[i_ax] / voxel_size[i_ax])
    if img_shape[i_ax] % 2 == 0:
        img_shape[i_ax] += 1  # make sure the image shape is odd
    img_shape: tuple[int, int, int] = tuple(img_shape.tolist())

# calculate the scanner iso center to set the image origin that we need for the projectors
scanner_iso_center = xp.asarray(all_detector_centers[0].reshape(-1, 3).mean(0))
img_origin = scanner_iso_center - 0.5 * (xp.asarray(img_shape) - 1) * xp.asarray(
    voxel_size
)

if verbose:
    print(f"Image shape: {img_shape}")
    print(f"Image origin: {img_origin}")
    print(f"Voxel size: {voxel_size}")

# %%
################################################################################
### CALCULATE THE SENSITIVTY IMAGE #############################################
################################################################################

sig = fwhm_mm / (2.35 * np.asarray(voxel_size))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

att_img: None | np.ndarray = None
if attenuation_image_fname is not None:
    print(f"Loading attenuation image from {attenuation_image_fname}")
    att_img = np.load(attenuation_image_fname)
    if att_img.shape != img_shape:
        raise ValueError(
            f"Attenuation image shape {att_img.shape} does not match image shape {img_shape}"
        )


if unity_sens_img:
    print("Using ones as sensitivity image ...")
    sens_img = xp.ones(img_shape, dtype="float32")
else:
    sens_img_path = Path(fname).with_suffix(".sens_img.npy")

    if sens_img_path.exists():
        print(f"Loading sensitivity image from {sens_img_path}")
        sens_img = xp.load(sens_img_path)

    else:
        print("Calculating sensitivity image ...")

        sens_img: xp.ndarray = backproject_efficiencies(
            scanner_info,
            all_detector_centers,
            img_shape,
            voxel_size,
            verbose=verbose,
            tof=tof,
            xp=xp,
            attenuation_image=att_img,
        )

    # apply adjoint of image-based resolution model
    sens_img = res_model.adjoint(sens_img)

    xp.save(sens_img_path, sens_img)

# %%
################################################################################
### CHECK WHETHER THE CALC SENS IMAGE MATCHES THE REF. SENSE IMAGE #############
################################################################################

ref_sens_img_path = Path(fname).parent / "reference_sensitivity_image.npy"

if ref_sens_img_path.exists():
    print(f"loading reference sensitivity image from {ref_sens_img_path}")
    ref_sens_img = np.load(ref_sens_img_path)
    if ref_sens_img.shape == sens_img.shape:
        if np.allclose(np.asarray(sens_img), ref_sens_img):
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

preprocess_file = Path(fname).with_suffix(".preprocessed_events.npz")

if not preprocess_file.exists():
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

    # store coords0, coords1, signed_tof_bins, effs, energy_idx0, energy_idx1 into a compressed npz file
    np.savez_compressed(
        preprocess_file,
        coords0=coords0,
        coords1=coords1,
        signed_tof_bins=signed_tof_bins,
        effs=effs,
        energy_idx0=energy_idx0,
        energy_idx1=energy_idx1,
    )
else:
    print(f"Loading preprocessed events from {preprocess_file}")
    with np.load(preprocess_file) as data:
        coords0 = data["coords0"]
        coords1 = data["coords1"]
        signed_tof_bins = data["signed_tof_bins"]
        effs = data["effs"]
        energy_idx0 = data["energy_idx0"]
        energy_idx1 = data["energy_idx1"]

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

xp.save(Path(fname).with_suffix(".non_tof_backproj.npy"), non_tof_backproj)

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

xp.save(Path(fname).with_suffix(".tof_backproj.npy"), tof_backproj)

del proj

# %%
################################################################################
### FILTER OUT EVENTS THAT HAVE 0 EFFICIENCY DUE TO ERRORS IN ENCODING #########
################################################################################

if effs.min() == 0:
    print("Filtering out events with zero efficiency ...")
    valid_events = effs > 0
    coords0 = coords0[valid_events]
    coords1 = coords1[valid_events]
    signed_tof_bins = signed_tof_bins[valid_events]
    effs = effs[valid_events]

    if store_energy_bins:
        energy_idx0 = energy_idx0[valid_events]
        energy_idx1 = energy_idx1[valid_events]


# %%
################################################################################
### PARALLELRPOJ LM OSEM RECONSTRUCTION WITHOUT ATTENUATION MODEL ##############
################################################################################

# NOTE: since we don't model additive contamination, the attenuation only enters
#       the sensitivity image calculation

print("Starting parallelproj LM OSEM reconstruction ...")

lm_subset_projs = []
subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

# init recon, sens and eff arrays and covert to xp (numpy or cupy) arrays
recon = xp.ones(img_shape, dtype="float32")
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

if att_img is None:
    vi = pv.ThreeAxisViewer([parallelproj.to_numpy_array(x) for x in [recon, sens_img]])
else:
    vi = pv.ThreeAxisViewer(
        [parallelproj.to_numpy_array(x) for x in [recon, att_img, sens_img]]
    )
