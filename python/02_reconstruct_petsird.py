import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import parallelproj
import petsird

from utils import (
    get_all_detector_centers,
    read_listmode_prompt_events,
    backproject_efficiencies,
)


################################################################################
################################################################################
################################################################################

# fname = "sim_points_400000_0/simulated_petsird_lm_file.bin"
fname = "data/sim_points_400000_0/simulated_petsird_lm_file.bin"

img_shape = (100, 100, 11)  # shape of the image to be reconstructed
voxel_size = (1.0, 1.0, 1.0)
fwhm_mm = 1.5
store_energy_bins = True
num_epochs = 5
num_subsets = 20

################################################################################
################################################################################
################################################################################

reader = petsird.BinaryPETSIRDReader(fname)
header: petsird.Header = reader.read_header()
scanner_info: petsird.ScannerInformation = header.scanner
scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry

num_replicated_modules = scanner_geom.number_of_replicated_modules()
print(f"Scanner with {num_replicated_modules} types of replicated modules.")

################################################################################
################################################################################
################################################################################

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# get all detector centers
all_detector_centers = get_all_detector_centers(scanner_geom, ax=ax)

################################################################################
################################################################################
################################################################################

sens_img = backproject_efficiencies(
    scanner_info, all_detector_centers, img_shape, voxel_size
)

# apply adjoint of image-based resolution model
sig = fwhm_mm / (2.35 * np.asarray(voxel_size))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)
sens_img = res_model.adjoint(sens_img)

################################################################################
################################################################################
################################################################################

# %%
# read the prompt events of all time blocks for all combinations of module types

coords0, coords1, signed_tof_bins, effs, energy_idx0, energy_idx1 = (
    read_listmode_prompt_events(
        reader, header, all_detector_centers, store_energy_bins=True
    )
)

################################################################################
################################################################################
################################################################################
# %%
# set axis limits

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

    for i in range(5):
        ax.plot(
            [coords0[i, 0], coords1[i, 0]],
            [coords0[i, 1], coords1[i, 1]],
            [coords0[i, 2], coords1[i, 2]],
        )
    fig.show()

################################################################################
################################################################################
################################################################################
# %%

#### HACK assumes same TOF parameters for all module type pairs
sigma_tof = scanner_info.tof_resolution[0][0] / 2.35
tof_bin_edges = scanner_info.tof_bin_edges[0][0].edges
num_tofbins = tof_bin_edges.size - 1
tofbin_width = float(tof_bin_edges[1] - tof_bin_edges[0])

tof_params = parallelproj.TOFParameters(
    num_tofbins=num_tofbins, tofbin_width=tofbin_width, sigma_tof=sigma_tof
)

lm_subset_projs = []
subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

recon = np.ones(img_shape, dtype="float32")

for i_subset, sl in enumerate(subset_slices):
    lm_subset_projs.append(
        parallelproj.ListmodePETProjector(
            coords0[sl, :].copy(), coords1[sl, :].copy(), img_shape, voxel_size
        )
    )
    ### HACK assumes same TOF parameters for all module type pairs
    lm_subset_projs[i_subset].tof_parameters = tof_params
    lm_subset_projs[i_subset].event_tofbins = signed_tof_bins[sl].copy()
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
np.save(opath, recon)
print(f"LM OSEM recon saved to {opath}")
