from importlib.metadata import version

# raise an error if petsird version is not at least 0.7.2
petsird_version = tuple(map(int, version("petsird").split(".")))
if petsird_version < (0, 7, 2):
    raise ImportError(
        f"petsird version {petsird_version} is not supported, please install petsird >= 0.7.2"
    )


import numpy as np
import argparse
from array_api_compat import size
from itertools import combinations

import parallelproj
import matplotlib.pyplot as plt
import math
import json

from pathlib import Path

import petsird

import petsird.helpers.geometry
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_BoxShape(ax, box: petsird.BoxShape) -> None:
    vertices = np.array([c.c for c in box.corners])
    edges = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [4, 7, 3, 0]],
    ]
    box_poly = Poly3DCollection(edges, alpha=0.1, linewidths=0.25, edgecolors="r")
    ax.add_collection3d(box_poly)


def get_all_detector_centers(
    scanner_geometry: petsird.ScannerGeometry, ax=None
) -> list[np.ndarray]:
    # a list containing the center of all detecting elements for all modules
    # every element of the list corresponds to one module type
    # for every module type, we have an numpy array of shape (num_modules, num_det_els, 3)
    # for a given module type, module number and detector el number, we can access the center of the detector element with
    # all_det_el_centers[module_type][module_number, detector_el_number, :]

    all_det_el_centers = []

    # draw all crystals
    for rep_module in scanner_geometry.replicated_modules:
        det_els = rep_module.object.detecting_elements
        det_el_centers = np.zeros(
            (len(rep_module.transforms), len(det_els.transforms), 3)
        )
        for i_mod, mod_transform in enumerate(rep_module.transforms):

            for i_det_el, transform in enumerate(det_els.transforms):
                transformed_boxshape = (
                    petsird.helpers.geometry.transform_BoxShape(
                        petsird.helpers.geometry.mult_transforms(
                            [mod_transform, transform]
                        ),
                        det_els.object.shape,
                    ),
                )[0]

                transformed_boxshape_vertices = np.array(
                    [c.c for c in transformed_boxshape.corners]
                )

                det_el_centers[i_mod, i_det_el, :] = transformed_boxshape_vertices.mean(
                    axis=0
                )

                if ax is not None:
                    draw_BoxShape(
                        ax,
                        transformed_boxshape,
                    )
        all_det_el_centers.append(det_el_centers)

    return all_det_el_centers


################################################################################
################################################################################
################################################################################

fname = "sim_points_400000_0/simulated_petsird_lm_file.bin"

img_shape = (100, 100, 11)  # shape of the image to be reconstructed
voxel_size = (1.0, 1.0, 1.0)

# %%

reader = petsird.BinaryPETSIRDReader(fname)
header: petsird.Header = reader.read_header()
scanner_info: petsird.ScannerInformation = header.scanner
scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry

num_replicated_modules = scanner_geom.number_of_replicated_modules()
print(f"Scanner with {num_replicated_modules} types of replicated modules.")

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax = None

# get all detector centers
all_detector_centers = get_all_detector_centers(scanner_geom, ax=ax)

################################################################################
################################################################################
################################################################################
# %%
# read detection element / module pair efficiencies

# get the dection bin efficiencies for all module types
# index via: det_bin_effs = all_detection_bin_effs[rep_mod_type]
# which returns a 1D array of shape (num_detection_bins,) = (num_det_els_in_module * num_energy_bins,)
all_detection_bin_effs: list[petsird.DetectionBinEfficiencies] | None = (
    scanner_info.detection_efficiencies.detection_bin_efficiencies
)

# get the symmetry group ID LUTs for all module types
# index via: 2D_SGID_LUT = all_module_pair_sgidlut[rep_mod_type 1][rep_mod_type 2]
# which returns a 2D array of shape (num_modules, num_modules)
all_module_pair_sgidluts: list[list[petsird.ModulePairSGIDLUT]] | None = (
    scanner_info.detection_efficiencies.module_pair_sgidlut
)

# get all module pair efficiencies vectors
# index via: mod_pair_effs = module_pair_efficiencies_vectors[rep_mod_type 1][rep_mod_type 2][sgid]
# which returns a 2D array of shape (num_det_els, num_det_els)
all_module_pair_efficiency_vectors: (
    list[list[list[petsird.ModulePairEfficiencies]]] | None
) = scanner_info.detection_efficiencies.module_pair_efficiencies_vectors

if all_detection_bin_effs is None:
    raise ValueError(
        "No detection bin efficiencies found in scanner information. "
        "Please check the scanner geometry and detection efficiencies."
    )

if all_module_pair_sgidluts is None:
    raise ValueError(
        "No module pair SGID LUTs found in scanner information. "
        "Please check the scanner geometry and detection efficiencies."
    )

if all_module_pair_efficiency_vectors is None:
    raise ValueError(
        "No module pair efficiencies vectors found in scanner information. "
        "Please check the scanner geometry and detection efficiencies."
    )

# %%

print("Generating sensitivity image")

for mod_type_1 in range(num_replicated_modules):
    num_modules_1 = len(scanner_geom.replicated_modules[mod_type_1].transforms)

    energy_bin_edges_1 = scanner_info.event_energy_bin_edges[mod_type_1].edges
    num_energy_bins_1 = energy_bin_edges_1.size - 1

    det_bin_effs_1 = all_detection_bin_effs[mod_type_1].reshape(
        num_modules_1, -1, num_energy_bins_1
    )

    for mod_type_2 in range(num_replicated_modules):
        num_modules_2 = len(scanner_geom.replicated_modules[mod_type_2].transforms)

        energy_bin_edges_2 = scanner_info.event_energy_bin_edges[mod_type_2].edges
        num_energy_bins_2 = energy_bin_edges_2.size - 1
        det_bin_effs_2 = all_detection_bin_effs[mod_type_2].reshape(
            num_modules_2, -1, num_energy_bins_2
        )

        print(
            f"Module type {mod_type_1} with {num_modules_1} modules vs. {mod_type_2} and {num_modules_2} modules"
        )

        sgid_lut = all_module_pair_sgidluts[mod_type_1][mod_type_2]

        # sigma TOF (mm) for module type combination
        sigma_tof = scanner_info.tof_resolution[mod_type_1][mod_type_2] / 2.35
        tof_bin_edges = scanner_info.tof_bin_edges[mod_type_1][mod_type_2].edges

        # raise an error if tof_bin_edges are non equidistant (up to 0.1%)
        if not np.allclose(
            np.diff(tof_bin_edges), tof_bin_edges[1] - tof_bin_edges[0], rtol=0.001
        ):
            raise ValueError(
                f"TOF bin edges for module types {mod_type_1} and {mod_type_2} are not equidistant."
            )

        num_tofbins = tof_bin_edges.size - 1
        tofbin_width = float(tof_bin_edges[1] - tof_bin_edges[0])

        for i_mod_1 in range(num_modules_1):
            for i_mod_2 in range(num_modules_2):

                sgid = all_module_pair_sgidluts[mod_type_1][mod_type_2][
                    i_mod_1, i_mod_2
                ]

                # if the symmetry group ID (sgid) is non-negative, the module pair is in coincidence
                if sgid >= 0:
                    print(
                        f"Module pair ({mod_type_1}, {i_mod_1}) vs. ({mod_type_2}, {i_mod_2}) with SGID {sgid}"
                    )
                    start_coords = all_detector_centers[mod_type_1][i_mod_1, :, :]
                    end_coords = all_detector_centers[mod_type_2][i_mod_2, :, :]

                    # setup a LM projector that we use for the sensitivity image calculation
                    proj = parallelproj.ListmodePETProjector(
                        start_coords, end_coords, img_shape, voxel_size
                    )

                    proj.tof_parameters = parallelproj.TOFParameters(
                        num_tofbins=num_tofbins,
                        tofbin_width=tofbin_width,
                        sigma_tof=sigma_tof,
                    )

                    module_pair_efficiencies = all_module_pair_efficiency_vectors[
                        mod_type_1
                    ][mod_type_2][sgid].values

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

    fig.show()
