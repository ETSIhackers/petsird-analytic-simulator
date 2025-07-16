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
    box = Poly3DCollection(edges, alpha=0.1, linewidths=0.25, edgecolors="r")
    ax.add_collection3d(box)


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

reader = petsird.BinaryPETSIRDReader(fname)
header: petsird.Header = reader.read_header()
scanner_info: petsird.ScannerInformation = header.scanner
scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# get all detector centers
all_detector_centers = get_all_detector_centers(scanner_geom, ax=ax)

# set axis limits

min_coords = all_detector_centers[0].reshape(-1, 3).min(0)
max_coords = all_detector_centers[0].reshape(-1, 3).max(0)

ax.set_xlim3d([min_coords.min(), max_coords.max()])
ax.set_ylim3d([min_coords.min(), max_coords.max()])
ax.set_zlim3d([min_coords.min(), max_coords.max()])

for i_d, detector_centers in enumerate(all_detector_centers):
    ax.scatter(
        detector_centers[:, :, 0].ravel(),
        detector_centers[:, :, 1].ravel(),
        detector_centers[:, :, 2].ravel(),
        s=1,
        c=plt.cm.tab10(i_d),
    )

fig.show()
