#  Copyright (C) 2024 University College London
#
#  SPDX-License-Identifier: Apache-2.0

# basic plotting of the scanner geometry
# preliminary code!

import array_api_compat.numpy as xp
import numpy.typing as npt
import petsird

from petsird_helpers import (
    get_module_and_element,
    get_detection_efficiency,
)

from pathlib import Path
import argparse

# %%


def transform_to_mat44(
    transform: petsird.RigidTransformation,
) -> npt.NDArray[xp.float32]:
    return xp.vstack([transform.matrix, [0, 0, 0, 1]])


def mat44_to_transform(mat: npt.NDArray[xp.float32]) -> petsird.RigidTransformation:
    return petsird.RigidTransformation(matrix=mat[0:3, :])


def coordinate_to_homogeneous(coord: petsird.Coordinate) -> npt.NDArray[xp.float32]:
    return xp.hstack([coord.c, 1])


def homogeneous_to_coordinate(
    hom_coord: npt.NDArray[xp.float32],
) -> petsird.Coordinate:
    return petsird.Coordinate(c=hom_coord[0:3])


def mult_transforms(
    transforms: list[petsird.RigidTransformation],
) -> petsird.RigidTransformation:
    """multiply rigid transformations"""
    mat = xp.array(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype="float32",
    )

    for t in reversed(transforms):
        mat = xp.matmul(transform_to_mat44(t), mat)
    return mat44_to_transform(mat)


def mult_transforms_coord(
    transforms: list[petsird.RigidTransformation], coord: petsird.Coordinate
) -> petsird.Coordinate:
    """apply list of transformations to coordinate"""
    # TODO better to multiply with coordinates in sequence, as first multiplying the matrices
    hom = xp.matmul(
        transform_to_mat44(mult_transforms(transforms)),
        coordinate_to_homogeneous(coord),
    )
    return homogeneous_to_coordinate(hom)


def transform_BoxShape(
    transform: petsird.RigidTransformation, box_shape: petsird.BoxShape
) -> petsird.BoxShape:

    return petsird.BoxShape(
        corners=[mult_transforms_coord([transform], c) for c in box_shape.corners]
    )


# %%
# parse the command line

parser = argparse.ArgumentParser()
parser.add_argument("--fname", default="sim_lm.bin")

args = parser.parse_args()
fname = args.fname
dev = "cpu"

if not Path(fname).exists():
    raise FileNotFoundError(
        f"{args.fname} not found. Create it first using the generator."
    )

# %%
# read the scanner geometry

# dictionary to store the transformations and centers of the detecting elements
# here we assume that we only have BoxShapes
element_transforms = dict()
element_centers = dict()

reader = petsird.BinaryPETSIRDReader(fname)
header = reader.read_header()

# draw all crystals
for rep_module in header.scanner.scanner_geometry.replicated_modules:
    det_el = rep_module.object.detecting_elements
    for i_mod, mod_transform in enumerate(rep_module.transforms):
        for rep_volume in det_el:
            for i_el, transform in enumerate(rep_volume.transforms):

                combined_transform = mult_transforms([mod_transform, transform])
                transformed_boxshape = transform_BoxShape(
                    combined_transform, rep_volume.object.shape
                )

                transformed_boxshape_vertices = xp.array(
                    [c.c for c in transformed_boxshape.corners]
                )

                element_transforms[(i_mod, i_el)] = combined_transform

                element_centers[(i_mod, i_el)] = transformed_boxshape_vertices.mean(
                    axis=0
                )

# %%
# read all coincidence events

num_prompts = 0
event_counter = 0
num_tof_bins = header.scanner.number_of_tof_bins()

xstart = []
xend = []
tof_bin = []
effs = []

for i_time_block, time_block in enumerate(reader.read_time_blocks()):
    if isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
        num_prompts += len(time_block.value.prompt_events)

        for i_event, event in enumerate(time_block.value.prompt_events):
            event_mods_and_els = get_module_and_element(
                header.scanner.scanner_geometry, event.detector_ids
            )

            event_start_coord = element_centers[
                event_mods_and_els[0].module, event_mods_and_els[0].el
            ]
            xstart.append(event_start_coord)

            event_end_coord = element_centers[
                event_mods_and_els[1].module, event_mods_and_els[1].el
            ]
            xend.append(event_end_coord)

            # get the event efficiencies
            effs.append(get_detection_efficiency(header.scanner, event))
            # get the signed event TOF bin (0 is the central bin)
            tof_bin.append(event.tof_idx - num_tof_bins // 2)

            event_counter += 1

reader.close()

xstart = xp.asarray(xstart, device=dev)
xend = xp.asarray(xend, device=dev)
effs = xp.asarray(effs, device=dev)
tof_bin = xp.asarray(tof_bin, device=dev)

# %%
