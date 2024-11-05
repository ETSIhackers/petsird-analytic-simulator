#  Copyright (C) 2024 University College London
#
#  SPDX-License-Identifier: Apache-2.0

# basic plotting of the scanner geometry
# preliminary code!

import numpy
import numpy.typing as npt
import petsird

from petsird_helpers import (
    get_module_and_element,
    get_detection_efficiency,
)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pathlib import Path
import argparse


def transform_to_mat44(
    transform: petsird.RigidTransformation,
) -> npt.NDArray[numpy.float32]:
    return numpy.vstack([transform.matrix, [0, 0, 0, 1]])


def mat44_to_transform(mat: npt.NDArray[numpy.float32]) -> petsird.RigidTransformation:
    return petsird.RigidTransformation(matrix=mat[0:3, :])


def coordinate_to_homogeneous(coord: petsird.Coordinate) -> npt.NDArray[numpy.float32]:
    return numpy.hstack([coord.c, 1])


def homogeneous_to_coordinate(
    hom_coord: npt.NDArray[numpy.float32],
) -> petsird.Coordinate:
    return petsird.Coordinate(c=hom_coord[0:3])


def mult_transforms(
    transforms: list[petsird.RigidTransformation],
) -> petsird.RigidTransformation:
    """multiply rigid transformations"""
    mat = numpy.array(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype="float32",
    )

    for t in reversed(transforms):
        mat = numpy.matmul(transform_to_mat44(t), mat)
    return mat44_to_transform(mat)


def mult_transforms_coord(
    transforms: list[petsird.RigidTransformation], coord: petsird.Coordinate
) -> petsird.Coordinate:
    """apply list of transformations to coordinate"""
    # TODO better to multiply with coordinates in sequence, as first multiplying the matrices
    hom = numpy.matmul(
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


def draw_BoxShape(ax, box: petsird.BoxShape) -> None:
    vertices = numpy.array([c.c for c in box.corners])
    edges = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [4, 7, 3, 0]],
    ]
    box = Poly3DCollection(edges, alpha=0.1, linewidths=0.1, edgecolors=plt.cm.tab10(0))
    ax.add_collection3d(box)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", default="test.bin")

    args = parser.parse_args()
    fname = args.fname

    if not Path(fname).exists():
        raise FileNotFoundError(
            f"{args.fname} not found. Create it first using the generator."
        )

    # Create a new figure
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")

    # dictionary to store the transformations and centers of the detecting elements
    # here we assume that we only have BoxShapes
    element_transforms = dict()
    element_centers = dict()

    # with petsird.BinaryPETSIRDReader(sys.stdin.buffer) as reader:
    with petsird.BinaryPETSIRDReader(fname) as reader:
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

                        transformed_boxshape_vertices = numpy.array(
                            [c.c for c in transformed_boxshape.corners]
                        )

                        draw_BoxShape(ax, transformed_boxshape)

                        element_transforms[(i_mod, i_el)] = combined_transform

                        element_centers[(i_mod, i_el)] = (
                            transformed_boxshape_vertices.mean(axis=0)
                        )

                        if i_el == 0 or i_el == len(rep_volume.transforms) - 1:
                            ax.text(
                                float(transformed_boxshape_vertices[0][0]),
                                float(transformed_boxshape_vertices[0][1]),
                                float(transformed_boxshape_vertices[0][2]),
                                f"{i_el:02}/{i_mod:02}",
                                fontsize=7,
                            )

        # ----
        # read and draw events

        num_prompts = 0
        event_counter = 0

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
                    event_end_coord = element_centers[
                        event_mods_and_els[1].module, event_mods_and_els[1].el
                    ]

                    # get the event efficiencies
                    event_eff = get_detection_efficiency(header.scanner, event)

                    # draw line between the two 3D points (event_start_coord, event_end_coord)
                    # for the first event in the first time block
                    if i_event < 3:
                        ax.plot(
                            [event_start_coord[0], event_end_coord[0]],
                            [event_start_coord[1], event_end_coord[1]],
                            [event_start_coord[2], event_end_coord[2]],
                        )

                        print(
                            f"time block {i_time_block:04}, event in time block {i_event:04}, event {event_counter:04}, {event_mods_and_els}"
                        )

                        print(
                            "start world coordinates",
                            event_start_coord[0],
                            event_start_coord[1],
                            event_start_coord[2],
                        )
                        print(
                            "end world coordinates",
                            event_end_coord[0],
                            event_end_coord[1],
                            event_end_coord[2],
                        )
                        print(f"event eff {event_eff}")
                        print()

                    event_counter += 1

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    # ax.set_title("figure not in scale (z axis is streched)")

    plt.show()
