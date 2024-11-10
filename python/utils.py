import petsird
import numpy.typing as npt
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_int_tuple(arg):
    return tuple(map(int, arg.split(",")))


def parse_float_tuple(arg):
    return tuple(map(float, arg.split(",")))


def transform_to_mat44(
    transform: petsird.RigidTransformation,
) -> npt.NDArray[np.float32]:
    return np.vstack([transform.matrix, [0, 0, 0, 1]])


def mat44_to_transform(mat: npt.NDArray[np.float32]) -> petsird.RigidTransformation:
    return petsird.RigidTransformation(matrix=mat[0:3, :])


def coordinate_to_homogeneous(coord: petsird.Coordinate) -> npt.NDArray[np.float32]:
    return np.hstack([coord.c, 1])


def homogeneous_to_coordinate(
    hom_coord: npt.NDArray[np.float32],
) -> petsird.Coordinate:
    return petsird.Coordinate(c=hom_coord[0:3])


def mult_transforms(
    transforms: list[petsird.RigidTransformation],
) -> petsird.RigidTransformation:
    """multiply rigid transformations"""
    mat = np.array(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype="float32",
    )

    for t in reversed(transforms):
        mat = np.matmul(transform_to_mat44(t), mat)
    return mat44_to_transform(mat)


def mult_transforms_coord(
    transforms: list[petsird.RigidTransformation], coord: petsird.Coordinate
) -> petsird.Coordinate:
    """apply list of transformations to coordinate"""
    # TODO better to multiply with coordinates in sequence, as first multiplying the matrices
    hom = np.matmul(
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
    vertices = np.array([c.c for c in box.corners])
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
