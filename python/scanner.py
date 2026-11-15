"""definition of the simulated block PET scanner

The scanner is defined once (ScannerConfig) and converted into
- a parallelproj scanner geometry / LOR descriptor (used for the simulation) and
- a PETSIRD ScannerGeometry (written to the PETSIRD file)
such that both are consistent by construction.
"""

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import petsird
from parallelproj.pet_lors import EqualBlockPETLORDescriptor
from parallelproj.pet_scanners import BlockPETScannerModule, ModularizedPETScannerGeometry


@dataclass(frozen=True)
class ScannerConfig:
    """block modules (cuboids of detecting elements) arranged on a circle"""

    # number of block modules on the circle
    num_modules: int = 12
    # distance from the scanner center to the center of the block modules (mm)
    radius: float = 100.0
    # number of detecting elements per module in the 3 module directions
    # (tangential, radial, axial)
    block_shape: tuple[int, int, int] = (10, 2, 9)
    # spacing between the detecting elements = size of the elements (mm)
    block_spacing: tuple[float, float, float] = (4.5, 10.0, 4.5)
    # module pairs with a circular distance <= this value are not in coincidence
    min_coincidence_distance: int = 3

    @property
    def num_det_els_per_module(self) -> int:
        return math.prod(self.block_shape)


def circular_distance(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    """number of modules between two modules on the circle (shortest way)"""
    clockwise_distance = abs(i_mod_1 - i_mod_2)
    return min(clockwise_distance, num_modules - clockwise_distance)


def module_transforms(config: ScannerConfig) -> list[np.ndarray]:
    """4x4 affine transformations from the module to the scanner coordinate system

    The modules are translated by the radius along the y-axis and then rotated around
    the z-axis (scanner axis).
    """
    translation = np.eye(4, dtype="float32")
    translation[1, -1] = config.radius

    transforms = []
    for phi in np.linspace(0, 2 * np.pi, config.num_modules, endpoint=False):
        rotation = np.asarray(
            [
                [math.cos(phi), -math.sin(phi), 0, 0],
                [math.sin(phi), math.cos(phi), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype="float32",
        )
        transforms.append(rotation @ translation)
    return transforms


def coincident_module_pairs(config: ScannerConfig) -> list[tuple[int, int]]:
    """all module pairs (i, j) with i < j that are in coincidence"""
    return [
        (i, j)
        for i, j in combinations(range(config.num_modules), 2)
        if circular_distance(i, j, config.num_modules) >= config.min_coincidence_distance
    ]


def create_parallelproj_lor_descriptor(
    config: ScannerConfig,
) -> EqualBlockPETLORDescriptor:
    """parallelproj LOR descriptor connecting all detecting elements of modules in coincidence

    The LOR end points are the centers of the detecting elements (= PETSIRD ADPs).
    """
    modules = [
        BlockPETScannerModule(
            np, "cpu", config.block_shape, config.block_spacing, affine_transformation_matrix=t
        )
        for t in module_transforms(config)
    ]
    scanner = ModularizedPETScannerGeometry(tuple(modules))
    return EqualBlockPETLORDescriptor(scanner, np.asarray(coincident_module_pairs(config)))


def create_petsird_scanner_geometry(config: ScannerConfig) -> petsird.ScannerGeometry:
    """PETSIRD scanner geometry (one module type)

    The hierarchy is
      ScannerGeometry(list[ReplicatedDetectorModule])
      ReplicatedDetectorModule(list[RigidTransformation], DetectorModule)
      DetectorModule(ReplicatedBoxSolidVolume)
      ReplicatedBoxSolidVolume(list[RigidTransformation], BoxSolidVolume)

    PETSIRD (>= 0.13) requires that the box of a detecting element is oriented along
    its local x-axis ("box axis", pointing away from the scanner center), that (0,0,0)
    is the average detection point (ADP), and a fixed corner order (see petsird.BoxShape).
    In the module frame, the radial (depth) direction is the y-axis (index 1 of
    block_shape). We therefore define the box with its depth along the local x-axis
    and rotate it into the module frame via the detecting element transforms.
    The ADP is placed at the center of the detecting element (= the LOR end points of
    the parallelproj scanner).
    """
    # centers of the detecting elements in the module frame
    # (same order as the LOR end points of the parallelproj modules)
    det_el_centers = BlockPETScannerModule(
        np, "cpu", config.block_shape, config.block_spacing
    ).lor_endpoints

    # box size along the local box axes (depth, width, axial)
    spacing = config.block_spacing
    box_size = np.array((spacing[1], spacing[0], spacing[2]), dtype="float32")
    L0, L1, L2 = box_size
    box_shape = petsird.BoxShape(
        corners=[
            petsird.Coordinate(c=np.array(c, dtype="float32") - 0.5 * box_size)
            for c in (
                (0, 0, 0),
                (0, 0, L2),
                (0, L1, L2),
                (0, L1, 0),
                (L0, 0, 0),
                (L0, 0, L2),
                (L0, L1, L2),
                (L0, L1, 0),
            )
        ]
    )
    rep_volume = petsird.ReplicatedBoxSolidVolume(
        object=petsird.BoxSolidVolume(shape=box_shape, material_id=1)
    )

    # rotation that maps the local box axis (x) onto the radial direction of the module (y)
    box_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="float32")
    for center in det_el_centers:
        matrix = np.zeros((3, 4), dtype="float32")
        matrix[:, :3] = box_rotation
        matrix[:, 3] = center
        rep_volume.transforms.append(petsird.RigidTransformation(matrix=matrix))

    rep_module = petsird.ReplicatedDetectorModule(
        object=petsird.DetectorModule(detecting_elements=rep_volume)
    )
    for t in module_transforms(config):
        rep_module.transforms.append(petsird.RigidTransformation(matrix=t[:-1, :]))

    return petsird.ScannerGeometry(replicated_modules=[rep_module])
