"""phantoms (activity and attenuation images) for the simulation

All images are centered at (0,0,0), i.e. the center of voxel (i,j,k) is at
((i - (n0 - 1) / 2) * v0, (j - (n1 - 1) / 2) * v1, (k - (n2 - 1) / 2) * v2).
Every phantom consists of an activity image and an attenuating water body.
"""

from collections.abc import Callable

import numpy as np

# linear attenuation coefficient of water at 511 keV (1/mm)
MU_WATER = 0.0096


def voxel_center_coordinates(
    img_shape: tuple[int, int, int], voxel_size: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, y, z coordinates (mm) of all voxel centers"""
    return np.meshgrid(
        *[(np.arange(n) - (n - 1) / 2) * vs for n, vs in zip(img_shape, voxel_size)],
        indexing="ij",
    )


def _centered_disk(img_shape: tuple[int, int, int]) -> np.ndarray:
    """transaxial disk with a radius of 70% of the image half-width"""
    tmp = np.linspace(-1, 1, img_shape[0])
    X0, X1 = np.meshgrid(tmp, tmp, indexing="ij")
    return np.astype(np.sqrt(X0**2 + X1**2) < 0.7, "float32")


def _centered_water_cylinder(img_shape: tuple[int, int, int]) -> np.ndarray:
    """centered water cylinder covering all slices (boolean mask)"""
    return np.broadcast_to(_centered_disk(img_shape)[..., None].astype(bool), img_shape)


def uniform_cylinder(img_shape, voxel_size) -> tuple[np.ndarray, np.ndarray]:
    """uniform cylinder centered in the field of view"""
    img = np.zeros(img_shape, dtype=np.float32)
    disk = _centered_disk(img_shape)
    for i in range(2, img_shape[2] - 2):
        img[..., i] = disk
    return img, _centered_water_cylinder(img_shape)


def offcenter_cylinder(img_shape, voxel_size) -> tuple[np.ndarray, np.ndarray]:
    """off-center (transaxially and axially) cylinder with 3 hot spheres

    The spheres have different size, contrast and position, such that axis flips,
    swaps and mirroring (e.g. in readers of PETSIRD data) are visible in reconstructions.
    The water body is the cylinder itself.
    """
    X, Y, Z = voxel_center_coordinates(img_shape, voxel_size)
    img = np.zeros(img_shape, dtype=np.float32)
    cylinder = (np.hypot(X - 10, Y + 5) <= 30) & (Z >= -12) & (Z <= 16)
    img[cylinder] = 1
    # (center x, y, z in mm, radius in mm, activity)
    for (cx, cy, cz), radius, activity in [
        ((22, -5, 6), 8, 3),
        ((5, 12, -4), 6, 2),
        ((0, -18, 10), 4, 3),
    ]:
        img[np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= radius] = activity
    return img, cylinder


def points(img_shape, voxel_size) -> tuple[np.ndarray, np.ndarray]:
    """point sources in a centered water cylinder"""
    img = np.zeros(img_shape, dtype=np.float32)
    n0, n1, n2 = img_shape
    for k in (n2 // 2, n2 // 6):
        img[n0 // 2, n1 // 2, k] = 8
        img[n0 // 2, n1 // 6, k] = 4
        img[n0 // 4, n1 // 2, k] = 6
    return img, _centered_water_cylinder(img_shape)


# all available phantoms: name -> function(img_shape, voxel_size) -> (activity, water body)
PHANTOMS: dict[str, Callable] = {
    "uniform_cylinder": uniform_cylinder,
    "offcenter_cylinder": offcenter_cylinder,
    "points": points,
}


def create_phantom(
    name: str, img_shape: tuple[int, int, int], voxel_size: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """activity image and attenuation image (1/mm) of a phantom"""
    if name not in PHANTOMS:
        raise ValueError(f"Invalid phantom {name}, available: {list(PHANTOMS)}")
    activity, water_body = PHANTOMS[name](img_shape, voxel_size)
    attenuation = np.where(water_body, MU_WATER, 0).astype(np.float32)
    return activity, attenuation
