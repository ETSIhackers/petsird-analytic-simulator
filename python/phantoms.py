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
def _letter_coverage(
    letter: str,
    x: np.ndarray,
    y: np.ndarray,
    height_mm: float,
    up: str,
    supersampling: int,
    voxel_size: tuple[float, float],
) -> np.ndarray:
    """fraction (0 ... 1) of the transaxial voxels (centers x, y) covered by a letter

    The letter (DejaVu Sans Bold, shipped with matplotlib) is scaled to a cap height of
    height_mm, centered at (0, 0), reads along +x and its top points to +y (up="+y") or
    -y (up="-y").
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath

    path = TextPath((0, 0), letter, size=1.0, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    verts = path.vertices
    (x0, y0), (x1, y1) = verts.min(0), verts.max(0)
    scale = height_mm / (y1 - y0)
    # sub-voxel sample points (in letter coordinates)
    offsets = (np.arange(supersampling) + 0.5) / supersampling - 0.5
    dx, dy = np.meshgrid(offsets * voxel_size[0], offsets * voxel_size[1], indexing="ij")
    px = x[..., None] + dx.ravel()
    py = y[..., None] + dy.ravel()
    if up == "-y":
        py = -py
    elif up != "+y":
        raise ValueError(f"up has to be '+y' or '-y', got {up}")
    letter_x = px / scale + 0.5 * (x0 + x1)
    letter_y = py / scale + 0.5 * (y0 + y1)
    # even-odd rule over the outlines of the glyph, such that the counters (holes, e.g.
    # of O, P, R) are empty (Path.contains_points of the compound path fills them)
    points = np.stack([letter_x.ravel(), letter_y.ravel()], axis=1)
    num_outlines = sum(
        Path(polygon).contains_points(points).astype(int) for polygon in path.to_polygons()
    )
    inside = num_outlines % 2 == 1
    return inside.reshape(px.shape).mean(-1)


def default_letter_z_mm(img_shape, voxel_size, num_letters: int) -> tuple[float, ...]:
    """centers of num_letters equally spaced axial sections of the image (mm)"""
    extent = img_shape[2] * voxel_size[2]
    return tuple((-0.5 + (i + 0.5) / num_letters) * extent for i in range(num_letters))


def add_letters(
    img: np.ndarray,
    mask: np.ndarray,
    voxel_size,
    text: str = "PET",
    contrast: float = 0.15,
    letter_z_mm: tuple[float, ...] | None = None,
    letter_height_mm: float = 40.0,
    letter_thickness_mm: float = 6.0,
    up: str = "-y",
) -> np.ndarray:
    """add hidden low-contrast letters to an image (within mask)

    Letter n of text is a "hot" transaxial letter (activity increased by contrast, cap
    height letter_height_mm, centered at x = y = 0, partial volume from 4 x 4 sub-voxel
    samples) in the slices within letter_thickness_mm / 2 of z = letter_z_mm[n]. The
    letters read along +x with their top towards `up` ("-y": upright in viewer.py, which
    shows x to the right and y downwards), the first letter at the lowest z.
    Default letter_z_mm: centers of len(text) equally spaced axial sections of the image
    (for the default scanner and image, approximately the centers of the 3 axial module
    positions).
    """
    img_shape = img.shape
    X, Y, Z = voxel_center_coordinates(img_shape, voxel_size)
    if letter_z_mm is None:
        letter_z_mm = default_letter_z_mm(img_shape, voxel_size, len(text))
    if len(letter_z_mm) != len(text):
        raise ValueError(f"{len(text)} letters but {len(letter_z_mm)} z positions")
    img = img.copy()
    for letter, zc in zip(text, letter_z_mm):
        if letter == " ":
            continue
        coverage = _letter_coverage(
            letter, X[..., 0], Y[..., 0], letter_height_mm, up, 4, (voxel_size[0], voxel_size[1])
        )
        slices = np.abs(Z[0, 0, :] - zc) < 0.5 * letter_thickness_mm
        img[:, :, slices] += contrast * coverage[..., None] * mask[:, :, slices]
    return img


def uniform_cylinder_text(
    img_shape, voxel_size, text: str = "PET", contrast: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """uniform cylinder (uniform_cylinder) with hidden low-contrast letters (add_letters)

    The contrast is low enough that the letters are hidden by the artifacts of
    reconstructions with ignored or wrongly decoded detection efficiencies, but readable
    in reconstructions with the correct ones. Mirrored letters or a wrong letter order
    reveal flipped axes or axial positions.
    """
    img, water = uniform_cylinder(img_shape, voxel_size)
    return add_letters(img, img > 0, voxel_size, text=text, contrast=contrast), water


PHANTOMS: dict[str, Callable] = {
    "uniform_cylinder": uniform_cylinder,
    "offcenter_cylinder": offcenter_cylinder,
    "points": points,
    "uniform_cylinder_text": uniform_cylinder_text,
}


def create_phantom(
    name: str, img_shape: tuple[int, int, int], voxel_size: tuple[float, float, float], **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """activity image and attenuation image (1/mm) of a phantom (kwargs: phantom options,
    e.g. text / contrast of uniform_cylinder_text)"""
    if name not in PHANTOMS:
        raise ValueError(f"Invalid phantom {name}, available: {list(PHANTOMS)}")
    activity, water_body = PHANTOMS[name](img_shape, voxel_size, **kwargs)
    attenuation = np.where(water_body, MU_WATER, 0).astype(np.float32)
    return activity, attenuation
