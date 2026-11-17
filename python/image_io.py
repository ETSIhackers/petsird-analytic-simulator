"""images as NIfTI-1 files with their grid in the PETSIRD scanner coordinate system

All images (ground truth, attenuation, sensitivity, back projections, reconstructions,
histo-images) are stored as NIfTI-1 files (.nii, float32). The affine of the file maps
voxel indices (i, j, k) of the array img[i, j, k] to the coordinates (in mm) of the voxel
center in the scanner coordinate system of the PETSIRD file (the coordinate system of
the detecting elements / ADPs):

    (x, y, z, 1) = affine @ (i, j, k, 1)

For the images of this repository, the affine is diagonal (axis-aligned grids):
affine = [[vx, 0, 0, ox], [0, vy, 0, oy], [0, 0, vz, oz], [0, 0, 0, 1]] with the voxel
size (vx, vy, vz) and the center (ox, oy, oz) of voxel [0, 0, 0] (img_origin).

The qform / sform code is 1 ("scanner-based anatomical coordinates"). Note that the
PETSIRD scanner coordinate system is not a patient coordinate system: the orientation
labels (L/R, A/P, S/I) that NIfTI viewers show for the axes have no meaning here.
The units of the image values are not stored (see the README for the conventions).
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from parallelproj import to_numpy_array

# qform / sform code "scanner-based anatomical coordinates"
NIFTI_XFORM_SCANNER_ANAT = 1


@dataclass(frozen=True)
class ImageGrid:
    """axis-aligned image grid: shape, voxel size (mm) and center of voxel [0, 0, 0] (mm)"""

    shape: tuple[int, int, int]
    voxel_size: tuple[float, float, float]
    origin: tuple[float, float, float]

    @property
    def affine(self) -> np.ndarray:
        """4 x 4 affine from voxel indices to scanner coordinates (mm)"""
        a = np.eye(4)
        a[:3, :3] = np.diag(self.voxel_size)
        a[:3, 3] = self.origin
        return a

    def matches(self, other: "ImageGrid", atol: float = 1e-3) -> bool:
        """same shape, voxel size and origin (within atol mm)"""
        return (
            tuple(self.shape) == tuple(other.shape)
            and np.allclose(self.voxel_size, other.voxel_size, atol=atol)
            and np.allclose(self.origin, other.origin, atol=atol)
        )

    def __str__(self) -> str:
        return (
            f"shape {tuple(self.shape)}, voxel size {tuple(float(v) for v in self.voxel_size)} mm, "
            f"center of voxel [0,0,0] {tuple(round(float(o), 3) for o in self.origin)} mm"
        )


def centered_grid(shape, voxel_size) -> ImageGrid:
    """grid centered at the origin of the scanner coordinate system (as the default grid
    of parallelproj projectors without img_origin)"""
    shape = tuple(int(s) for s in shape)
    voxel_size = tuple(float(v) for v in voxel_size)
    origin = tuple(float(o) for o in -0.5 * (np.asarray(shape) - 1) * np.asarray(voxel_size))
    return ImageGrid(shape, voxel_size, origin)


def save_image(path: str | Path, img, grid: ImageGrid, description: str = "") -> Path:
    """save an image (any array type, converted to a float32 NumPy array) as NIfTI-1 file"""
    data = np.asarray(to_numpy_array(img), dtype=np.float32)
    if tuple(data.shape) != tuple(grid.shape):
        raise ValueError(f"image shape {data.shape} does not match the grid {grid}")
    nii = nib.Nifti1Image(data, grid.affine)
    nii.set_qform(grid.affine, code=NIFTI_XFORM_SCANNER_ANAT)
    nii.set_sform(grid.affine, code=NIFTI_XFORM_SCANNER_ANAT)
    nii.header.set_xyzt_units(xyz="mm")
    # the description field holds up to 79 characters
    nii.header["descrip"] = description[:79].encode()
    path = Path(path)
    nib.save(nii, path)
    return path


def load_image(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """load a NIfTI image: (float32 array img[i, j, k], 4 x 4 affine)"""
    nii = nib.load(str(path))
    # C order (NIfTI stores the first index fastest, i.e. nibabel returns Fortran order)
    data = np.ascontiguousarray(nii.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D image, got shape {data.shape}")
    return data, np.asarray(nii.affine, dtype=np.float64)


def grid_from_affine(shape, affine: np.ndarray) -> ImageGrid:
    """axis-aligned grid of an affine (ValueError for rotated / sheared grids)"""
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-6) or np.any(np.diag(linear) <= 0):
        raise ValueError(
            "only axis-aligned image grids with positive voxel sizes are supported as "
            f"image grids (affine:\n{affine})"
        )
    return ImageGrid(
        tuple(int(s) for s in shape),
        tuple(float(v) for v in np.diag(linear)),
        tuple(float(o) for o in affine[:3, 3]),
    )


def resample_image(img: np.ndarray, affine: np.ndarray, grid: ImageGrid) -> np.ndarray:
    """trilinear interpolation of an image (any affine) at the voxel centers of a grid

    Points outside of the image (more than half a voxel beyond the outer voxel centers)
    are 0; between the outer voxel centers and the image border the outer values are
    used.
    """
    img = np.asarray(img, dtype=np.float32)
    # voxel indices of the source image at the voxel centers of the target grid
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in grid.shape], indexing="ij"), axis=-1)
    target_to_source = np.linalg.solve(affine, grid.affine)
    src = idx.reshape(-1, 3) @ target_to_source[:3, :3].T + target_to_source[:3, 3]

    shape = np.asarray(img.shape)
    inside = np.all((src >= -0.5) & (src <= shape - 0.5), axis=1)
    src = np.clip(src, 0, shape - 1)
    i0 = np.minimum(np.floor(src).astype(int), shape - 2).clip(min=0)
    frac = src - i0
    result = np.zeros(src.shape[0], dtype=np.float64)
    for corner in np.ndindex(2, 2, 2):
        c = np.asarray(corner)
        ic = np.minimum(i0 + c, shape - 1)
        weight = np.prod(np.where(c == 1, frac, 1 - frac), axis=1)
        result += weight * img[ic[:, 0], ic[:, 1], ic[:, 2]]
    result[~inside] = 0
    return result.reshape(grid.shape).astype(np.float32)


def load_image_on_grid(path: str | Path, grid: ImageGrid, description: str = "image") -> np.ndarray:
    """load a NIfTI image and resample it onto a grid if its grid differs"""
    img, affine = load_image(path)
    try:
        same = grid_from_affine(img.shape, affine).matches(grid)
    except ValueError:
        same = False
    if same:
        return img
    diag = tuple(round(float(v), 4) for v in np.diag(affine)[:3])
    offset = tuple(round(float(v), 3) for v in affine[:3, 3])
    print(
        f"Resampling the {description} {path} (shape {img.shape}, affine diagonal {diag}, "
        f"offset {offset} mm) onto the image grid ({grid}) with trilinear interpolation"
    )
    return resample_image(img, affine, grid)
