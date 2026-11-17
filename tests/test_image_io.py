import nibabel as nib
import numpy as np
import pytest

from image_io import (
    ImageGrid,
    centered_grid,
    grid_from_affine,
    load_image,
    load_image_on_grid,
    resample_image,
    save_image,
)


def test_centered_grid():
    grid = centered_grid((5, 4, 3), (2.0, 1.0, 3.0))
    np.testing.assert_allclose(grid.origin, (-4.0, -1.5, -3.0))
    # the center of the grid is at the scanner origin
    center = grid.affine @ np.array([2.0, 1.5, 1.0, 1.0])
    np.testing.assert_allclose(center[:3], 0)


def test_save_and_load_roundtrip(tmp_path):
    grid = ImageGrid((5, 4, 3), (2.0, 1.5, 3.0), (-10.0, 3.0, 7.5))
    img = np.random.default_rng(0).random(grid.shape).astype(np.float32)
    path = save_image(tmp_path / "img.nii", img, grid, "test image")
    data, affine = load_image(path)
    np.testing.assert_array_equal(data, img)
    np.testing.assert_allclose(affine, grid.affine)
    assert grid_from_affine(data.shape, affine).matches(grid)

    # the NIfTI header: scanner coordinates (code 1) in mm, voxel (i, j, k) = img[i, j, k]
    nii = nib.load(path)
    assert int(nii.header["qform_code"]) == 1 and int(nii.header["sform_code"]) == 1
    assert nii.header.get_xyzt_units()[0] == "mm"
    np.testing.assert_allclose(nii.header.get_zooms(), grid.voxel_size)
    assert nii.get_fdata()[4, 1, 2] == pytest.approx(img[4, 1, 2])
    assert nii.header["descrip"].tobytes().startswith(b"test image")

    with pytest.raises(ValueError):
        save_image(tmp_path / "wrong.nii", img[:-1], grid)


def test_grid_from_affine_rejects_rotations():
    a = np.eye(4)
    a[:2, :2] = [[0, -1], [1, 0]]
    with pytest.raises(ValueError):
        grid_from_affine((2, 2, 2), a)


def _linear(points):
    return 1.0 + 0.5 * points[..., 0] - 0.25 * points[..., 1] + 0.1 * points[..., 2]


def _voxel_centers(grid):
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in grid.shape], indexing="ij"), axis=-1)
    return idx @ grid.affine[:3, :3].T + grid.affine[:3, 3]


def test_resample_identity_and_linear_function():
    src = ImageGrid((20, 18, 10), (2.0, 2.0, 3.0), (-19.0, -17.0, -13.5))
    img = _linear(_voxel_centers(src)).astype(np.float32)
    np.testing.assert_array_equal(resample_image(img, src.affine, src), img)

    # trilinear interpolation is exact for a linear function (inside of the source grid)
    dst = ImageGrid((11, 9, 7), (2.7, 3.1, 2.2), (-13.0, -12.0, -6.0))
    np.testing.assert_allclose(
        resample_image(img, src.affine, dst), _linear(_voxel_centers(dst)), rtol=1e-5
    )


def test_resample_outside_is_zero():
    src = ImageGrid((4, 4, 4), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    img = np.ones(src.shape, dtype=np.float32)
    dst = ImageGrid((10, 1, 1), (1.0, 1.0, 1.0), (-3.0, 1.0, 1.0))
    out = resample_image(img, src.affine, dst)[:, 0, 0]
    # source voxel centers 0 ... 3, image extent -0.5 ... 3.5
    np.testing.assert_array_equal(out, [0, 0, 0, 1, 1, 1, 1, 0, 0, 0])


def test_load_image_on_grid(tmp_path, capsys):
    grid = ImageGrid((6, 6, 4), (2.0, 2.0, 2.0), (-5.0, -5.0, -3.0))
    img = _linear(_voxel_centers(grid)).astype(np.float32)
    path = save_image(tmp_path / "img.nii", img, grid)
    np.testing.assert_array_equal(load_image_on_grid(path, grid), img)
    assert "Resampling" not in capsys.readouterr().out
    # half the voxel size, shifted by half a (new) voxel: resampled
    fine = ImageGrid((10, 10, 6), (1.0, 1.0, 1.0), (-4.5, -4.5, -2.5))
    out = load_image_on_grid(path, fine, "attenuation image")
    assert "Resampling the attenuation image" in capsys.readouterr().out
    np.testing.assert_allclose(out, _linear(_voxel_centers(fine)), rtol=1e-5)


def test_loaded_images_are_c_contiguous(tmp_path):
    """(parallelproj requires C-contiguous float32 arrays)"""
    grid = ImageGrid((5, 4, 3), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    path = save_image(tmp_path / "img.nii", np.ones(grid.shape), grid)
    data, _ = load_image(path)
    assert data.flags["C_CONTIGUOUS"] and data.dtype == np.float32
