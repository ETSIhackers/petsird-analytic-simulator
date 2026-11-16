import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from viewer import gaussian_filter_mm, show_vol_cuts  # noqa: E402


def test_gaussian_filter_mm_fwhm_and_sum():
    """the filter has the requested FWHM in mm (anisotropic voxels) and keeps the sum"""
    vol = np.zeros((41, 41, 21), dtype=np.float32)
    vol[20, 20, 10] = 1
    voxel_size = (2.0, 2.0, 3.0)
    out = gaussian_filter_mm(vol, 8.0, voxel_size)
    np.testing.assert_allclose(out.sum(), 1, rtol=1e-4)
    for axis in range(3):
        profile = out.sum(axis=tuple(a for a in range(3) if a != axis))
        x = (np.arange(profile.size) - profile.size // 2) * voxel_size[axis]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt((profile * x**2).sum() / profile.sum())
        assert abs(fwhm - 8.0) < 0.2, (axis, fwhm)
    assert gaussian_filter_mm(vol, 0, voxel_size) is vol


def test_viewer_filter_box():
    """entering a FWHM in the viewer shows the filtered volume, 0 the original one"""
    vol = np.zeros((11, 13, 9), dtype=np.float32)
    vol[5, 6, 4] = 1
    fig, axs, widgets = show_vol_cuts(vol, voxel_size=(2.0, 2.0, 2.0))
    widgets["tb_filter"].set_val("4")
    expected = gaussian_filter_mm(vol, 4.0, (2.0, 2.0, 2.0))
    np.testing.assert_allclose(axs[2].images[0].get_array(), expected[:, :, 4].T, rtol=1e-6)
    widgets["tb_filter"].set_val("0")
    np.testing.assert_allclose(axs[2].images[0].get_array(), vol[:, :, 4].T)
