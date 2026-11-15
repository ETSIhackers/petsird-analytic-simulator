import numpy as np
from parallelproj.operators import GaussianFilterOperator
from parallelproj.projectors import EqualBlockPETProjector

from conftest import TINY_IMG_SHAPE, TINY_VOXEL_SIZE
from efficiencies import lor_efficiencies
from petsird_reader import get_all_detector_centers
from reconstruction import default_image_grid, lm_osem, sensitivity_image
from scanner import create_parallelproj_lor_descriptor


def test_sensitivity_image_matches_simulator(tiny_config, tiny_efficiency_model, tiny_header):
    """non-TOF sensitivity image computed from the PETSIRD header == simulator's"""
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    res_model = GaussianFilterOperator(TINY_IMG_SHAPE, sigma=0.5)

    # simulator: back projection of the LOR efficiencies (image centered at 0)
    proj = EqualBlockPETProjector(lor_desc, TINY_IMG_SHAPE, TINY_VOXEL_SIZE)
    ref = res_model.adjoint(proj.adjoint(lor_efficiencies(tiny_efficiency_model, lor_desc)))

    centers = get_all_detector_centers(tiny_header.scanner.scanner_geometry)
    img_shape, img_origin = default_image_grid(centers, TINY_VOXEL_SIZE, TINY_IMG_SHAPE)
    np.testing.assert_allclose(img_origin, proj.img_origin, atol=1e-4)
    sens = sensitivity_image(
        tiny_header.scanner, centers, img_shape, TINY_VOXEL_SIZE, img_origin, res_model, tof=False
    )
    np.testing.assert_allclose(sens, ref, rtol=1e-4, atol=1e-4 * ref.max())


def test_lm_osem_finite_with_zero_sensitivity_and_zero_expectations():
    """voxels with zero sensitivity and events with zero expected counts do not lead to NaN"""
    img_shape, voxel_size = (8, 8, 2), (2.0, 2.0, 2.0)
    img_origin = -(np.asarray(img_shape) - 1) / 2 * np.asarray(voxel_size)
    # all events far away from the image -> LORs do not intersect the image
    coords0 = np.array([[100.0, 100.0, 0.0]] * 4, dtype="float32")
    coords1 = np.array([[100.0, -100.0, 0.0]] * 4, dtype="float32")
    sens = np.ones(img_shape, dtype="float32")
    sens[0] = 0
    recon = lm_osem(
        coords0,
        coords1,
        np.zeros(4, dtype="int16"),
        np.ones(4, dtype="float32"),
        img_shape,
        voxel_size,
        img_origin,
        GaussianFilterOperator(img_shape, sigma=0.5),
        sens,
        num_epochs=2,
        num_subsets=2,
        eps=1e-6,
    )
    assert np.all(np.isfinite(recon))
    assert np.all(recon[0] == 0)
