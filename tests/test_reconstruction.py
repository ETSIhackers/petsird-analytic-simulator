import numpy as np
from parallelproj.operators import GaussianFilterOperator
from parallelproj.projectors import EqualBlockPETProjector

from conftest import TINY_IMG_SHAPE, TINY_VOXEL_SIZE
from efficiencies import lor_efficiencies
from petsird_reader import get_all_detector_centers
from reconstruction import (
    create_lm_subset_objectives,
    default_image_grid,
    histogram_mlem,
    lm_osem,
    sensitivity_image,
)
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
    objectives = create_lm_subset_objectives(
        coords0,
        coords1,
        np.zeros(4, dtype="int16"),
        np.ones(4, dtype="float32"),
        img_shape,
        voxel_size,
        img_origin,
        GaussianFilterOperator(img_shape, sigma=0.5),
        sens,
        num_subsets=2,
        eps=1e-6,
    )
    recon = lm_osem(objectives, sens, num_epochs=2)
    assert np.all(np.isfinite(recon))
    assert np.all(recon[0] == 0)


def test_lm_objectives_use_eps_only_without_positive_contamination():
    img_shape, voxel_size = (8, 8, 2), (2.0, 2.0, 2.0)
    img_origin = -(np.asarray(img_shape) - 1) / 2 * np.asarray(voxel_size)
    coords0 = np.array([[-20.0, 1.0, 0.0]] * 4, dtype="float32")
    coords1 = np.array([[20.0, 1.0, 0.0]] * 4, dtype="float32")
    args = (
        coords0,
        coords1,
        np.zeros(4, dtype="int16"),
        np.ones(4, dtype="float32"),
        img_shape,
        voxel_size,
        img_origin,
        GaussianFilterOperator(img_shape, sigma=0.5),
        np.ones(img_shape, dtype="float32"),
        2,
        1e-3,
    )
    assert all(o.eps == 1e-3 for o in create_lm_subset_objectives(*args))
    with_contamination = create_lm_subset_objectives(
        *args, contamination_list=np.full(4, 0.1, dtype="float32"), contamination_sum=1.0
    )
    assert all(o.eps == 0.0 for o in with_contamination)


def test_histogram_mlem_converges_for_consistent_data():
    """MLEM of noise-free data of a 2x2 matrix operator converges to the true image"""
    from parallelproj.operators import MatrixOperator

    A = MatrixOperator(np.array([[1.0, 0.5], [0.2, 1.0], [0.3, 0.3]], dtype="float32"))
    x_true = np.array([2.0, 5.0], dtype="float32")
    sens = A.adjoint(np.ones(3, dtype="float32"))
    recon = histogram_mlem(A, A(x_true), sens, num_epochs=500)
    np.testing.assert_allclose(recon, x_true, rtol=1e-2)


def test_constant_randoms_contamination_units():
    """randoms per event refer to a TOF bin (TOF) or the whole LOR (non-TOF)"""
    from reconstruction import constant_randoms_contamination

    tof = constant_randoms_contamination(0.01, num_events=3, num_tofbins=7, tof=True)
    non_tof = constant_randoms_contamination(0.01, num_events=3, num_tofbins=7, tof=False)
    np.testing.assert_allclose(tof, 0.01)
    # the randoms of a LOR are the sum over its TOF bins
    np.testing.assert_allclose(non_tof, 7 * tof)
