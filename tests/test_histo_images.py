import numpy as np
import pytest

from histo_images import splat, tof_bin_center_positions, tof_bin_centers
from scanner import create_parallelproj_lor_descriptor
from simulation import ListmodeEvents, to_petsird_coincidences

EDGES = np.array([-15.0, -5.0, 5.0, 15.0])  # 3 TOF bins with centers -10, 0, 10 mm


def test_tof_bin_centers():
    np.testing.assert_allclose(tof_bin_centers(EDGES), [-10, 0, 10])


def test_positions_follow_petsird_tof_convention():
    """x = (t1 - t2) c / 2 > 0: emission closer to the second detection bin, and swapping
    the detection bins with a mirrored TOF bin gives the same position"""
    a = np.array([[-100.0, 0.0, 0.0]])
    b = np.array([[100.0, 0.0, 0.0]])
    p = tof_bin_center_positions(a, b, [2], EDGES)
    np.testing.assert_allclose(p, [[10.0, 0.0, 0.0]])
    np.testing.assert_allclose(tof_bin_center_positions(b, a, [0], EDGES), p)
    np.testing.assert_allclose(tof_bin_center_positions(a, b, [1], EDGES), [[0.0, 0.0, 0.0]])


def test_positions_of_simulated_events(tiny_config, tiny_tof_params):
    """the simulator's ordering / TOF mirroring (to_petsird_coincidences) and the decoding
    give the position expected from the parallelproj (unsigned) TOF bin"""
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    scanner = lor_desc.scanner
    n_el, n_tof = tiny_config.num_det_els_per_module, tiny_tof_params.num_tofbins
    # LOR from module 0 to module 4 (opposite), all TOF bins
    events = ListmodeEvents(
        start_module=np.zeros(n_tof, dtype="uint32"),
        start_el=np.full(n_tof, 1, dtype="uint32"),
        end_module=np.full(n_tof, 4, dtype="uint32"),
        end_el=np.full(n_tof, 2, dtype="uint32"),
        tof_bin=np.arange(n_tof, dtype="uint32"),
    )
    start = scanner.get_lor_endpoints(events.start_module, events.start_el)
    end = scanner.get_lor_endpoints(events.end_module, events.end_el)
    # parallelproj: unsigned TOF bin k is centered at (k - n_tof // 2) * width from the
    # LOR center towards the end point
    width = tiny_tof_params.tofbin_width
    direction = (end - start) / np.linalg.norm(end - start, axis=1, keepdims=True)
    expected = 0.5 * (start + end) + ((events.tof_bin.astype(float) - n_tof // 2) * width)[:, None] * direction

    det_bin_0, det_bin_1, tof_idx = to_petsird_coincidences(events, n_el, 1, n_tof)
    adps = scanner.get_lor_endpoints(np.repeat(np.arange(tiny_config.num_modules), n_el), np.tile(np.arange(n_el), tiny_config.num_modules))
    edges = (np.arange(n_tof + 1) - n_tof / 2) * width
    positions = tof_bin_center_positions(adps[det_bin_0], adps[det_bin_1], tof_idx, edges)
    np.testing.assert_allclose(positions, expected, atol=1e-4)


def test_splat_nearest_and_trilinear():
    shape, voxel, origin = (4, 4, 4), (2.0, 2.0, 2.0), np.zeros(3)
    on_voxel = np.array([[2.0, 4.0, 6.0]])  # voxel (1, 2, 3)
    between = np.array([[3.0, 4.0, 6.0]])  # halfway between voxels (1,2,3) and (2,2,3)
    outside = np.array([[100.0, 0.0, 0.0]])
    for interpolation in ("nearest", "trilinear"):
        img, n_out = splat(on_voxel, shape, voxel, origin, interpolation=interpolation)
        assert img[1, 2, 3] == pytest.approx(1) and img.sum() == pytest.approx(1) and n_out == 0
    img, _ = splat(between, shape, voxel, origin, weights=[2.0], interpolation="trilinear")
    assert img[1, 2, 3] == pytest.approx(1) and img[2, 2, 3] == pytest.approx(1)
    img, n_out = splat(np.vstack([on_voxel, outside]), shape, voxel, origin)
    assert n_out == 1 and img.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        splat(on_voxel, shape, voxel, origin, interpolation="cubic")


def test_uniform_jitter_stays_within_tof_bin():
    rng = np.random.default_rng(0)
    a = np.zeros((1000, 3)); a[:, 0] = -100
    b = np.zeros((1000, 3)); b[:, 0] = 100
    p = tof_bin_center_positions(a, b, np.full(1000, 2), EDGES, uniform_jitter=rng)
    assert np.all((p[:, 0] >= 5) & (p[:, 0] <= 15)) and abs(p[:, 0].mean() - 10) < 0.5
    np.testing.assert_allclose(p[:, 1:], 0)
