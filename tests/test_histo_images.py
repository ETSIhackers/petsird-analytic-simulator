import numpy as np
import pytest

from parallelproj.tof import TOFParameters

from histo_images import tof_bin_center_positions, tof_bin_centers, tof_histo_image
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


SHAPE, VOXEL = (41, 41, 21), (2.0, 2.0, 2.0)
ORIGIN = -0.5 * (np.asarray(SHAPE) - 1) * np.asarray(VOXEL)
TOF = TOFParameters(num_tofbins=3, tofbin_width=10.0, sigma_tof=1.0)  # = EDGES


def _histo(a, b, bins, **kw):
    return tof_histo_image(a, b, np.asarray(bins), TOF, SHAPE, VOXEL, ORIGIN, **kw)


def _center_of_mass(img):
    idx = np.indices(img.shape).reshape(3, -1)
    com = (idx * img.ravel()).sum(1) / img.sum()
    return com * np.asarray(VOXEL) + ORIGIN


def _random_events(n, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d[:, 2] *= 0.3
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = rng.uniform(-8, 8, (n, 3))
    c[:, 2] = rng.uniform(-4, 4, n)
    return c - 100 * d, c + 100 * d, rng.integers(0, 3, n)


def test_center_mode_matches_tof_bin_centers():
    """every event (oblique LORs) is placed at the center of its TOF bin (PETSIRD
    convention, reference: tof_bin_center_positions) with weight 1"""
    a, b, bins = _random_events(30)
    expected = tof_bin_center_positions(a, b, bins, EDGES)
    for i in range(len(bins)):
        img = np.asarray(_histo(a[i : i + 1], b[i : i + 1], bins[i : i + 1]))
        assert img.sum() == pytest.approx(1, rel=1e-4)
        np.testing.assert_allclose(_center_of_mass(img), expected[i], atol=0.3)


def test_uniform_mode_spreads_over_the_tof_bin():
    a, b = np.array([[-100.0, 0.0, 0.0]]), np.array([[100.0, 0.0, 0.0]])
    center = np.asarray(_histo(a, b, [2]))
    uniform = np.asarray(_histo(a, b, [2], tof_position="uniform"))
    for img in (center, uniform):
        assert img.sum() == pytest.approx(1, rel=1e-4)
        np.testing.assert_allclose(_center_of_mass(img), [10, 0, 0], atol=0.2)
    x = np.arange(SHAPE[0]) * VOXEL[0] + ORIGIN[0]
    profile_c, profile_u = center.sum((1, 2)), uniform.sum((1, 2))
    # center: (almost) all within +-2 mm, uniform: flat between the bin edges 5 ... 15 mm
    assert profile_c[np.abs(x - 10) <= 2].sum() > 0.95
    inner = np.abs(x - 10) <= 3
    assert profile_u[inner].min() > 0.8 * profile_u[inner].max()
    assert profile_u[np.abs(x - 10) > 8].sum() < 0.01
    with pytest.raises(ValueError):
        _histo(a, b, [2], tof_position="cubic")


def test_swapped_detection_bins_and_weights():
    """swapping the detection bins and mirroring the TOF bin gives the same image (PETSIRD
    convention), weights scale the contributions, events outside are not counted"""
    a, b, bins = _random_events(500, seed=1)
    img = np.asarray(_histo(a, b, bins))
    np.testing.assert_allclose(np.asarray(_histo(b, a, 2 - bins)), img, atol=1e-5)
    assert img.sum() == pytest.approx(500, rel=1e-3)
    w = np.random.default_rng(2).uniform(0.5, 2, 500)
    assert np.asarray(_histo(a, b, bins, weights=w)).sum() == pytest.approx(w.sum(), rel=1e-3)
    outside = np.array([[-100.0, 500.0, 0.0]]), np.array([[100.0, 500.0, 0.0]])
    assert np.asarray(_histo(*outside, [1])).sum() == 0
