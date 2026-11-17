import contextlib
import io

import numpy as np
from petsird.validate.geometry import validate_detecting_element

from geometry_plots import get_detecting_element_geometry
from petsird_reader import get_all_detector_centers
from scanner import (
    ScannerConfig,
    circular_distance,
    coincident_module_pairs,
    create_parallelproj_lor_descriptor,
    create_petsird_scanner_geometry,
    module_transforms,
)


def test_petsird_adps_match_parallelproj_lor_endpoints(tiny_config):
    """the PETSIRD geometry (written) and the parallelproj geometry (simulated) agree"""
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    pp_endpoints = np.asarray(lor_desc.scanner.all_lor_endpoints).reshape(
        tiny_config.num_modules, tiny_config.num_det_els_per_module, 3
    )
    adps = get_all_detector_centers(create_petsird_scanner_geometry(tiny_config))[0]
    np.testing.assert_allclose(adps, pp_endpoints, atol=1e-4)


def test_detecting_elements_follow_petsird_constraints(tiny_config):
    geom = create_petsird_scanner_geometry(tiny_config)
    box = geom.replicated_modules[0].object.detecting_elements.object.shape
    with contextlib.redirect_stdout(io.StringIO()):  # petsird.validate prints debug output
        validate_detecting_element(box)

    # the box axes of all detecting elements point away from the scanner axis, along the
    # normal of their panel
    g = get_detecting_element_geometry_from_geometry(geom)
    radial = g["adp"][..., :2] / np.linalg.norm(g["adp"][..., :2], axis=-1, keepdims=True)
    assert np.all(np.sum(g["box_axis"][..., :2] * radial, axis=-1) > 0.9)
    t = np.arange(tiny_config.num_modules) % tiny_config.num_transaxial_modules
    phi = 2 * np.pi * (t // tiny_config.modules_per_panel) / tiny_config.num_panels
    normal = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)
    np.testing.assert_allclose(g["box_axis"], np.broadcast_to(normal[:, None, :], g["box_axis"].shape), atol=1e-5)


def get_detecting_element_geometry_from_geometry(geom):
    class _Scanner:
        scanner_geometry = geom

    return get_detecting_element_geometry(_Scanner())[0]


def test_coincident_module_pairs(tiny_config):
    pairs = coincident_module_pairs(tiny_config)
    assert all(i < j for i, j in pairs)
    nt, na = tiny_config.num_transaxial_modules, tiny_config.num_axial_modules
    assert all(
        circular_distance(i % nt, j % nt, nt) >= tiny_config.min_coincidence_distance
        for i, j in pairs
    )
    # all pairs except transaxial neighbors / the same transaxial module (circular
    # distance 0 or 1), for all axial module offsets
    ordered_per_axial_pair = nt * (nt - 3)
    assert len(pairs) == na * ordered_per_axial_pair // 2 + na * (na - 1) // 2 * ordered_per_axial_pair


def _rotation_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _module_detector_centers(config):
    lor_desc = create_parallelproj_lor_descriptor(config)
    return np.asarray(lor_desc.scanner.all_lor_endpoints).reshape(config.num_modules, -1, 3)


def _by_axial_position(config, centers):
    return centers.reshape(config.num_axial_modules, config.num_transaxial_modules, -1, 3)


def test_default_scanner_symmetries():
    """6 flat panels of 2 modules at 3 axial positions: a rotation by one panel (60 deg)
    maps transaxial module t onto t + 2 at the same axial position (element by element),
    a rotation by 30 deg does not map the scanner onto itself (as it does for equally
    spaced modules); the axial positions are shifted copies of each other"""
    config = ScannerConfig()
    assert (config.num_panels, config.modules_per_panel, config.num_axial_modules) == (6, 2, 3)
    assert config.num_modules * config.num_det_els_per_module == 2160
    centers = _by_axial_position(config, _module_detector_centers(config))
    rotated = centers @ _rotation_z(np.pi / 3).T
    np.testing.assert_allclose(rotated, np.roll(centers, -2, axis=1), atol=1e-3)
    rotated_30 = centers @ _rotation_z(np.pi / 6).T
    assert np.abs(rotated_30 - np.roll(centers, -1, axis=1)).max() > 1.0

    # axial positions along +z, separated by the axial gap, centered at z = 0
    pitch = config.block_shape[2] * config.block_spacing[2] + config.axial_gap
    for a in range(1, config.num_axial_modules):
        np.testing.assert_allclose(centers[a], centers[0] + np.array([0, 0, a * pitch]), atol=1e-4)
    np.testing.assert_allclose(centers[..., 2].mean(), 0, atol=1e-4)

    # equally spaced modules: symmetric under a rotation by one module
    equidistant = ScannerConfig(modules_per_panel=1, num_axial_modules=1)
    centers = _module_detector_centers(equidistant)
    np.testing.assert_allclose(centers @ _rotation_z(np.pi / 6).T, np.roll(centers, -1, axis=0), atol=1e-3)


def test_panel_layout():
    """the modules of a panel are coplanar, separated by the gap, in counterclockwise order"""
    config = ScannerConfig()
    t = module_transforms(config)
    width = config.block_shape[0] * config.block_spacing[0]
    for k in range(config.num_panels):
        m0, m1 = t[2 * k], t[2 * k + 1]
        np.testing.assert_allclose(m0[:3, :3], m1[:3, :3])
        # module 2k + 1 is displaced along -x of the panel (counterclockwise)
        shift = np.linalg.inv(m0) @ m1[:, 3]
        np.testing.assert_allclose(shift[:3], (-(width + config.panel_gap), 0, 0), atol=1e-4)
    # no overlap of neighboring panels at the front face
    inner = config.radius - config.block_shape[1] * config.block_spacing[1] / 2
    assert width + config.panel_gap / 2 <= inner * np.tan(np.pi / config.num_panels)


def test_equidistant_module_transforms():
    """modules_per_panel = 1: modules equally spaced on the circle, facing the center"""
    config = ScannerConfig(modules_per_panel=1, num_axial_modules=1)
    for m, t in enumerate(module_transforms(config)):
        phi = 2 * np.pi * m / config.num_modules
        np.testing.assert_allclose(t[:3, 3], config.radius * np.array([-np.sin(phi), np.cos(phi), 0]), atol=1e-4)


def test_invalid_modules_per_panel():
    import pytest

    with pytest.raises(ValueError):
        ScannerConfig(num_transaxial_modules=12, modules_per_panel=5)
    with pytest.raises(ValueError):
        ScannerConfig(num_axial_modules=0)
