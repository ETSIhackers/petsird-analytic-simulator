import contextlib
import io

import numpy as np
from petsird.validate.geometry import validate_detecting_element

from geometry_plots import get_detecting_element_geometry
from petsird_reader import get_all_detector_centers
from scanner import (
    circular_distance,
    coincident_module_pairs,
    create_parallelproj_lor_descriptor,
    create_petsird_scanner_geometry,
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

    # the box axes of all detecting elements point away from the scanner axis
    g = get_detecting_element_geometry_from_geometry(geom)
    radial = g["adp"][..., :2] / np.linalg.norm(g["adp"][..., :2], axis=-1, keepdims=True)
    assert np.all(np.sum(g["box_axis"][..., :2] * radial, axis=-1) > 0.99)


def get_detecting_element_geometry_from_geometry(geom):
    class _Scanner:
        scanner_geometry = geom

    return get_detecting_element_geometry(_Scanner())[0]


def test_coincident_module_pairs(tiny_config):
    pairs = coincident_module_pairs(tiny_config)
    assert all(i < j for i, j in pairs)
    assert all(
        circular_distance(i, j, tiny_config.num_modules) >= tiny_config.min_coincidence_distance
        for i, j in pairs
    )
    # all pairs except neighbors (circular distance 1)
    n = tiny_config.num_modules
    assert len(pairs) == n * (n - 1) // 2 - n
