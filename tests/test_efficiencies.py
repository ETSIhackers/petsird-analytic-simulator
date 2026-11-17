import numpy as np
import petsird
from petsird.helpers import get_detection_efficiency

from efficiencies import (
    generate_efficiency_model,
    lor_efficiencies,
    module_offset,
    module_pair_class,
    transposed_class,
)
from petsird_reader import get_module_pair_sgid_lut
from simulation import create_petsird_header, tof_parameters
from scanner import ScannerConfig, coincident_module_pairs, create_parallelproj_lor_descriptor


def test_lor_efficiencies_match_petsird_tables(tiny_config, tiny_efficiency_model, tiny_header):
    """efficiencies used in the simulation == efficiencies decoded from the PETSIRD tables
    (by petsird.helpers.get_detection_efficiency) for every LOR"""
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    sim_effs = lor_efficiencies(tiny_efficiency_model, lor_desc)
    num_el = tiny_config.num_det_els_per_module
    type_pair = petsird.TypeOfModulePair((0, 0))

    for i_bp, (start_mod, end_mod) in enumerate(lor_desc.all_block_pairs):
        for i_lor in range(num_el**2):
            start_el, end_el = divmod(i_lor, num_el)
            # PETSIRD: first detection bin is in the module with the higher index
            det_bin_0 = int(end_mod) * num_el + end_el
            det_bin_1 = int(start_mod) * num_el + start_el
            petsird_eff = get_detection_efficiency(
                tiny_header.scanner, type_pair, det_bin_0, det_bin_1
            )
            np.testing.assert_allclose(sim_effs[i_bp, i_lor], petsird_eff, rtol=1e-6)


def test_module_pair_efficiencies_symmetries(tiny_config, tiny_efficiency_model):
    n = tiny_config.num_modules
    effs = tiny_efficiency_model.module_pair_effs
    for cls, m in effs.items():
        transposed = transposed_class(cls, n)
        np.testing.assert_array_equal(effs[transposed], m.T)
        if transposed == cls:
            np.testing.assert_array_equal(m, m.T)
        else:
            # non-symmetric to detect transposition errors in readers
            assert not np.allclose(m, m.T)


def test_module_offset():
    assert module_offset(5, 2, 8) == 3
    assert module_offset(2, 5, 8) == 5


def test_module_pair_class():
    assert module_pair_class(5, 2, 12, 2) == (1, 0, 3)
    assert module_pair_class(4, 1, 12, 2) == (0, 1, 3)
    assert module_pair_class(5, 2, 12, 1) == (0, 0, 3)
    assert transposed_class((1, 0, 3), 12) == (0, 1, 9)
    assert transposed_class((0, 0, 6), 12) == (0, 0, 6)


# SGID LUT of the default scanner (6 panels of 2 modules): rows = first module i,
# columns = second module j < i, None = not in coincidence (circular distance < 3).
# SGID = index of the class (p1, p2, d) ordered by d, p1, p2:
# 0:(0,1,3) 1:(1,0,3) 2:(0,0,4) 3:(1,1,4) 4:(0,1,5) 5:(1,0,5) 6:(0,0,6) 7:(1,1,6)
# 8:(0,1,7) 9:(1,0,7) 10:(0,0,8) 11:(1,1,8) 12:(0,1,9) 13:(1,0,9)
DEFAULT_SGID_LUT_ROWS = {
    3: [1],
    4: [2, 0],
    5: [5, 3, 1],
    6: [6, 4, 2, 0],
    7: [9, 7, 5, 3, 1],
    8: [10, 8, 6, 4, 2, 0],
    9: [13, 11, 9, 7, 5, 3, 1],
    10: [None, 12, 10, 8, 6, 4, 2, 0],
    11: [None, None, 13, 11, 9, 7, 5, 3, 1],
}


def _default_sgid_lut(config, model):
    header = create_petsird_header(config, model, tof_parameters(20.0, (8, 8, 4), (4.0, 4.0, 4.0)), 20.0)
    return get_module_pair_sgid_lut(header.scanner, 0, 0)


def test_sgid_lut_of_default_scanner():
    config = ScannerConfig()
    np.random.seed(0)
    model = generate_efficiency_model(config)
    assert len(model.module_pair_effs) == 14
    lut = _default_sgid_lut(config, model)
    for i in range(config.num_modules):
        for j in range(i):
            expected = DEFAULT_SGID_LUT_ROWS.get(i, [])[j] if j < len(DEFAULT_SGID_LUT_ROWS.get(i, [])) else None
            assert lut[i, j] == (-1 if expected is None else expected), (i, j)
    # transposed SGIDs, 6 and 7 (opposite modules) are symmetric
    by_sgid = {model.sgid(c): m for c, m in model.module_pair_effs.items()}
    transposed = {0: 13, 1: 12, 2: 10, 3: 11, 4: 9, 5: 8, 6: 6, 7: 7}
    for k, k_t in transposed.items():
        np.testing.assert_array_equal(by_sgid[k_t], by_sgid[k].T)


def test_equidistant_scanner_has_one_sgid_per_offset():
    config = ScannerConfig(modules_per_panel=1)
    np.random.seed(0)
    model = generate_efficiency_model(config)
    assert sorted(model.module_pair_effs) == [(0, 0, d) for d in range(3, 10)]
    lut = _default_sgid_lut(config, model)
    for i, j in coincident_module_pairs(config):
        assert lut[j, i] == (j - i) - 3


def test_lor_efficiencies_are_invariant_under_panel_rotation():
    """the efficiency of a LOR (module pair efficiency x detecting element efficiencies
    with uniform detecting elements) is invariant under a rotation by one panel"""
    config = ScannerConfig()
    np.random.seed(0)
    model = generate_efficiency_model(config, uniform_crystal_eff=True)
    lor_desc = create_parallelproj_lor_descriptor(config)
    effs = lor_efficiencies(model, lor_desc)
    pairs = [tuple(map(int, p)) for p in lor_desc.all_block_pairs]
    index = {p: i for i, p in enumerate(pairs)}
    n, k = config.num_modules, config.modules_per_panel
    num_el = config.num_det_els_per_module
    # LOR order within a block pair: (start element, end element)
    e = effs.reshape(-1, num_el, num_el)
    for i_bp, (start, end) in enumerate(pairs):
        s2, e2 = (start + k) % n, (end + k) % n
        if s2 < e2:
            np.testing.assert_array_equal(e[index[(s2, e2)]], e[i_bp])
        else:
            np.testing.assert_array_equal(e[index[(e2, s2)]], e[i_bp].T)


def _relative_change(a, b):
    """median relative change between efficiencies (where both are > 0)"""
    ok = (a > 0) & (b > 0)
    return float(np.median(np.abs(a[ok] - b[ok]) / a[ok]))


def test_structured_detecting_element_efficiencies_have_no_symmetries():
    """flipping / swapping the tangential, layer or axial index order (typical reader
    errors) changes the detecting element efficiencies substantially"""
    from efficiencies import generate_efficiency_model
    from scanner import ScannerConfig

    config = ScannerConfig()
    np.random.seed(0)
    effs = generate_efficiency_model(config).det_el_effs.reshape(config.num_modules, *config.block_shape)
    for axis, label in ((1, "tangential"), (2, "layer"), (3, "axial")):
        assert _relative_change(effs, np.flip(effs, axis=axis)) > 0.2, label
    # module order around the ring (e.g. mirrored scanner geometry)
    assert _relative_change(effs, np.flip(effs, axis=0)) > 0.2
    # one dead patch of 3 x 3 elements in the front layer of module 7, all others alive
    dead = effs == 0
    assert dead.sum() == 9 and np.all(dead[7, :, 0, :].sum() == 9)


def test_structured_module_pair_efficiencies_are_strongly_non_symmetric():
    """transposed module pair matrices (typical reader error) differ substantially, except
    for opposite modules where the matrix has to be symmetric"""
    from efficiencies import generate_efficiency_model
    from scanner import ScannerConfig

    config = ScannerConfig()
    np.random.seed(0)
    model = generate_efficiency_model(config)
    for cls, m in model.module_pair_effs.items():
        transposed = transposed_class(cls, config.num_modules)
        np.testing.assert_array_equal(model.module_pair_effs[transposed], m.T)
        if transposed == cls:
            np.testing.assert_array_equal(m, m.T)
        else:
            assert _relative_change(m, m.T) > 0.3, cls
    # levels alternating between module offsets and different for the same offset
    # (different positions of the modules in their panels)
    means = {cls: m.mean() for cls, m in model.module_pair_effs.items()}
    assert means[(0, 0, 4)] > 1.3 * means[(1, 0, 3)] and means[(1, 0, 3)] > 2 * means[(0, 1, 3)]
    assert means[(1, 1, 4)] > 1.2 * means[(0, 0, 4)]


def test_random_efficiency_model_still_available():
    from efficiencies import generate_efficiency_model
    from scanner import ScannerConfig

    config = ScannerConfig()
    np.random.seed(0)
    effs = generate_efficiency_model(config, model="random").det_el_effs
    assert effs.max() > 3 and 0 < np.count_nonzero(effs == 0) < 100
    import pytest

    with pytest.raises(ValueError):
        generate_efficiency_model(config, model="unknown")
