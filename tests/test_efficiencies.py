import numpy as np
import petsird
from petsird.helpers import get_detection_efficiency

from efficiencies import lor_efficiencies, module_offset
from scanner import create_parallelproj_lor_descriptor


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
    for d, m in effs.items():
        np.testing.assert_array_equal(effs[n - d], m.T)
        if 2 * d == n:
            np.testing.assert_array_equal(m, m.T)
        else:
            # non-symmetric to detect transposition errors in readers
            assert not np.allclose(m, m.T)


def test_module_offset():
    assert module_offset(5, 2, 8) == 3
    assert module_offset(2, 5, 8) == 5


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
    for d, m in model.module_pair_effs.items():
        np.testing.assert_array_equal(model.module_pair_effs[config.num_modules - d], m.T)
        if 2 * d == config.num_modules:
            np.testing.assert_array_equal(m, m.T)
        else:
            assert _relative_change(m, m.T) > 0.3, d
    # alternating levels between module offsets (symmetry groups)
    means = {d: m.mean() for d, m in model.module_pair_effs.items()}
    assert means[4] > 2 * means[3] and means[6] > 2 * means[5]


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
