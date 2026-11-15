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
