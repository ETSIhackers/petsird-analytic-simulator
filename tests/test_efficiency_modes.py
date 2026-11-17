"""02 --efficiencies: only some of the detection efficiency factors are used"""

import numpy as np
import petsird
import pytest
from parallelproj.operators import GaussianFilterOperator

from conftest import TINY_IMG_SHAPE, TINY_VOXEL_SIZE
from efficiencies import generate_efficiency_model
from petsird_reader import (
    get_all_detector_centers,
    get_module_pair_efficiencies,
    get_module_pair_sgid_lut,
    read_listmode_events,
    select_efficiencies,
)
from reconstruction import default_image_grid, recon_params_hash, sensitivity_image
from scanner import create_parallelproj_lor_descriptor
from simulation import ListmodeEvents, create_petsird_header, to_petsird_coincidences, write_petsird_file


def test_missing_module_pair_efficiencies_are_one_for_all_sgids(tiny_header):
    """a file with an SGID LUT but without module pair efficiencies (efficiency 1)"""
    header = select_efficiencies(tiny_header, "detection_bins")
    lut = get_module_pair_sgid_lut(header.scanner, 0, 0)
    mp_effs = get_module_pair_efficiencies(header.scanner, 0, 0)
    assert lut.max() > 0
    assert mp_effs.shape[0] == lut.max() + 1
    assert np.all(mp_effs == 1)


def test_select_efficiencies_keeps_original_and_sgid_lut(tiny_header):
    original = tiny_header.scanner.detection_efficiencies
    for mode in ("module_pairs", "detection_bins", "none"):
        h = select_efficiencies(tiny_header, mode)
        effs = h.scanner.detection_efficiencies
        assert effs.module_pair_sgidlut == original.module_pair_sgidlut
        assert effs.calibration_factor == 1.0
        assert (len(effs.detection_bin_efficiencies) > 0) == (mode == "detection_bins")
        assert (len(effs.module_pair_efficiencies_vectors) > 0) == (mode == "module_pairs")
    # the original header is not modified
    assert tiny_header.scanner.detection_efficiencies is original
    assert len(original.detection_bin_efficiencies) > 0
    assert len(original.module_pair_efficiencies_vectors) > 0
    assert select_efficiencies(tiny_header, "all") is tiny_header
    with pytest.raises(ValueError):
        select_efficiencies(tiny_header, "half")


def test_event_efficiencies_of_all_modes(tmp_path, tiny_config, tiny_header, tiny_tof_params):
    """all = module_pairs x detection_bins (calibration factor 1), none = 1, for events of
    all module pairs in coincidence"""
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    num_el = tiny_config.num_det_els_per_module
    rng = np.random.default_rng(0)
    n = 300
    i_bp = rng.integers(0, lor_desc.num_block_pairs, n)
    i_lor = rng.integers(0, num_el**2, n)
    pairs = np.asarray(lor_desc.all_block_pairs)
    events = ListmodeEvents(
        start_module=pairs[i_bp, 0].astype("uint32"),
        start_el=(i_lor // num_el).astype("uint32"),
        end_module=pairs[i_bp, 1].astype("uint32"),
        end_el=(i_lor % num_el).astype("uint32"),
        tof_bin=rng.integers(0, tiny_tof_params.num_tofbins, n).astype("uint32"),
    )
    fname = tmp_path / "effs.bin"
    write_petsird_file(fname, tiny_header, *to_petsird_coincidences(events, num_el, 1, tiny_tof_params.num_tofbins), 2, 50)

    effs = {}
    for mode in ("all", "module_pairs", "detection_bins", "none"):
        reader = petsird.BinaryPETSIRDReader(str(fname))
        h = reader.read_header()
        centers = get_all_detector_centers(h.scanner.scanner_geometry)
        effs[mode] = read_listmode_events(reader, select_efficiencies(h, mode), centers, backend="petsird")["prompts"]["effs"]
    assert tiny_header.scanner.detection_efficiencies.calibration_factor == 1.0
    np.testing.assert_allclose(effs["all"], effs["module_pairs"] * effs["detection_bins"], rtol=1e-6)
    np.testing.assert_array_equal(effs["none"], 1)
    # the tiny efficiency model has dead crystals and non-uniform module pair efficiencies
    assert not np.allclose(effs["module_pairs"], 1) and not np.allclose(effs["detection_bins"], 1)


def test_sensitivity_without_efficiencies_is_geometric(tiny_config, tiny_header, tiny_tof_params):
    """sensitivity image of mode "none" == sensitivity image of a scanner with uniform
    efficiencies (all module pairs in coincidence with weight 1)"""
    uniform = create_petsird_header(
        tiny_config,
        generate_efficiency_model(tiny_config, uniform_crystal_eff=True, uniform_sg_eff=True),
        tiny_tof_params,
        20.0,
    )
    centers = get_all_detector_centers(tiny_header.scanner.scanner_geometry)
    img_shape, img_origin = default_image_grid(centers, TINY_VOXEL_SIZE, TINY_IMG_SHAPE)
    res_model = GaussianFilterOperator(img_shape, sigma=0.5)
    kw = dict(all_detector_centers=centers, img_shape=img_shape, voxel_size=TINY_VOXEL_SIZE, img_origin=img_origin, res_model=res_model, tof=False)
    sens_none = sensitivity_image(select_efficiencies(tiny_header, "none").scanner, **kw)
    sens_uniform = sensitivity_image(uniform.scanner, **kw)
    sens_all = sensitivity_image(tiny_header.scanner, **kw)
    np.testing.assert_allclose(sens_none, sens_uniform, rtol=1e-5, atol=1e-6 * sens_uniform.max())
    assert not np.allclose(sens_all, sens_none, rtol=1e-2)


def test_recon_params_hash_of_efficiency_modes():
    args = ((4, 4, 4), (1.0, 1.0, 1.0), True, None, 2.0)
    # unchanged for the default mode, different for all other modes
    assert recon_params_hash(*args) == recon_params_hash(*args, "all")
    hashes = {recon_params_hash(*args, mode) for mode in ("all", "module_pairs", "detection_bins", "none")}
    assert len(hashes) == 4
