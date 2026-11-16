import numpy as np
import petsird

from efficiencies import lor_efficiencies
from petsird_reader import get_all_detector_centers, read_listmode_events
from scanner import create_parallelproj_lor_descriptor
from simulation import ListmodeEvents, to_petsird_coincidences, write_petsird_file


def test_tof_parameters(tiny_tof_params):
    assert tiny_tof_params.num_tofbins % 2 == 1


def test_coincidences_are_ordered_and_tof_mirrored():
    num_el, num_tofbins = 4, 7
    events = ListmodeEvents(
        start_module=np.array([0, 3], dtype="uint32"),
        start_el=np.array([1, 2], dtype="uint32"),
        end_module=np.array([2, 1], dtype="uint32"),
        end_el=np.array([3, 0], dtype="uint32"),
        tof_bin=np.array([1, 1], dtype="uint32"),
    )
    det_bin_0, det_bin_1, tof_idx = to_petsird_coincidences(events, num_el, 1, num_tofbins)
    # event 0: start bin 1 < end bin 11 -> swapped and TOF bin mirrored
    # event 1: start bin 14 > end bin 4 -> unchanged
    np.testing.assert_array_equal(det_bin_0, [11, 14])
    np.testing.assert_array_equal(det_bin_1, [1, 4])
    np.testing.assert_array_equal(tof_idx, [num_tofbins - 1 - 1, 1])


def test_write_read_roundtrip(tmp_path, tiny_config, tiny_efficiency_model, tiny_header, tiny_tof_params):
    """events written to PETSIRD are read back with the correct LOR end points,
    TOF bins and efficiencies"""
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    num_el = tiny_config.num_det_els_per_module
    rng = np.random.default_rng(0)
    num_events = 200
    i_bp = rng.integers(0, lor_desc.num_block_pairs, num_events)
    i_lor = rng.integers(0, num_el**2, num_events)
    pairs = np.asarray(lor_desc.all_block_pairs)
    events = ListmodeEvents(
        start_module=pairs[i_bp, 0].astype("uint32"),
        start_el=(i_lor // num_el).astype("uint32"),
        end_module=pairs[i_bp, 1].astype("uint32"),
        end_el=(i_lor % num_el).astype("uint32"),
        tof_bin=rng.integers(0, tiny_tof_params.num_tofbins, num_events).astype("uint32"),
    )
    det_bin_0, det_bin_1, tof_idx = to_petsird_coincidences(
        events, num_el, 1, tiny_tof_params.num_tofbins
    )
    fname = tmp_path / "test.bin"
    write_petsird_file(fname, tiny_header, det_bin_0, det_bin_1, tof_idx, 2, 50)

    reader = petsird.BinaryPETSIRDReader(str(fname))
    header = reader.read_header()
    centers = get_all_detector_centers(header.scanner.scanner_geometry)
    prompts = read_listmode_events(reader, header, centers)["prompts"]
    coords0, coords1, tof_bins, effs = (
        prompts[k] for k in ("coords0", "coords1", "tof_bins", "effs")
    )
    # global detection bin indices (one module type, all events written in time order)
    np.testing.assert_array_equal(prompts["det_bin0"], det_bin_0)
    np.testing.assert_array_equal(prompts["det_bin1"], det_bin_1)

    # PETSIRD first detection bin = parallelproj end point (end module has the higher index)
    scanner = lor_desc.scanner
    np.testing.assert_allclose(
        coords0, scanner.get_lor_endpoints(events.end_module, events.end_el), atol=1e-4
    )
    np.testing.assert_allclose(
        coords1, scanner.get_lor_endpoints(events.start_module, events.start_el), atol=1e-4
    )
    # TOF bins w.r.t. the PETSIRD order (first -> second detection bin) are mirrored
    np.testing.assert_array_equal(tof_bins, tiny_tof_params.num_tofbins - 1 - events.tof_bin)
    np.testing.assert_allclose(
        effs, lor_efficiencies(tiny_efficiency_model, lor_desc)[i_bp, i_lor], rtol=1e-6
    )


def test_referenced_materials_are_defined(tiny_header):
    """all material ids of the scanner geometry exist in bulk_materials (PETSIRD constraint)"""
    scanner = tiny_header.scanner
    materials = {m.id: m for m in scanner.bulk_materials}
    geom = scanner.scanner_geometry
    referenced = {rm.object.detecting_elements.object.material_id for rm in geom.replicated_modules}
    referenced |= {v.material_id for v in (geom.non_detecting_volumes or [])}
    assert referenced <= set(materials)
    for m in materials.values():
        assert len(m.atoms) == len(m.mass_fractions)
        assert abs(sum(m.mass_fractions) - 1) < 1e-6
        assert m.density > 0


def test_randoms_expectation(tiny_config):
    """randoms of a LOR are proportional to the product of the singles and uniform in TOF"""
    from simulation import lor_products, randoms_expectation

    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    singles = np.random.default_rng(0).uniform(0.5, 2, (tiny_config.num_modules, tiny_config.num_det_els_per_module))
    singles[1, 2] = 0
    randoms, scaled = randoms_expectation(singles, lor_desc, num_tofbins=5, total_expected_randoms=100.0)
    assert randoms.shape == (lor_desc.num_block_pairs, tiny_config.num_det_els_per_module**2, 5)
    np.testing.assert_allclose(randoms.sum(), 100.0, rtol=1e-5)
    np.testing.assert_allclose(randoms.sum(-1), lor_products(scaled, lor_desc), rtol=1e-5)
    np.testing.assert_allclose(randoms, randoms[..., :1].repeat(5, -1))
    # scaled singles are proportional to the singles
    ratio = scaled[singles > 0] / singles[singles > 0]
    np.testing.assert_allclose(ratio, ratio[0])
    assert scaled[1, 2] == 0


def test_singles_model():
    """singles = eff x (uniform part + flux part), 0 for dead elements"""
    from simulation import singles_model

    effs = np.array([[1.0, 2.0], [0.0, 1.0]])
    trues_fan_sums = np.array([[1.0, 2.0], [0.0, 3.0]])  # flux 1, 1, -, 3 (mean 5/3)
    s = singles_model(trues_fan_sums, effs, uniform_fraction=0.5)
    np.testing.assert_allclose(s, [[0.5 + 0.3, 2 * (0.5 + 0.3)], [0.0, 0.5 + 0.9]])


def test_write_read_delayeds_and_singles(tmp_path, tiny_config, tiny_efficiency_model, tiny_tof_params):
    """delayed coincidences (with the delayed policy) and singles histograms (level ALL)
    are written and read back"""
    from simulation import create_petsird_header

    header = create_petsird_header(
        tiny_config,
        tiny_efficiency_model,
        tiny_tof_params,
        20.0,
        store_delayeds=True,
        store_singles=True,
    )
    assert header.scanner.delayed_event_policy == petsird.CoincidencePolicy.MULTIPLES_AS_ALL_COINCIDENCES
    prompts = (np.array([30, 40]), np.array([1, 2]), np.array([3, 4]))
    delayeds = (np.array([35, 41, 50]), np.array([5, 6, 7]), np.array([0, 1, 2]))
    fname = tmp_path / "delayeds.bin"
    num_bins = tiny_config.num_modules * tiny_config.num_det_els_per_module
    singles = np.arange(2 * num_bins).reshape(2, num_bins)
    write_petsird_file(fname, header, *prompts, 2, 50, delayeds=delayeds, singles_counts=singles)

    reader = petsird.BinaryPETSIRDReader(str(fname))
    h = reader.read_header()
    events = read_listmode_events(reader, h, get_all_detector_centers(h.scanner.scanner_geometry))
    assert events["prompts"]["coords0"].shape[0] == 2
    assert events["delayeds"]["coords0"].shape[0] == 3
    np.testing.assert_array_equal(np.sort(events["delayeds"]["tof_bins"]), [0, 1, 2])
    # singles histograms (one per time block)
    assert h.scanner.singles_histogram_level == petsird.SinglesHistogramLevelType.ALL
    np.testing.assert_array_equal(events["singles"]["counts"], singles)
    np.testing.assert_array_equal(events["singles"]["durations_ms"], [50, 50])


def test_read_time_range(tmp_path, tiny_config, tiny_efficiency_model, tiny_tof_params):
    """only time blocks whose interval lies within the time range are read"""
    from simulation import create_petsird_header

    header = create_petsird_header(
        tiny_config, tiny_efficiency_model, tiny_tof_params, 20.0,
        store_delayeds=True, store_singles=True,
    )
    rng = np.random.default_rng(0)
    prompts = (rng.integers(40, 48, 30), rng.integers(0, 8, 30), rng.integers(0, 3, 30))
    delayeds = (rng.integers(40, 48, 9), rng.integers(0, 8, 9), rng.integers(0, 3, 9))
    num_bins = tiny_config.num_modules * tiny_config.num_det_els_per_module
    singles = rng.integers(0, 100, (3, num_bins))
    fname = tmp_path / "range.bin"
    # 3 time blocks [0, 50], [50, 100], [100, 150] ms with 10 prompts / 3 delayeds each
    write_petsird_file(fname, header, *prompts, 3, 50, delayeds=delayeds, singles_counts=singles)

    for backend in ("petsird", "auto"):
        reader = petsird.BinaryPETSIRDReader(str(fname))
        h = reader.read_header()
        events = read_listmode_events(
            reader, h, get_all_detector_centers(h.scanner.scanner_geometry),
            fname=str(fname), backend=backend, time_range_ms=(50, 150),
        )
        # event + singles time blocks of the first interval are skipped
        assert events["num_skipped_time_blocks"] == 2
        np.testing.assert_array_equal(events["prompts"]["det_bin0"], prompts[0][10:])
        np.testing.assert_array_equal(events["delayeds"]["det_bin1"], delayeds[1][3:])
        np.testing.assert_array_equal(events["singles"]["counts"], singles[1:])
        np.testing.assert_array_equal(events["singles"]["durations_ms"], [50, 50])


def test_first_events_match_read_first_events(tmp_path, tiny_config, tiny_efficiency_model, tiny_tof_params):
    """the first events taken from read_listmode_events() == read from the file"""
    from geometry_plots import read_first_events
    from petsird_reader import first_events
    from simulation import create_petsird_header

    header = create_petsird_header(tiny_config, tiny_efficiency_model, tiny_tof_params, 20.0)
    rng = np.random.default_rng(1)
    prompts = (rng.integers(40, 48, 30), rng.integers(0, 8, 30), rng.integers(0, 3, 30))
    fname = tmp_path / "first.bin"
    write_petsird_file(fname, header, *prompts, 2, 50)

    reader = petsird.BinaryPETSIRDReader(str(fname))
    h = reader.read_header()
    ref = read_first_events(reader, 8)
    reader = petsird.BinaryPETSIRDReader(str(fname))
    h = reader.read_header()
    events = read_listmode_events(reader, h, get_all_detector_centers(h.scanner.scanner_geometry))
    np.testing.assert_array_equal(first_events(events["prompts"], h.scanner, 8), ref)
