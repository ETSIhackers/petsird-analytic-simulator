"""reading and interpreting PETSIRD data (scanner information and coincidence events)

Everything is derived from the PETSIRD data following the PETSIRD definitions only
(independent of how the data were simulated).
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import petsird
from parallelproj.tof import TOFParameters

from utils import FWHM_PER_SIGMA, Progress, petsird_fast, resolve_petsird_backend

# NOTE on conventions used in this module (PETSIRD >= 0.13):
# - a DetectionBin of a module type encodes
#     energy_idx + (element_idx + module_idx * num_det_els) * num_energy_bins
# - all tables over pairs of module types (TOF bin edges, TOF resolution,
#   prompt events, SGID LUTs, module pair efficiencies) are lower-triangular,
#   i.e. they only exist for [type_1][type_2] with type_2 <= type_1
# - for two modules of the same type, the SGID LUT is lower-triangular as well,
#   i.e. lut[module_1][module_2] only exists for module_2 <= module_1
# - coincidences are ordered such that the first detection bin is in the module
#   of type_1 and (for equal types) detection_bins[0] >= detection_bins[1]
# - parallelproj (>= 2) uses unsigned TOF bins (0 ... num_tofbins - 1) which
#   correspond directly to the PETSIRD tof_idx


class UnsupportedPETSIRDFeatureError(NotImplementedError):
    """a PETSIRD feature that is used in a file is not supported by the reconstruction"""


def _is_identity_transform(transform: petsird.RigidTransformation | None) -> bool:
    return transform is None or np.allclose(
        transform.matrix, np.eye(4)[:3], atol=1e-6
    )


def check_header_support(header: petsird.Header) -> None:
    """check whether the PETSIRD header uses features not supported by the reconstruction

    Raises UnsupportedPETSIRDFeatureError for features that would lead to wrong
    reconstructions and warns about information that is ignored.
    Features of the time blocks are checked in read_listmode_events().
    """
    scanner: petsird.ScannerInformation = header.scanner
    num_module_types = scanner.scanner_geometry.number_of_module_types()

    if header.contains_weighted_event_time_blocks:
        raise UnsupportedPETSIRDFeatureError(
            "Weighted event time blocks are not supported."
        )

    if scanner.prompt_event_policy == petsird.CoincidencePolicy.NONE:
        raise UnsupportedPETSIRDFeatureError(
            "The file does not contain prompt events (prompt_event_policy is NONE)."
        )

    # the reconstruction uses the same TOF parameters for all module type pairs
    tof_params = [
        get_tof_parameters(scanner, m1, m2)
        for m1 in range(num_module_types)
        for m2 in range(m1 + 1)
    ]
    if any(
        (p.num_tofbins, p.tofbin_width, p.sigma_tof)
        != (tof_params[0].num_tofbins, tof_params[0].tofbin_width, tof_params[0].sigma_tof)
        for p in tof_params
    ):
        raise UnsupportedPETSIRDFeatureError(
            "Different TOF bin edges / TOF resolutions for different module type pairs "
            "are not supported."
        )

    if not _is_identity_transform(scanner.gantry_alignment):
        warnings.warn(
            "The gantry alignment of the scanner is ignored: the reconstructed image "
            "(and an attenuation image) are in the PET gantry coordinate system.",
            UserWarning,
        )

    ignored = []
    if scanner.single_event_policy != petsird.SingleEventPolicy.NONE:
        ignored.append("single events")
    if scanner.triple_event_policy != petsird.TripleEventPolicy.NONE:
        ignored.append("triple events")
    if scanner.quadruple_event_policy != petsird.QuadrupleEventPolicy.NONE:
        ignored.append("quadruple events")
    if scanner.singles_histogram_level == petsird.SinglesHistogramLevelType.MODULE:
        ignored.append("singles histograms per singles unit (level MODULE)")
    if scanner.scanner_geometry.non_detecting_volumes:
        ignored.append("non-detecting volumes")
    if ignored:
        warnings.warn(
            f"The following information in the file is ignored: {', '.join(ignored)}",
            UserWarning,
        )


def get_all_detector_centers(
    scanner_geometry: petsird.ScannerGeometry,
) -> list[np.ndarray]:
    """average detection points (ADPs) of all detecting elements of all modules

    Returns a list with one array per module type of shape (num_modules, num_det_els, 3).
    For a given module type, module number and detecting element number, the ADP is
    all_det_el_centers[module_type][module_number, detecting_element_number, :]

    By definition, the ADP of a detecting element is (0,0,0) in the coordinate system
    of its box shape, such that the ADP in scanner coordinates is given by the
    translation part of the combined module and detecting element transform.
    This is a vectorized version of petsird.helpers.geometry.get_ADP().
    (see geometry_plots.py for a visualization of the detecting elements)
    """

    all_det_el_centers = []

    for rep_module in scanner_geometry.replicated_modules:
        det_els = rep_module.object.detecting_elements

        # (num_modules, 3, 4) and (num_det_els, 3, 4) affine matrices
        mod_mats = np.array([t.matrix for t in rep_module.transforms], dtype="float64")
        el_mats = np.array([t.matrix for t in det_els.transforms], dtype="float64")

        # R_mod @ t_el + t_mod for all module / detecting element combinations
        det_el_centers = (
            np.einsum("mij,ej->mei", mod_mats[:, :, :3], el_mats[:, :, 3])
            + mod_mats[:, None, :, 3]
        )
        all_det_el_centers.append(det_el_centers.astype("float32"))

    return all_det_el_centers


def module_type_info(
    scanner_info: petsird.ScannerInformation, module_type: int
) -> tuple[int, int, int]:
    """number of modules, detecting elements per module and energy bins of a module type"""
    rep_module = scanner_info.scanner_geometry.replicated_modules[module_type]
    num_modules = len(rep_module.transforms)
    num_det_els = len(rep_module.object.detecting_elements.transforms)
    num_energy_bins = scanner_info.event_energy_bin_edges[module_type].number_of_bins()
    return num_modules, num_det_els, num_energy_bins


def get_detection_bin_efficiencies(
    scanner_info: petsird.ScannerInformation, module_type: int
) -> np.ndarray:
    """detection bin efficiencies of a module type as array of shape (num_modules, num_det_els, num_energy_bins)

    Missing (or empty) efficiencies are considered to be 1 (see petsird.DetectionEfficiencies).
    """
    num_modules, num_det_els, num_energy_bins = module_type_info(
        scanner_info, module_type
    )
    shape = (num_modules, num_det_els, num_energy_bins)

    det_effs = scanner_info.detection_efficiencies
    if (
        det_effs is None
        or not det_effs.detection_bin_efficiencies
        or len(det_effs.detection_bin_efficiencies[module_type]) == 0
    ):
        return np.ones(shape, dtype="float32")

    return np.asarray(
        det_effs.detection_bin_efficiencies[module_type], dtype="float32"
    ).reshape(shape)


def get_module_pair_sgid_lut(
    scanner_info: petsird.ScannerInformation, module_type_1: int, module_type_2: int
) -> np.ndarray:
    """dense (num_modules_1, num_modules_2) array with the SGIDs of all module pairs

    The PETSIRD SGID LUT is lower-triangular for equal module types. Entries outside
    the lower triangle are set to -1 (not in coincidence), such that every module pair
    is contained exactly once.

    If no SGID LUT is given, all module pairs (excluding a module with itself)
    are assumed to be in coincidence with SGID 0.
    """
    num_modules_1 = module_type_info(scanner_info, module_type_1)[0]
    num_modules_2 = module_type_info(scanner_info, module_type_2)[0]
    same_type = module_type_1 == module_type_2

    lut = np.full((num_modules_1, num_modules_2), -1, dtype="int32")

    det_effs = scanner_info.detection_efficiencies
    if (
        det_effs is None
        or not det_effs.module_pair_sgidlut
        or len(det_effs.module_pair_sgidlut[module_type_1][module_type_2]) == 0
    ):
        for i in range(num_modules_1):
            lut[i, : (i if same_type else num_modules_2)] = 0
        return lut

    for i, row in enumerate(det_effs.module_pair_sgidlut[module_type_1][module_type_2]):
        lut[i, : len(row)] = row

    return lut


def get_module_pair_efficiencies(
    scanner_info: petsird.ScannerInformation, module_type_1: int, module_type_2: int
) -> np.ndarray:
    """module pair efficiencies as array of shape (num_SGIDs, num_det_bins_in_module_1, num_det_bins_in_module_2)

    If no module pair efficiencies are given, they are considered to be 1 (for a single SGID 0).
    """
    _, num_det_els_1, num_energy_bins_1 = module_type_info(scanner_info, module_type_1)
    _, num_det_els_2, num_energy_bins_2 = module_type_info(scanner_info, module_type_2)
    shape = (num_det_els_1 * num_energy_bins_1, num_det_els_2 * num_energy_bins_2)

    det_effs = scanner_info.detection_efficiencies
    if (
        det_effs is None
        or not det_effs.module_pair_efficiencies_vectors
        or len(det_effs.module_pair_efficiencies_vectors[module_type_1][module_type_2])
        == 0
    ):
        return np.ones((1,) + shape, dtype="float32")

    mp_effs_vector = det_effs.module_pair_efficiencies_vectors[module_type_1][
        module_type_2
    ]
    mp_effs = np.zeros((len(mp_effs_vector),) + shape, dtype="float32")
    for x in mp_effs_vector:
        mp_effs[x.sgid] = np.asarray(x.values, dtype="float32").reshape(shape)

    return mp_effs


def get_calibration_factor(scanner_info: petsird.ScannerInformation) -> float:
    det_effs = scanner_info.detection_efficiencies
    if det_effs is None:
        return 1.0
    if det_effs.calibration_factor == 0:
        warnings.warn(
            "Calibration factor in scanner information is 0 - using 1 instead.",
            UserWarning,
        )
        return 1.0
    return float(det_effs.calibration_factor)


def get_tof_parameters(
    scanner_info: petsird.ScannerInformation, module_type_1: int, module_type_2: int
) -> TOFParameters:
    """parallelproj TOF parameters for a pair of module types (module_type_2 <= module_type_1)"""
    tof_bin_edges = scanner_info.tof_bin_edges[module_type_1][module_type_2].edges
    tofbin_width = float(tof_bin_edges[1] - tof_bin_edges[0])

    # raise an error if tof_bin_edges are non equidistant (up to 0.1%)
    if not np.allclose(np.diff(tof_bin_edges), tofbin_width, rtol=0.001):
        raise UnsupportedPETSIRDFeatureError(
            f"TOF bin edges for module types {module_type_1} and {module_type_2} are not equidistant."
        )

    # TOF bins have to be symmetric around 0, since parallelproj assumes that the
    # central TOF bin is centered at the LOR center
    if not np.isclose(tof_bin_edges[0], -tof_bin_edges[-1], atol=1e-3 * tofbin_width):
        raise UnsupportedPETSIRDFeatureError(
            f"TOF bin edges for module types {module_type_1} and {module_type_2} are not symmetric around 0."
        )

    return TOFParameters(
        num_tofbins=tof_bin_edges.size - 1,
        tofbin_width=tofbin_width,
        # TOF resolution is stored as FWHM in mm
        sigma_tof=float(scanner_info.tof_resolution[module_type_1][module_type_2])
        / FWHM_PER_SIGMA,
    )


def get_num_data_bins(scanner_info: petsird.ScannerInformation, tof: bool) -> int:
    """number of data bins (pairs of detection bins in coincidence, times TOF bins)

    i.e. the number of bins of a (TOF) histogram of all possible coincidences
    """
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()
    num_bins = 0
    for m1 in range(num_module_types):
        _, num_det_els_1, num_energy_bins_1 = module_type_info(scanner_info, m1)
        for m2 in range(m1 + 1):
            _, num_det_els_2, num_energy_bins_2 = module_type_info(scanner_info, m2)
            num_module_pairs = int(
                np.count_nonzero(get_module_pair_sgid_lut(scanner_info, m1, m2) >= 0)
            )
            num_tofbins = (
                scanner_info.tof_bin_edges[m1][m2].number_of_bins() if tof else 1
            )
            num_bins += (
                num_module_pairs
                * num_det_els_1
                * num_energy_bins_1
                * num_det_els_2
                * num_energy_bins_2
                * num_tofbins
            )
    return num_bins


def detection_bin_offsets(scanner_info: petsird.ScannerInformation) -> np.ndarray:
    """offsets of the detection bins of all module types in a global detection bin index

    The global index of detection bin b of module type m is offsets[m] + b, i.e. the
    detection bins of all module types are numbered consecutively.
    offsets[-1] is the total number of detection bins.
    """
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()
    num_bins = [math.prod(module_type_info(scanner_info, m)) for m in range(num_module_types)]
    return np.concatenate([[0], np.cumsum(num_bins)]).astype(np.int64)


def first_events(
    events: dict[str, np.ndarray], scanner_info: petsird.ScannerInformation, num_events: int
) -> np.ndarray:
    """the first events of events read with read_listmode_events() in the format of
    geometry_plots.read_first_events(): array of shape (num_events, 5) with the columns
    (module type 0, module type 1, detection bin 0, detection bin 1, tof_idx)

    (for multiple module types, the events are ordered by module type pair)
    """
    offsets = detection_bin_offsets(scanner_info)
    columns = []
    for name in ("det_bin0", "det_bin1"):
        global_bins = events[name][:num_events]
        module_type = np.searchsorted(offsets, global_bins, side="right") - 1
        columns += [module_type, global_bins - offsets[module_type]]
    m0, b0, m1, b1 = columns
    return np.stack([m0, m1, b0, b1, events["tof_bins"][:num_events]], axis=1).astype(np.int64)


# names of the arrays describing listmode events (see read_listmode_events)
EVENT_ARRAYS = (
    "coords0",
    "coords1",
    "tof_bins",
    "effs",
    "energy_idx0",
    "energy_idx1",
    "det_bin0",
    "det_bin1",
    "event_index",
)


@dataclass
class DetectionBinTables:
    """lookup tables indexed by the detection bins of every module type (from the header)

    They convert detection bins of events to LOR end points and efficiencies with 1D
    lookups (no expansion into module / element / energy indices). The tables are small
    (one entry per detection bin / module pair matrix element) and could also be moved to
    a GPU to do the lookups there.
    """

    # per module type: ADPs (num_detection_bins, 3), detection bin efficiencies
    # (num_detection_bins,), number of detection bins per module and energy bins
    adps: list[np.ndarray]
    effs: list[np.ndarray]
    bins_per_module: list[int]
    num_energy_bins: list[int]
    # per module type pair (type_1, type_2 <= type_1): dense SGID LUT and the flattened
    # module pair efficiencies [sgid, detection bin in module 1, detection bin in module 2]
    sgid_luts: dict[tuple[int, int], np.ndarray]
    module_pair_effs: dict[tuple[int, int], np.ndarray]
    # global detection bin index offsets of all module types (see detection_bin_offsets)
    bin_offsets: np.ndarray
    calibration_factor: float


def detection_bin_tables(
    scanner_info: petsird.ScannerInformation,
    all_detector_centers: list[np.ndarray],
    with_calibration_factor: bool = True,
) -> DetectionBinTables:
    """lookup tables of all detection bins (see DetectionBinTables)"""
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()
    infos = [module_type_info(scanner_info, m) for m in range(num_module_types)]
    return DetectionBinTables(
        # detection bin = energy_idx + (element_idx + module_idx * num_det_els) * num_energy_bins
        adps=[
            np.repeat(all_detector_centers[m].reshape(-1, 3), infos[m][2], axis=0)
            for m in range(num_module_types)
        ],
        effs=[
            get_detection_bin_efficiencies(scanner_info, m).ravel()
            for m in range(num_module_types)
        ],
        bins_per_module=[info[1] * info[2] for info in infos],
        num_energy_bins=[info[2] for info in infos],
        sgid_luts={
            (m1, m2): get_module_pair_sgid_lut(scanner_info, m1, m2)
            for m1 in range(num_module_types)
            for m2 in range(m1 + 1)
        },
        module_pair_effs={
            (m1, m2): get_module_pair_efficiencies(scanner_info, m1, m2).reshape(-1)
            for m1 in range(num_module_types)
            for m2 in range(m1 + 1)
        },
        bin_offsets=detection_bin_offsets(scanner_info),
        calibration_factor=(
            get_calibration_factor(scanner_info) if with_calibration_factor else 1.0
        ),
    )


# raw coincidences of one stream per module type pair: (detection bin 0, detection bin 1,
# tof_idx) as uint32 arrays and the index of the events among all events of the stream and
# module type pair in the file (int64, file order, also counting skipped time blocks)
RawEvents = dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def read_listmode_events(
    reader: petsird.BinaryPETSIRDReader,
    header: petsird.Header,
    all_detector_centers: list[np.ndarray],
    store_energy_bins: bool = True,
    verbose: bool = False,
    with_calibration_factor: bool = True,
    fname: str | None = None,
    backend: str = "auto",
    time_range_ms: tuple[int, int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """read all prompt and delayed events and convert them to LOR coordinates, TOF bins
    and efficiencies, and read all singles histograms

    backend (see utils.PETSIRD_BACKENDS): with "petsird_fast" (default for "auto" if the
    optional compiled package petsird_fast is installed and the file name fname is given),
    the time blocks of the file fname are read with the yardl-generated C++ code into
    flat arrays, which is much faster. With "petsird", the time blocks are read with
    reader (yardl-generated Python code). The results are identical.

    time_range_ms: (start, stop) in ms. Only time blocks (of all types) whose time
    interval lies within [start, stop] are used, all others are skipped (their number is
    returned as "num_skipped_time_blocks").

    Returns
    -------
    {"prompts": events, "delayeds": events, "singles": singles} where singles contains
    the singles histograms (only for singles_histogram_level ALL)
    counts : (num_singles_time_blocks, num_singles_detection_bins) int64 array, the
        singles detection bins of all module types are numbered consecutively
        (energy index over the singles energy bins)
    durations_ms : (num_singles_time_blocks,) int64 array with the durations of the
        time intervals of the singles time blocks
    and events is a dictionary with
    coords0, coords1 : (num_events, 3) float32 arrays with the ADPs of the two detection bins
    tof_bins : (num_events,) int16 array with the unsigned TOF bins (= PETSIRD tof_idx)
    effs : (num_events,) float32 array with the detection efficiencies
    energy_idx0, energy_idx1 : (num_events,) uint16 arrays with the energy bin indices
        (empty if store_energy_bins is False)
    det_bin0, det_bin1 : (num_events,) int64 arrays with the global detection bin
        indices (see detection_bin_offsets) of the two detection bins
    event_index : (num_events,) int64 array with the index of every event among all events
        of its stream and module type pair in the file (file order, i.e. also counting the
        events of time blocks outside of time_range_ms)
    The result also contains "event_time_ms": {"selected": ..., "all": ...}, the summed
    durations of the selected / all event time blocks (ms).

    The efficiencies are computed in a vectorized way following the definition in
    petsird.DetectionEfficiencies (equivalent to petsird.helpers.get_detection_efficiency).
    The TOF bins are only meaningful if all module type pairs use the same TOF binning.
    """
    scanner_info: petsird.ScannerInformation = header.scanner

    # without the file name, only the yardl Python reader can be used
    backend = resolve_petsird_backend("petsird" if backend == "auto" and fname is None else backend)
    if backend == "petsird_fast" and fname is None:
        raise ValueError("the petsird_fast backend requires the file name fname")
    if backend == "petsird_fast":
        if verbose:
            print("\nReading time blocks with petsird_fast (C++) ...")
        raw_events, singles, num_ignored, num_skipped, event_time_ms = (
            _read_time_blocks_petsird_fast(fname, scanner_info, time_range_ms)
        )
    else:
        if verbose:
            print("\nReading time blocks with the yardl-generated Python code ...")
        raw_events, singles, num_ignored, num_skipped, event_time_ms = _read_time_blocks_yardl(
            reader, scanner_info, verbose, time_range_ms
        )

    for name, num in num_ignored.items():
        warnings.warn(f"Ignored {num} {name}.", UserWarning)

    tables = detection_bin_tables(scanner_info, all_detector_centers, with_calibration_factor)
    result = {
        stream: _convert_events(raw_events[stream], tables, store_energy_bins, stream)
        for stream in ("prompts", "delayeds")
    }
    result["singles"] = singles
    result["num_skipped_time_blocks"] = num_skipped
    result["event_time_ms"] = event_time_ms
    return result


def _read_time_blocks_yardl(
    reader: petsird.BinaryPETSIRDReader,
    scanner_info: petsird.ScannerInformation,
    verbose: bool,
    time_range_ms: tuple[int, int] | None = None,
) -> tuple[dict[str, RawEvents], dict[str, np.ndarray], dict[str, int], int, dict[str, int]]:
    """raw events, singles histograms, ignored content, number of skipped time blocks and
    durations of the selected / all event time blocks (yardl-generated Python code)"""
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()
    pairs = [(m1, m2) for m1 in range(num_module_types) for m2 in range(m1 + 1)]
    chunks: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {
        stream: {pair: [] for pair in pairs} for stream in ("prompts", "delayeds")
    }

    # singles histograms (level ALL) of all time blocks
    singles_counts: list[np.ndarray] = []
    singles_durations_ms: list[int] = []

    # number of time blocks / events that are ignored
    num_ignored: dict[str, int] = {}

    def _count_ignored(name: str, num: int = 1) -> None:
        if num > 0:
            num_ignored[name] = num_ignored.get(name, 0) + num

    def _num_events(nested) -> int:
        # total number of events in nested lists of event lists
        if len(nested) == 0 or not isinstance(nested[0], list):
            return len(nested)
        return sum(_num_events(x) for x in nested)

    num_skipped = 0
    num_read_events = 0
    # number of events per stream and module type pair in all event time blocks so far
    next_index = {stream: {pair: 0 for pair in pairs} for stream in chunks}
    index_chunks: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {
        stream: {pair: [] for pair in pairs} for stream in chunks
    }
    event_time_ms = {"selected": 0, "all": 0}

    def _nested(time_block, stream):
        return time_block.value.prompt_events if stream == "prompts" else time_block.value.delayed_events

    progress = Progress("Reading time blocks (petsird)")
    for i_t, time_block in enumerate(reader.read_time_blocks()):
        progress.update(1, events=num_read_events)
        interval = time_block.value.time_interval
        is_event_block = isinstance(time_block, petsird.TimeBlock.EventTimeBlock)
        if is_event_block:
            event_time_ms["all"] += interval.stop - interval.start
        if time_range_ms is not None and (
            interval.start < time_range_ms[0] or interval.stop > time_range_ms[1]
        ):
            num_skipped += 1
            if is_event_block:
                for stream in chunks:
                    nested = _nested(time_block, stream)
                    for m1, m2 in pairs:
                        if len(nested) > 0:
                            next_index[stream][(m1, m2)] += len(nested[m1][m2])
            continue
        if isinstance(time_block, petsird.TimeBlock.WeightedEventTimeBlock):
            raise UnsupportedPETSIRDFeatureError(
                "Weighted event time blocks are not supported."
            )
        if isinstance(time_block, petsird.TimeBlock.BedMovementTimeBlock):
            if not _is_identity_transform(time_block.value.transform):
                raise UnsupportedPETSIRDFeatureError(
                    "Bed movement time blocks are not supported."
                )
            continue
        if isinstance(time_block, petsird.TimeBlock.GantryMovementTimeBlock):
            if not all(_is_identity_transform(t) for t in time_block.value.transforms):
                raise UnsupportedPETSIRDFeatureError(
                    "Gantry movement time blocks are not supported."
                )
            continue
        if isinstance(time_block, petsird.TimeBlock.DeadTimeTimeBlock):
            _count_ignored(_IGNORED_NAMES["dead_time_time_blocks"])
            continue
        if isinstance(time_block, petsird.TimeBlock.SinglesHistogramTimeBlock):
            if scanner_info.singles_histogram_level != petsird.SinglesHistogramLevelType.ALL:
                _count_ignored(_IGNORED_NAMES["singles_unit_histogram_time_blocks"])
                continue
            interval = time_block.value.time_interval
            singles_durations_ms.append(interval.stop - interval.start)
            singles_counts.append(
                np.concatenate(
                    [
                        np.asarray(h, dtype=np.int64)
                        for h in time_block.value.singles_per_detection_bin_histogram
                    ]
                )
            )
            continue
        if isinstance(time_block, petsird.TimeBlock.ExternalSignalTimeBlock):
            _count_ignored(_IGNORED_NAMES["external_signal_time_blocks"])
            continue
        if not isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
            _count_ignored(f"time blocks of type {type(time_block).__name__}")
            continue

        _count_ignored(_IGNORED_NAMES["single_events"], _num_events(time_block.value.single_events))
        _count_ignored(_IGNORED_NAMES["triple_events"], _num_events(time_block.value.triple_events))
        _count_ignored(
            _IGNORED_NAMES["quadruple_events"], _num_events(time_block.value.quadruple_events)
        )

        event_time_ms["selected"] += interval.stop - interval.start
        for stream, nested_events in (
            ("prompts", time_block.value.prompt_events),
            ("delayeds", time_block.value.delayed_events),
        ):
            if len(nested_events) == 0:
                continue
            for m1, m2 in pairs:
                events = nested_events[m1][m2]
                if len(events) == 0:
                    continue
                num_read_events += len(events)
                first = next_index[stream][(m1, m2)]
                index_chunks[stream][(m1, m2)].append(np.arange(first, first + len(events)))
                next_index[stream][(m1, m2)] += len(events)
                chunks[stream][(m1, m2)].append(
                    np.array(
                        [(e.detection_bins[0], e.detection_bins[1], e.tof_idx) for e in events],
                        dtype=np.uint32,
                    )
                )

    progress.update(0, events=num_read_events)
    progress.close()

    raw_events: dict[str, RawEvents] = {}
    for stream, per_pair in chunks.items():
        raw_events[stream] = {}
        for pair, c in per_pair.items():
            ev = np.concatenate(c) if c else np.zeros((0, 3), dtype=np.uint32)
            idx = index_chunks[stream][pair]
            index = np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, dtype=np.int64)
            raw_events[stream][pair] = (ev[:, 0], ev[:, 1], ev[:, 2], index)

    singles = {
        "counts": np.array(singles_counts, dtype=np.int64)
        if singles_counts
        else np.zeros((0, 0), dtype=np.int64),
        "durations_ms": np.array(singles_durations_ms, dtype=np.int64),
    }
    return raw_events, singles, num_ignored, num_skipped, event_time_ms


# names of the ignored content in the warnings (keys: names used by petsird_fast)
_IGNORED_NAMES = {
    "dead_time_time_blocks": "dead time time blocks (dead time is not corrected)",
    "singles_unit_histogram_time_blocks": "singles histogram time blocks (not per detection bin)",
    "external_signal_time_blocks": "external signal time blocks",
    "single_events": "single events",
    "triple_events": "triple events",
    "quadruple_events": "quadruple events",
}


def _event_index(coincidences: dict[str, np.ndarray]) -> np.ndarray:
    """file order index of all events of petsird_fast coincidences (from the offsets and
    the file order index of the first event of every event time block)"""
    offsets = coincidences["block_offsets"]
    counts = np.diff(offsets)
    shift = coincidences["block_first_index"] - offsets[:-1]
    return np.repeat(shift, counts) + np.arange(offsets[-1], dtype=np.int64)


def _read_petsird_fast_chunks(
    fname: str, time_range_ms: tuple[int, int] | None, chunk_size: int = 10
) -> dict:
    """petsird_fast.read_time_blocks() in chunks of time blocks (with progress display)"""
    progress = Progress("Reading time blocks (petsird_fast)")
    chunks = []
    num_events = 0
    for chunk in petsird_fast.iter_time_blocks(fname, chunk_size, time_range_ms):
        chunks.append(chunk)
        num_events += sum(
            a["det_bin0"].size for s in ("prompts", "delayeds") for a in chunk[s].values()
        )
        progress.update(chunk["num_time_blocks"], events=num_events)
    progress.close()
    if not chunks:
        data = petsird_fast.read_time_blocks(fname, time_range_ms)
        for stream in ("prompts", "delayeds"):
            for a in data[stream].values():
                a["event_index"] = _event_index(a)
        return data

    merged: dict = {
        "other": {},
        "num_skipped_time_blocks": 0,
        "selected_event_time_ms": 0,
        "all_event_time_ms": 0,
    }
    for stream in ("prompts", "delayeds"):
        merged[stream] = {
            pair: {
                name: np.concatenate([c[stream][pair][name] for c in chunks])
                for name in ("det_bin0", "det_bin1", "tof_idx")
            }
            for pair in chunks[0][stream]
        }
        for pair in chunks[0][stream]:
            merged[stream][pair]["event_index"] = np.concatenate(
                [_event_index(c[stream][pair]) for c in chunks]
            )
    # singles histograms of all chunks that contain singles time blocks
    with_singles = [c for c in chunks if c["singles"]["start"].size > 0]
    num_types = len(chunks[0]["singles"]["counts"])
    merged["singles"] = {
        "counts": [
            np.concatenate([c["singles"]["counts"][m] for c in with_singles])
            if with_singles
            else np.zeros((0, 0), dtype=np.uint64)
            for m in range(num_types)
        ],
        "start": np.concatenate([c["singles"]["start"] for c in chunks]),
        "stop": np.concatenate([c["singles"]["stop"] for c in chunks]),
    }
    for c in chunks:
        merged["num_skipped_time_blocks"] += c["num_skipped_time_blocks"]
        merged["selected_event_time_ms"] += c["selected_event_time_ms"]
        merged["all_event_time_ms"] += c["all_event_time_ms"]
        for name, n in c["other"].items():
            merged["other"][name] = merged["other"].get(name, 0) + n
    return merged


def _read_time_blocks_petsird_fast(
    fname: str,
    scanner_info: petsird.ScannerInformation,
    time_range_ms: tuple[int, int] | None = None,
) -> tuple[dict[str, RawEvents], dict[str, np.ndarray], dict[str, int], int, dict[str, int]]:
    """raw events, singles histograms, ignored content, number of skipped time blocks and
    durations of the selected / all event time blocks (yardl-generated C++ code via the
    optional package petsird_fast)

    The file is read in chunks of time blocks to be able to show the progress.
    """
    data = _read_petsird_fast_chunks(fname, time_range_ms)
    other = dict(data["other"])

    if other.pop("weighted_event_time_blocks", 0) > 0:
        raise UnsupportedPETSIRDFeatureError("Weighted event time blocks are not supported.")
    if other.pop("bed_movement_time_blocks", 0) > 0:
        raise UnsupportedPETSIRDFeatureError("Bed movement time blocks are not supported.")
    if other.pop("gantry_movement_time_blocks", 0) > 0:
        raise UnsupportedPETSIRDFeatureError("Gantry movement time blocks are not supported.")
    other.pop("identity_bed_movement_time_blocks", None)
    other.pop("identity_gantry_movement_time_blocks", None)
    num_ignored = {_IGNORED_NAMES.get(k, k): v for k, v in other.items() if v > 0}

    raw_events = {
        stream: {
            pair: (a["det_bin0"], a["det_bin1"], a["tof_idx"], a["event_index"])
            for pair, a in data[stream].items()
        }
        for stream in ("prompts", "delayeds")
    }

    counts = data["singles"]["counts"]
    num_singles_blocks = counts[0].shape[0] if counts else 0
    if num_singles_blocks > 0 and (
        scanner_info.singles_histogram_level != petsird.SinglesHistogramLevelType.ALL
    ):
        key = _IGNORED_NAMES["singles_unit_histogram_time_blocks"]
        num_ignored[key] = num_ignored.get(key, 0) + num_singles_blocks
        num_singles_blocks = 0
    singles = {
        "counts": np.concatenate(counts, axis=1).astype(np.int64)
        if num_singles_blocks > 0
        else np.zeros((0, 0), dtype=np.int64),
        "durations_ms": (
            data["singles"]["stop"].astype(np.int64) - data["singles"]["start"]
            if num_singles_blocks > 0
            else np.zeros(0, dtype=np.int64)
        ),
    }
    event_time_ms = {
        "selected": int(data["selected_event_time_ms"]),
        "all": int(data["all_event_time_ms"]),
    }
    return raw_events, singles, num_ignored, data["num_skipped_time_blocks"], event_time_ms


def _convert_events(
    raw_events: RawEvents,
    tables: DetectionBinTables,
    store_energy_bins: bool,
    stream: str,
) -> dict[str, np.ndarray]:
    """convert raw (detection bin 0, detection bin 1, tof_idx) events of all module type
    pairs to LOR coordinates, TOF bins, efficiencies and energy bin indices

    All quantities are looked up with 1D indices into the detection bin tables. Only
    the module index (for the SGID) and, if requested, the energy index are computed
    (see petsird.helpers.expand_detection_bin).
    """
    out: dict[str, list[np.ndarray]] = {name: [] for name in EVENT_ARRAYS}

    for (m1, m2), (det_bin_0, det_bin_1, tof_idx, event_index) in raw_events.items():
        if det_bin_0.size == 0:
            continue

        if m1 == m2 and np.any(det_bin_0 < det_bin_1):
            warnings.warn(
                f"{np.count_nonzero(det_bin_0 < det_bin_1)} {stream} of module type pair "
                f"({m1}, {m2}) are not ordered (detection_bins[0] < detection_bins[1])",
                UserWarning,
            )

        bins_per_module_1 = tables.bins_per_module[m1]
        bins_per_module_2 = tables.bins_per_module[m2]
        mod_0 = det_bin_0 // bins_per_module_1
        mod_1 = det_bin_1 // bins_per_module_2

        # (np.take is faster than fancy indexing for row gathers)
        out["coords0"].append(np.take(tables.adps[m1], det_bin_0, axis=0))
        out["coords1"].append(np.take(tables.adps[m2], det_bin_1, axis=0))
        out["tof_bins"].append(tof_idx)
        out["event_index"].append(event_index)
        out["det_bin0"].append(tables.bin_offsets[m1] + det_bin_0.astype(np.int64))
        out["det_bin1"].append(tables.bin_offsets[m2] + det_bin_1.astype(np.int64))

        sgid_lut = tables.sgid_luts[(m1, m2)]
        sgid = np.take(sgid_lut.ravel(), mod_0.astype(np.int64) * sgid_lut.shape[1] + mod_1)
        in_coinc = sgid >= 0

        if not np.all(in_coinc):
            warnings.warn(
                f"{np.count_nonzero(~in_coinc)} {stream} of module type pair ({m1}, {m2}) "
                "are in module pairs that are not in coincidence (efficiency set to 0)",
                UserWarning,
            )

        # flat index into the module pair efficiencies [sgid, bin in module 1, bin in module 2]
        mp_index = (
            np.where(in_coinc, sgid, 0).astype(np.int64) * bins_per_module_1
            + (det_bin_0 - mod_0 * bins_per_module_1)
        ) * bins_per_module_2 + (det_bin_1 - mod_1 * bins_per_module_2)

        eff = (
            tables.calibration_factor
            * np.take(tables.effs[m1], det_bin_0)
            * np.take(tables.effs[m2], det_bin_1)
            * np.take(tables.module_pair_effs[(m1, m2)], mp_index)
        )
        out["effs"].append(np.where(in_coinc, eff, 0).astype("float32"))

        if store_energy_bins:
            for name, bins, m in (("energy_idx0", det_bin_0, m1), ("energy_idx1", det_bin_1, m2)):
                num_energy_bins = tables.num_energy_bins[m]
                out[name].append(
                    bins % num_energy_bins
                    if num_energy_bins > 1
                    else np.zeros(bins.shape, dtype=np.uint16)
                )

    shapes = {"coords0": (0, 3), "coords1": (0, 3)}
    dtypes = {
        "coords0": "float32",
        "coords1": "float32",
        "tof_bins": "int16",
        "effs": "float32",
        "energy_idx0": "uint16",
        "energy_idx1": "uint16",
        "det_bin0": "int64",
        "det_bin1": "int64",
        "event_index": "int64",
    }
    return {
        name: (x[0] if len(x) == 1 else np.concatenate(x)).astype(dtypes[name], copy=False)
        if len(x) > 0
        else np.zeros(shapes.get(name, (0,)), dtype=dtypes[name])
        for name, x in out.items()
    }
