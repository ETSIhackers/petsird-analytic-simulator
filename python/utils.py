from importlib.metadata import version

# raise an error if petsird version is not at least 0.13.1
petsird_version = tuple(map(int, version("petsird").split(".")))
if petsird_version < (0, 13, 1):
    raise ImportError(
        f"petsird version {petsird_version} is not supported, please install petsird >= 0.13.1"
    )


import warnings
from types import ModuleType

import numpy as np
from parallelproj.projectors import ListmodePETProjector
from parallelproj.tof import TOFParameters

import petsird
from petsird.helpers.geometry import mult_transforms, transform_BoxShape
from petsird.helpers.plot_scanner import draw_BoxShape

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


def get_all_detector_centers(
    scanner_geometry: petsird.ScannerGeometry, ax=None
) -> list[np.ndarray]:
    """average detection points (ADPs) of all detecting elements of all modules

    Returns a list with one array per module type of shape (num_modules, num_det_els, 3).
    For a given module type, module number and detecting element number, the ADP is
    all_det_el_centers[module_type][module_number, detecting_element_number, :]

    By definition, the ADP of a detecting element is (0,0,0) in the coordinate system
    of its box shape, such that the ADP in scanner coordinates is given by the
    translation part of the combined module and detecting element transform.
    This is a vectorized version of petsird.helpers.geometry.get_ADP().

    If a matplotlib 3D axis is given, the boxes of all detecting elements are drawn.
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

        if ax is not None:
            for mod_transform in rep_module.transforms:
                for transform in det_els.transforms:
                    draw_BoxShape(
                        ax,
                        transform_BoxShape(
                            mult_transforms([mod_transform, transform]),
                            det_els.object.shape,
                        ),
                    )

    return all_det_el_centers


def _module_type_info(
    scanner_info: petsird.ScannerInformation, module_type: int
) -> tuple[int, int, int]:
    """number of modules, detecting elements per module and energy bins of a module type"""
    rep_module = scanner_info.scanner_geometry.replicated_modules[module_type]
    num_modules = len(rep_module.transforms)
    num_det_els = len(rep_module.object.detecting_elements.transforms)
    num_energy_bins = scanner_info.event_energy_bin_edges[module_type].number_of_bins()
    return num_modules, num_det_els, num_energy_bins


def _get_detection_bin_efficiencies(
    scanner_info: petsird.ScannerInformation, module_type: int
) -> np.ndarray:
    """detection bin efficiencies of a module type as array of shape (num_modules, num_det_els, num_energy_bins)

    Missing (or empty) efficiencies are considered to be 1 (see petsird.DetectionEfficiencies).
    """
    num_modules, num_det_els, num_energy_bins = _module_type_info(
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


def _get_module_pair_sgid_lut(
    scanner_info: petsird.ScannerInformation, module_type_1: int, module_type_2: int
) -> np.ndarray:
    """dense (num_modules_1, num_modules_2) array with the SGIDs of all module pairs

    The PETSIRD SGID LUT is lower-triangular for equal module types. Entries outside
    the lower triangle are set to -1 (not in coincidence), such that every module pair
    is contained exactly once.

    If no SGID LUT is given, all module pairs (excluding a module with itself)
    are assumed to be in coincidence with SGID 0.
    """
    num_modules_1 = _module_type_info(scanner_info, module_type_1)[0]
    num_modules_2 = _module_type_info(scanner_info, module_type_2)[0]
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


def _get_module_pair_efficiencies(
    scanner_info: petsird.ScannerInformation, module_type_1: int, module_type_2: int
) -> np.ndarray:
    """module pair efficiencies as array of shape (num_SGIDs, num_det_bins_in_module_1, num_det_bins_in_module_2)

    If no module pair efficiencies are given, they are considered to be 1 (for a single SGID 0).
    """
    _, num_det_els_1, num_energy_bins_1 = _module_type_info(scanner_info, module_type_1)
    _, num_det_els_2, num_energy_bins_2 = _module_type_info(scanner_info, module_type_2)
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


def _get_calibration_factor(scanner_info: petsird.ScannerInformation) -> float:
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


def _get_tof_parameters(
    scanner_info: petsird.ScannerInformation, module_type_1: int, module_type_2: int
) -> TOFParameters:
    """parallelproj TOF parameters for a pair of module types (module_type_2 <= module_type_1)"""
    tof_bin_edges = scanner_info.tof_bin_edges[module_type_1][module_type_2].edges
    tofbin_width = float(tof_bin_edges[1] - tof_bin_edges[0])

    # raise an error if tof_bin_edges are non equidistant (up to 0.1%)
    if not np.allclose(np.diff(tof_bin_edges), tofbin_width, rtol=0.001):
        raise ValueError(
            f"TOF bin edges for module types {module_type_1} and {module_type_2} are not equidistant."
        )

    # TOF bins have to be symmetric around 0, since parallelproj assumes that the
    # central TOF bin is centered at the LOR center
    if not np.isclose(tof_bin_edges[0], -tof_bin_edges[-1], atol=1e-3 * tofbin_width):
        raise ValueError(
            f"TOF bin edges for module types {module_type_1} and {module_type_2} are not symmetric around 0."
        )

    return TOFParameters(
        num_tofbins=tof_bin_edges.size - 1,
        tofbin_width=tofbin_width,
        # TOF resolution is stored as FWHM in mm
        sigma_tof=float(scanner_info.tof_resolution[module_type_1][module_type_2])
        / 2.35,
    )


def backproject_efficiencies(
    scanner_info: petsird.ScannerInformation,
    all_detector_centers: list[np.ndarray],
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    img_origin: np.ndarray,  # xp.array
    tof: bool = False,
    verbose: bool = False,
    xp: ModuleType = np,
    attenuation_image: np.ndarray | None = None,
    with_calibration_factor: bool = True,
) -> np.ndarray:
    """sensitivity image: back projection of the detection efficiencies of all LORs

    All module pairs (of all module type pairs) that are in coincidence (SGID >= 0)
    are back projected exactly once. For TOF, all TOF bins are back projected.
    The efficiencies of all energy bin combinations of a LOR are summed, since the
    geometry (and the attenuation) of a LOR does not depend on the energy bins.
    """

    scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry
    num_module_types = scanner_geom.number_of_module_types()

    calibration_factor = (
        _get_calibration_factor(scanner_info) if with_calibration_factor else 1.0
    )

    if attenuation_image is not None:
        attenuation_image = xp.asarray(attenuation_image, dtype=xp.float32)

    if verbose:
        print("Generating sensitivity image")
    sens_img = xp.zeros(img_shape, dtype=xp.float32)

    for mod_type_1 in range(num_module_types):
        det_bin_effs_1 = _get_detection_bin_efficiencies(scanner_info, mod_type_1)
        num_det_els_1, num_energy_bins_1 = det_bin_effs_1.shape[1:]

        # lower-triangular in the module types
        for mod_type_2 in range(mod_type_1 + 1):
            det_bin_effs_2 = _get_detection_bin_efficiencies(scanner_info, mod_type_2)
            num_det_els_2, num_energy_bins_2 = det_bin_effs_2.shape[1:]

            sgid_lut = _get_module_pair_sgid_lut(scanner_info, mod_type_1, mod_type_2)
            # (num_SGIDs, num_det_els_1, num_energy_bins_1, num_det_els_2, num_energy_bins_2)
            mp_effs = _get_module_pair_efficiencies(
                scanner_info, mod_type_1, mod_type_2
            ).reshape(
                -1, num_det_els_1, num_energy_bins_1, num_det_els_2, num_energy_bins_2
            )

            tof_params = _get_tof_parameters(scanner_info, mod_type_1, mod_type_2)

            if verbose:
                print(
                    f"Module type {mod_type_1} with {sgid_lut.shape[0]} modules vs. "
                    f"{mod_type_2} with {sgid_lut.shape[1]} modules"
                )

            for i_mod_1 in range(sgid_lut.shape[0]):
                mods_2 = np.nonzero(sgid_lut[i_mod_1] >= 0)[0]

                if mods_2.size == 0:
                    continue

                if verbose:
                    print(
                        f"  backprojecting LORs of module ({mod_type_1}, {i_mod_1}) "
                        f"and {mods_2.size} modules of type {mod_type_2}"
                    )

                # LOR end points: all detecting elements of module 1 vs all detecting
                # elements of all modules 2 in coincidence
                # shape (num_mods_2, num_det_els_1, num_det_els_2, 3)
                start_coords = np.broadcast_to(
                    all_detector_centers[mod_type_1][i_mod_1][None, :, None, :],
                    (mods_2.size, num_det_els_1, num_det_els_2, 3),
                )
                end_coords = np.broadcast_to(
                    all_detector_centers[mod_type_2][mods_2][:, None, :, :],
                    (mods_2.size, num_det_els_1, num_det_els_2, 3),
                )

                # LOR efficiencies summed over all energy bin combinations
                # shape (num_mods_2, num_det_els_1, num_det_els_2)
                lor_effs = calibration_factor * np.einsum(
                    "ae,mbf,maebf->mab",
                    det_bin_effs_1[i_mod_1],
                    det_bin_effs_2[mods_2],
                    mp_effs[sgid_lut[i_mod_1, mods_2]],
                )

                proj = ListmodePETProjector(
                    xp.asarray(np.ascontiguousarray(start_coords.reshape(-1, 3))),
                    xp.asarray(np.ascontiguousarray(end_coords.reshape(-1, 3))),
                    img_shape,
                    voxel_size,
                    img_origin=img_origin,
                )

                values = xp.asarray(lor_effs.reshape(-1), dtype=xp.float32)

                # assume the same attenuation for all energy bins
                if attenuation_image is not None:
                    values = values * xp.exp(-proj(attenuation_image))

                if tof:
                    proj.tof_parameters = tof_params
                    for tofbin in range(tof_params.num_tofbins):
                        proj.event_tofbins = xp.full(
                            proj.num_events, tofbin, dtype=xp.int16
                        )
                        proj.tof = True
                        sens_img += proj.adjoint(values)
                else:
                    sens_img += proj.adjoint(values)

    return sens_img


def read_listmode_prompt_events(
    reader: petsird.BinaryPETSIRDReader,
    header: petsird.Header,
    all_detector_centers: list[np.ndarray],
    store_energy_bins: bool = True,
    unity_effs: bool = False,
    verbose: bool = False,
    flip_tofbin_sign: bool = False,
    with_calibration_factor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """read all prompt events and convert them to LOR coordinates, TOF bins and efficiencies

    Returns
    -------
    coords0, coords1 : (num_events, 3) float32 arrays with the ADPs of the two detection bins
    tof_bins : (num_events,) int16 array with the unsigned TOF bins (= PETSIRD tof_idx)
    effs : (num_events,) float32 array with the detection efficiencies
    energy_idx0, energy_idx1 : (num_events,) uint16 arrays with the energy bin indices
        (empty if store_energy_bins is False)

    The efficiencies are computed in a vectorized way following the definition in
    petsird.DetectionEfficiencies (equivalent to petsird.helpers.get_detection_efficiency).
    The TOF bins are only meaningful if all module type pairs use the same TOF binning.
    """

    scanner_info: petsird.ScannerInformation = header.scanner
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()

    calibration_factor = (
        _get_calibration_factor(scanner_info) if with_calibration_factor else 1.0
    )

    # collect the raw detection bins and TOF indices for all module type pairs
    raw_events: dict[tuple[int, int], list[np.ndarray]] = {
        (m1, m2): [] for m1 in range(num_module_types) for m2 in range(m1 + 1)
    }

    if verbose:
        print("\nReading prompt events from time blocks ...")

    for i_t, time_block in enumerate(reader.read_time_blocks()):
        if isinstance(time_block, petsird.TimeBlock.WeightedEventTimeBlock):
            raise NotImplementedError("Weighted event time blocks are not supported.")

        if not isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
            continue

        if verbose:
            interval = time_block.value.time_interval
            print(
                f"Processing time block {i_t} with time interval {interval.start} ... {interval.stop}"
            )

        prompt_events = time_block.value.prompt_events
        for m1, m2 in raw_events:
            events = prompt_events[m1][m2]
            if len(events) == 0:
                continue
            raw_events[(m1, m2)].append(
                np.array(
                    [(e.detection_bins[0], e.detection_bins[1], e.tof_idx) for e in events],
                    dtype=np.int64,
                )
            )

    coords0 = []
    coords1 = []
    tof_bins = []
    effs = []
    energy_idx0 = []
    energy_idx1 = []

    for (m1, m2), chunks in raw_events.items():
        if len(chunks) == 0:
            continue

        ev = np.concatenate(chunks)
        det_bin_0, det_bin_1, tof_idx = ev[:, 0], ev[:, 1], ev[:, 2]

        if m1 == m2 and np.any(det_bin_0 < det_bin_1):
            warnings.warn(
                f"{np.count_nonzero(det_bin_0 < det_bin_1)} events of module type pair "
                f"({m1}, {m2}) are not ordered (detection_bins[0] < detection_bins[1])",
                UserWarning,
            )

        # expand detection bins (see petsird.helpers.expand_detection_bins)
        _, num_det_els_1, num_energy_bins_1 = _module_type_info(scanner_info, m1)
        _, num_det_els_2, num_energy_bins_2 = _module_type_info(scanner_info, m2)

        mod_0 = det_bin_0 // (num_det_els_1 * num_energy_bins_1)
        el_0 = (det_bin_0 // num_energy_bins_1) % num_det_els_1
        en_0 = det_bin_0 % num_energy_bins_1
        mod_1 = det_bin_1 // (num_det_els_2 * num_energy_bins_2)
        el_1 = (det_bin_1 // num_energy_bins_2) % num_det_els_2
        en_1 = det_bin_1 % num_energy_bins_2

        coords0.append(all_detector_centers[m1][mod_0, el_0])
        coords1.append(all_detector_centers[m2][mod_1, el_1])

        num_tofbins = scanner_info.tof_bin_edges[m1][m2].number_of_bins()
        if flip_tofbin_sign:
            tof_bins.append(num_tofbins - 1 - tof_idx)
        else:
            tof_bins.append(tof_idx)

        if unity_effs:
            effs.append(np.ones(det_bin_0.size, dtype="float32"))
        else:
            det_bin_effs_1 = _get_detection_bin_efficiencies(scanner_info, m1).ravel()
            det_bin_effs_2 = _get_detection_bin_efficiencies(scanner_info, m2).ravel()
            sgid_lut = _get_module_pair_sgid_lut(scanner_info, m1, m2)
            mp_effs = _get_module_pair_efficiencies(scanner_info, m1, m2)

            sgid = sgid_lut[mod_0, mod_1]
            in_coinc = sgid >= 0

            if not np.all(in_coinc):
                warnings.warn(
                    f"{np.count_nonzero(~in_coinc)} events of module type pair ({m1}, {m2}) "
                    "are in module pairs that are not in coincidence (efficiency set to 0)",
                    UserWarning,
                )

            eff = (
                calibration_factor
                * det_bin_effs_1[det_bin_0]
                * det_bin_effs_2[det_bin_1]
                * mp_effs[
                    np.where(in_coinc, sgid, 0),
                    det_bin_0 % (num_det_els_1 * num_energy_bins_1),
                    det_bin_1 % (num_det_els_2 * num_energy_bins_2),
                ]
            )
            effs.append(np.where(in_coinc, eff, 0).astype("float32"))

        if store_energy_bins:
            energy_idx0.append(en_0)
            energy_idx1.append(en_1)

    def _concat(x: list[np.ndarray], shape: tuple, dtype: str) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(shape, dtype=dtype)
        return np.concatenate(x).astype(dtype)

    return (
        _concat(coords0, (0, 3), "float32"),
        _concat(coords1, (0, 3), "float32"),
        _concat(tof_bins, (0,), "int16"),
        _concat(effs, (0,), "float32"),
        _concat(energy_idx0, (0,), "uint16"),
        _concat(energy_idx1, (0,), "uint16"),
    )
