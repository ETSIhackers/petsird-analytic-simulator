from importlib.metadata import version
import warnings

# raise an error if petsird version is not at least 0.7.2
petsird_version = tuple(map(int, version("petsird").split(".")))
if petsird_version < (0, 7, 2):
    raise ImportError(
        f"petsird version {petsird_version} is not supported, please install petsird >= 0.7.2"
    )


import numpy as np

import parallelproj

import petsird
from petsird.helpers import (
    expand_detection_bin,
    get_detection_efficiency,
)

import petsird.helpers.geometry
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_BoxShape(ax, box: petsird.BoxShape) -> None:
    vertices = np.array([c.c for c in box.corners])
    edges = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [4, 7, 3, 0]],
    ]
    box_poly = Poly3DCollection(edges, alpha=0.1, linewidths=0.25, edgecolors="r")
    ax.add_collection3d(box_poly)


def get_all_detector_centers(
    scanner_geometry: petsird.ScannerGeometry, ax=None
) -> list[np.ndarray]:
    # a list containing the center of all detecting elements for all modules
    # every element of the list corresponds to one module type
    # for every module type, we have an numpy array of shape (num_modules, num_det_els, 3)
    # for a given module type, module number and detector el number, we can access the center of the detector element with
    # all_det_el_centers[module_type][module_number, detector_el_number, :]

    all_det_el_centers = []

    # draw all crystals
    for rep_module in scanner_geometry.replicated_modules:
        det_els = rep_module.object.detecting_elements
        det_el_centers = np.zeros(
            (len(rep_module.transforms), len(det_els.transforms), 3)
        )
        for i_mod, mod_transform in enumerate(rep_module.transforms):

            for i_det_el, transform in enumerate(det_els.transforms):
                transformed_boxshape = (
                    petsird.helpers.geometry.transform_BoxShape(
                        petsird.helpers.geometry.mult_transforms(
                            [mod_transform, transform]
                        ),
                        det_els.object.shape,
                    ),
                )[0]

                transformed_boxshape_vertices = np.array(
                    [c.c for c in transformed_boxshape.corners]
                )

                det_el_centers[i_mod, i_det_el, :] = transformed_boxshape_vertices.mean(
                    axis=0
                )

                if ax is not None:
                    draw_BoxShape(
                        ax,
                        transformed_boxshape,
                    )
        all_det_el_centers.append(det_el_centers)

    return all_det_el_centers


def backproject_efficiencies(
    scanner_info: petsird.ScannerInformation,
    all_detector_centers: list[np.ndarray],
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    tof: bool = False,
    verbose: bool = False,
) -> np.ndarray:

    scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry
    num_replicated_modules = scanner_geom.number_of_replicated_modules()

    # read detection element / module pair efficiencies

    # get the dection bin efficiencies for all module types
    # index via: det_bin_effs = all_detection_bin_effs[rep_mod_type]
    # which returns a 1D array of shape (num_detection_bins,) = (num_det_els_in_module * num_energy_bins,)
    all_detection_bin_effs: list[petsird.DetectionBinEfficiencies] | None = (
        scanner_info.detection_efficiencies.detection_bin_efficiencies
    )

    # get the symmetry group ID LUTs for all module types
    # index via: 2D_SGID_LUT = all_module_pair_sgidlut[rep_mod_type 1][rep_mod_type 2]
    # which returns a 2D array of shape (num_modules, num_modules)
    all_module_pair_sgidluts: list[list[petsird.ModulePairSGIDLUT]] | None = (
        scanner_info.detection_efficiencies.module_pair_sgidlut
    )

    # get all module pair efficiencies vectors
    # index via: mod_pair_effs = module_pair_efficiencies_vectors[rep_mod_type 1][rep_mod_type 2][sgid]
    # which returns a 2D array of shape (num_det_els, num_det_els)
    all_module_pair_efficiency_vectors: (
        list[list[list[petsird.ModulePairEfficiencies]]] | None
    ) = scanner_info.detection_efficiencies.module_pair_efficiencies_vectors

    # number of modules for every module type
    all_num_modules: list[int] = [
        len(x.transforms) for x in scanner_geom.replicated_modules
    ]
    # number of detecting elements for every module type
    all_num_det_els: list[int] = [
        x.object.detecting_elements.number_of_objects()
        for x in scanner_geom.replicated_modules
    ]

    if all_detection_bin_effs is None:
        all_detection_bin_effs = [
            np.ones(x * y, dtype="float32")
            for (x, y) in zip(all_num_modules, all_num_det_els)
        ]
        warnings.warn(
            "No detection bin efficiencies found in scanner information - assuming all detection efficiencies are 1.0.",
        )

    if all_module_pair_sgidluts is None:
        # no SGID LUTs are giving, assume that all modules pairs are in SGID 0

        all_module_pair_sgidluts = []
        for num_mod1 in all_num_modules:
            tmp_list = []
            for num_mod2 in all_num_modules:
                lut = np.full((num_mod1, num_mod2), -1, dtype="int")
                for i in range(num_mod1):
                    for j in range(i + 1, num_mod2):
                        lut[i, j] = 0

                tmp_list.append(lut)
            all_module_pair_sgidluts.append(tmp_list)

        warnings.warn(
            "No module pair SGID LUTs found in scanner information. Asumming all module pairs are in SGID 0.",
            UserWarning,
        )

    if all_module_pair_efficiency_vectors is None:

        all_module_pair_efficiency_vectors = []
        for num_det_els1 in all_num_det_els:
            tmp_list = []
            for num_det_els2 in all_num_det_els:
                # create a dummy efficiency vector with all ones
                # the shape is (num_det_els1, num_energy_bins, num_det_els2, num_energy_bins)
                # where num_energy_bins is the number of energy bins for the respective module type
                tmp_list.append(
                    [
                        petsird.ModulePairEfficiencies(
                            values=np.ones(
                                (num_det_els1, num_det_els2), dtype="float32"
                            )
                        )
                    ]
                )
            all_module_pair_efficiency_vectors.append(tmp_list)

        warnings.warn(
            "No module pair efficiencies vectors found in scanner information. Assuming all ones.",
            UserWarning,
        )

    # %%
    # generate the sensitivity image

    if verbose:
        print("Generating sensitivity image")
    sens_img = np.zeros(img_shape, dtype="float32")

    for mod_type_1 in range(num_replicated_modules):
        num_modules_1 = len(scanner_geom.replicated_modules[mod_type_1].transforms)

        energy_bin_edges_1 = scanner_info.event_energy_bin_edges[mod_type_1].edges
        num_energy_bins_1 = energy_bin_edges_1.size - 1

        det_bin_effs_1 = all_detection_bin_effs[mod_type_1].reshape(
            num_modules_1, -1, num_energy_bins_1
        )

        for mod_type_2 in range(num_replicated_modules):
            num_modules_2 = len(scanner_geom.replicated_modules[mod_type_2].transforms)

            energy_bin_edges_2 = scanner_info.event_energy_bin_edges[mod_type_2].edges
            num_energy_bins_2 = energy_bin_edges_2.size - 1
            det_bin_effs_2 = all_detection_bin_effs[mod_type_2].reshape(
                num_modules_2, -1, num_energy_bins_2
            )

            if verbose:
                print(
                    f"Module type {mod_type_1} with {num_modules_1} modules vs. {mod_type_2} and {num_modules_2} modules"
                )

            # sigma TOF (mm) for module type combination
            sigma_tof = scanner_info.tof_resolution[mod_type_1][mod_type_2] / 2.35
            tof_bin_edges = scanner_info.tof_bin_edges[mod_type_1][mod_type_2].edges
            num_tofbins = tof_bin_edges.size - 1
            tofbin_width = float(tof_bin_edges[1] - tof_bin_edges[0])

            # raise an error if tof_bin_edges are non equidistant (up to 0.1%)
            if not np.allclose(
                np.diff(tof_bin_edges), tof_bin_edges[1] - tof_bin_edges[0], rtol=0.001
            ):
                raise ValueError(
                    f"TOF bin edges for module types {mod_type_1} and {mod_type_2} are not equidistant."
                )

            for i_mod_1 in range(num_modules_1):
                for i_mod_2 in range(num_modules_2):

                    sgid = all_module_pair_sgidluts[mod_type_1][mod_type_2][
                        i_mod_1, i_mod_2
                    ]

                    # if the symmetry group ID (sgid) is non-negative, the module pair is in coincidence
                    if sgid >= 0:
                        if verbose:
                            print(
                                f"  Module pair ({mod_type_1}, {i_mod_1}) vs. ({mod_type_2}, {i_mod_2}) with SGID {sgid}"
                            )

                        # 2D array containg the 3 coordinates of all detecting elements the start module
                        start_det_coords = all_detector_centers[mod_type_1][
                            i_mod_1, :, :
                        ]
                        # 2D array containg the 3 coordinates of all detecting elements the end module
                        end_det_coords = all_detector_centers[mod_type_2][i_mod_2, :, :]

                        # 2D array of start coordinates of all LORs connecting all detecting elements
                        # of the start module with all detecting elements of the end module
                        start_coords = np.repeat(
                            start_det_coords, start_det_coords.shape[0], axis=0
                        )

                        # 2D array of end coordinates of all LORs connecting all detecting elements
                        # of the start module with all detecting elements of the end module
                        end_coords = np.tile(
                            end_det_coords, (end_det_coords.shape[0], 1)
                        )

                        # setup a LM projector that we use for the sensitivity image calculation
                        proj = parallelproj.ListmodePETProjector(
                            start_coords, end_coords, img_shape, voxel_size
                        )

                        if tof:
                            proj.tof_parameters = parallelproj.TOFParameters(
                                num_tofbins=num_tofbins,
                                tofbin_width=tofbin_width,
                                sigma_tof=sigma_tof,
                            )

                        # 2D array of shape (num_detection_bins, num_detection_bins) =
                        # (num_det_els * num_energy_bins, num_det_els * num_energy_bins)
                        module_pair_efficiencies = all_module_pair_efficiency_vectors[
                            mod_type_1
                        ][mod_type_2][sgid].values

                        module_pair_efficiencies = module_pair_efficiencies.reshape(
                            module_pair_efficiencies.shape[0] // num_energy_bins_1,
                            num_energy_bins_1,
                            module_pair_efficiencies.shape[0] // num_energy_bins_2,
                            num_energy_bins_2,
                        )

                        for i_e_1 in range(num_energy_bins_1):
                            for i_e_2 in range(num_energy_bins_2):
                                if verbose:
                                    print(f"    Energy bin pair ({i_e_1}, {i_e_2})")

                                # get the detection bin efficiencies for the start module
                                # 1D array of shape (num_det_els,)
                                start_det_bin_effs = det_bin_effs_1[i_mod_1, :, i_e_1]
                                # get the detection bin efficiencies for the end module
                                # 1D array of shape (num_det_els,)
                                end_det_bin_effs = det_bin_effs_2[i_mod_2, :, i_e_2]

                                # (non-TOF) sensitivity values to be back-projected
                                ##########
                                # in case of modeled attenuation, multiply them as well
                                ##########
                                to_be_back_projected = (
                                    np.outer(
                                        start_det_bin_effs,
                                        end_det_bin_effs,  # multiplied start and end det els. effs for all LORs
                                    ).ravel()
                                    * module_pair_efficiencies[
                                        :,
                                        i_e_1,
                                        :,
                                        i_e_2,  # module pair effs for current module pair and energy bin pair
                                    ].ravel()
                                )

                                if tof:
                                    for signed_tofbin in np.arange(
                                        -(num_tofbins // 2), num_tofbins // 2 + 1
                                    ):
                                        proj.event_tofbins = np.full(
                                            start_coords.shape[0],
                                            signed_tofbin,
                                            dtype="int32",
                                        )
                                        proj.tof = True
                                        sens_img += proj.adjoint(to_be_back_projected)
                                else:
                                    proj.tof = False
                                    sens_img += proj.adjoint(to_be_back_projected)

                        # clean up the projector (stores many coordinates ...)
                        del proj

    return sens_img


def read_listmode_prompt_events(
    reader: petsird.BinaryPETSIRDReader,
    header: petsird.Header,
    all_detector_centers: list[np.ndarray],
    store_energy_bins: bool = True,
    unity_effs: bool = False,
    verbose: bool = False,
    flip_tofbin_sign: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    scanner_info: petsird.ScannerInformation = header.scanner
    scanner_geom: petsird.ScannerGeometry = scanner_info.scanner_geometry
    num_replicated_modules = scanner_geom.number_of_replicated_modules()

    if verbose:
        print("\nReading prompt events from time blocks ...")

    i_t = 0

    ## list of dictionaries, each dictionary contains the prompt detection bins for a time block for all module type and energy bin combinations
    # all_prompt_detection_bins: list[dict[tuple[int, int], np.ndarray]] = []

    coords0 = []
    coords1 = []
    effs = []
    signed_tof_bins = []
    energy_idx0 = []
    energy_idx1 = []

    for time_block in reader.read_time_blocks():
        if isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
            start_time = time_block.value.time_interval.start
            stop_time = time_block.value.time_interval.stop
            if verbose:
                print(
                    f"Processing time block {i_t} with time interval {start_time} ... {stop_time}"
                )

            # time_block_prompt_detection_bins = dict()

            for mtype0 in range(num_replicated_modules):
                for mtype1 in range(num_replicated_modules):
                    tof_bin_edges = scanner_info.tof_bin_edges[mtype0][mtype1].edges
                    num_tofbins = tof_bin_edges.size - 1

                    for event in time_block.value.prompt_events[mtype0][mtype1]:
                        expanded_det_bin0 = expand_detection_bin(
                            scanner_info, mtype0, event.detection_bins[0]
                        )
                        expanded_det_bin1 = expand_detection_bin(
                            scanner_info, mtype1, event.detection_bins[1]
                        )

                        coords0.append(
                            all_detector_centers[mtype0][
                                expanded_det_bin0.module_index,
                                expanded_det_bin0.element_index,
                            ]
                        )
                        coords1.append(
                            all_detector_centers[mtype1][
                                expanded_det_bin1.module_index,
                                expanded_det_bin1.element_index,
                            ]
                        )

                        if flip_tofbin_sign:
                            signed_tof_bins.append(-(event.tof_idx - num_tofbins // 2))
                        else:
                            signed_tof_bins.append(event.tof_idx - num_tofbins // 2)

                        if unity_effs:
                            effs.append(1.0)
                        else:
                            effs.append(
                                get_detection_efficiency(
                                    scanner_info,
                                    petsird.TypeOfModulePair((mtype0, mtype1)),
                                    event,
                                )
                            )

                            if store_energy_bins:
                                energy_idx0.append(expanded_det_bin0.energy_index)
                                energy_idx1.append(expanded_det_bin1.energy_index)

            # all_prompt_detection_bins.append(time_block_prompt_detection_bins)
            i_t += 1

    # convert lists to numpy arrays
    coords0 = np.array(coords0, dtype="float32")
    coords1 = np.array(coords1, dtype="float32")
    signed_tof_bins = np.array(signed_tof_bins, dtype="int16")
    effs = np.array(effs, dtype="float32")
    energy_idx0 = np.array(energy_idx0, dtype="uint16")
    energy_idx1 = np.array(energy_idx1, dtype="uint16")

    return coords0, coords1, signed_tof_bins, effs, energy_idx0, energy_idx1
