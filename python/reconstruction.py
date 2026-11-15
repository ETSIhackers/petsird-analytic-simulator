"""listmode OSEM reconstruction of PETSIRD data with parallelproj"""

import hashlib
from types import ModuleType

import numpy as np
import petsird
from parallelproj.functions import NegPoissonLogLListmode
from parallelproj.operators import (
    CompositeLinearOperator,
    ElementwiseMultiplicationOperator,
    GaussianFilterOperator,
)
from parallelproj.projectors import ListmodePETProjector
from parallelproj.tof import TOFParameters

from petsird_reader import (
    get_calibration_factor,
    get_detection_bin_efficiencies,
    get_module_pair_efficiencies,
    get_module_pair_sgid_lut,
    get_tof_parameters,
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
        get_calibration_factor(scanner_info) if with_calibration_factor else 1.0
    )

    if attenuation_image is not None:
        attenuation_image = xp.asarray(attenuation_image, dtype=xp.float32)

    if verbose:
        print("Generating sensitivity image")
    sens_img = xp.zeros(img_shape, dtype=xp.float32)

    for mod_type_1 in range(num_module_types):
        det_bin_effs_1 = get_detection_bin_efficiencies(scanner_info, mod_type_1)
        num_det_els_1, num_energy_bins_1 = det_bin_effs_1.shape[1:]

        # lower-triangular in the module types
        for mod_type_2 in range(mod_type_1 + 1):
            det_bin_effs_2 = get_detection_bin_efficiencies(scanner_info, mod_type_2)
            num_det_els_2, num_energy_bins_2 = det_bin_effs_2.shape[1:]

            sgid_lut = get_module_pair_sgid_lut(scanner_info, mod_type_1, mod_type_2)
            # (num_SGIDs, num_det_els_1, num_energy_bins_1, num_det_els_2, num_energy_bins_2)
            mp_effs = get_module_pair_efficiencies(
                scanner_info, mod_type_1, mod_type_2
            ).reshape(
                -1, num_det_els_1, num_energy_bins_1, num_det_els_2, num_energy_bins_2
            )

            tof_params = get_tof_parameters(scanner_info, mod_type_1, mod_type_2)

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


def default_image_grid(
    all_detector_centers: list[np.ndarray],
    voxel_size: tuple[float, float, float],
    img_shape: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """image shape and origin (center of voxel [0,0,0]) for a reconstruction

    The image is centered at the scanner iso center (mean of all ADPs of the first
    module type). If no image shape is given, the transaxial FOV covers 53% of the
    extent of the ADPs and the axial FOV (the axis whose extent differs most from the
    others) all ADPs (with an odd number of voxels).
    """
    adps = all_detector_centers[0].reshape(-1, 3)

    if img_shape is None:
        scanner_bbox = adps.max(0) - adps.min(0)
        i_ax = int(
            np.argmin(
                np.array(
                    [
                        np.abs(scanner_bbox[1] - scanner_bbox[2]),
                        np.abs(scanner_bbox[0] - scanner_bbox[2]),
                        np.abs(scanner_bbox[0] - scanner_bbox[1]),
                    ]
                )
            )
        )
        shape = (0.53 * scanner_bbox / np.array(voxel_size)).astype(int)
        shape[i_ax] = int(scanner_bbox[i_ax] / voxel_size[i_ax])
        if shape[i_ax] % 2 == 0:
            shape[i_ax] += 1
        img_shape = tuple(shape.tolist())

    iso_center = adps.mean(0)
    img_origin = iso_center - 0.5 * (np.asarray(img_shape) - 1) * np.asarray(voxel_size)
    return tuple(img_shape), img_origin


def recon_params_hash(
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    tof: bool,
    att_img: np.ndarray | None,
    fwhm_mm: float,
) -> str:
    """short hash of all parameters that affect the sensitivity image (used in file names)"""
    hash_obj = hashlib.sha256()
    hash_obj.update(str(img_shape).encode())
    hash_obj.update(f"{voxel_size[0]:.6f},{voxel_size[1]:.6f},{voxel_size[2]:.6f}".encode())
    hash_obj.update(str(tof).encode())
    hash_obj.update(f"{fwhm_mm:.6f}".encode())
    hash_obj.update(att_img.tobytes() if att_img is not None else b"no_attenuation")
    return hash_obj.hexdigest()[:6]


def sensitivity_image(
    scanner_info: petsird.ScannerInformation,
    all_detector_centers: list[np.ndarray],
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    img_origin,
    res_model: GaussianFilterOperator,
    tof: bool,
    xp: ModuleType = np,
    attenuation_image: np.ndarray | None = None,
    verbose: bool = False,
):
    """sensitivity image G^T B^T (effs x att) incl. the adjoint of the resolution model G"""
    return res_model.adjoint(
        backproject_efficiencies(
            scanner_info,
            all_detector_centers,
            img_shape,
            voxel_size,
            img_origin,
            tof=tof,
            verbose=verbose,
            xp=xp,
            attenuation_image=attenuation_image,
        )
    )


def lm_osem(
    coords0: np.ndarray,
    coords1: np.ndarray,
    tof_bins: np.ndarray,
    effs: np.ndarray,
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    img_origin,
    res_model: GaussianFilterOperator,
    sens_img,
    num_epochs: int,
    num_subsets: int,
    eps: float,
    tof_params: TOFParameters | None = None,
    att_img: np.ndarray | None = None,
    xp: ModuleType = np,
):
    """listmode OSEM reconstruction

    Forward model of the expected counts of an event e:
      ybar_e = eff_e * att_e * (P G x)_e + eps
    with the detection efficiency eff_e, the attenuation factor att_e, the (TOF) LM
    projector P and the image-based resolution model G.
    Without additive contamination (randoms, scatter), the multiplicative factors
    cancel in the OSEM update, but they are needed for an additive term like eps
    (and contaminations in the future). eps avoids divisions by zero for events with
    zero expected counts (e.g. when the image is 0 along their LOR), see
    parallelproj.functions.NegPoissonLogLListmode.

    The events are split into subsets by taking every num_subsets-th event.
    OSEM update: x <- x - (x / sens_subset) * grad_subset(x)
                    = (x / sens_subset) * A_subset^T (1 / (A_subset x + eps))
    Voxels with zero sensitivity are not "seen" by the scanner and set to 0.
    """
    subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

    effs = xp.asarray(effs, dtype="float32")
    if att_img is not None:
        att_img = xp.asarray(att_img, dtype=xp.float32)

    # negative Poisson log-likelihood of every subset
    subset_objectives: list[NegPoissonLogLListmode] = []
    for sl in subset_slices:
        lm_proj = ListmodePETProjector(
            xp.asarray(coords0[sl, :]).copy(),
            xp.asarray(coords1[sl, :]).copy(),
            img_shape,
            voxel_size,
            img_origin=img_origin,
        )

        # multiplicative factors of all events: efficiencies x attenuation factors
        # (non-TOF line integrals through the attenuation image)
        mult_factors = effs[sl]
        if att_img is not None:
            mult_factors = mult_factors * xp.exp(-lm_proj(att_img))

        if tof_params is not None:
            lm_proj.tof_parameters = tof_params
            lm_proj.event_tofbins = xp.asarray(tof_bins[sl], dtype=xp.int16).copy()
            lm_proj.tof = True

        subset_objectives.append(
            NegPoissonLogLListmode(
                CompositeLinearOperator(
                    [ElementwiseMultiplicationOperator(mult_factors), lm_proj, res_model]
                ),
                # the sensitivity image of a subset is (approx.) sens_img / num_subsets
                sensitivity_image=sens_img / num_subsets,
                contamination_list=xp.zeros(int(mult_factors.shape[0]), dtype=xp.float32),
                contamination_sinogram_sum=0.0,
                eps=eps,
            )
        )

    step_scale = xp.where(sens_img > 0, num_subsets / xp.where(sens_img > 0, sens_img, 1), 0)
    recon = xp.where(sens_img > 0, xp.ones(img_shape, dtype="float32"), 0)

    for i_epoch in range(num_epochs):
        for i_subset, objective in enumerate(subset_objectives):
            print(
                f"it {(i_epoch + 1):03} / {num_epochs:03}, ss {(i_subset + 1):03} / {num_subsets:03}",
                end="\r",
            )
            recon = recon - recon * step_scale * objective.gradient(recon)
            # remove negative values due to floating point rounding
            recon = xp.clip(recon, 0, None)
    print("")
    return recon
