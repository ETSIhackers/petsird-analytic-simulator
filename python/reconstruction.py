"""reconstruction with parallelproj

Two kinds of reconstructions:
- LM OSEM of PETSIRD data (the reference implementation of this repository):
  sensitivity_image() from the PETSIRD scanner information (geometry and detection
  efficiency tables) and lm_osem() of the prompt events read with
  petsird_reader.read_listmode_events() (LOR end points, TOF bins, efficiencies)
- histogram_mlem(): MLEM of histogrammed (sinogram) data with a parallelproj forward
  operator, used by the simulator to create a reference reconstruction with the exact
  simulation model (independent of PETSIRD)

Both use the same EM update (em_update) written as a preconditioned gradient step of
the negative Poisson log-likelihood (as in the parallelproj examples).
"""

import hashlib
from types import ModuleType

import numpy as np
import petsird
from array_api_compat import array_namespace
from parallelproj import Array
from parallelproj.functions import (
    C1Function,
    C2AffineObjective,
    NegPoissonLogL,
    NegPoissonLogLListmode,
)
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
    """back projection of the detection efficiencies of all LORs of a PETSIRD scanner

    The LORs and their efficiencies are derived from the PETSIRD scanner information
    (ADPs of the detecting elements and detection efficiency tables).
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
    """sensitivity image of a LM reconstruction of PETSIRD data

    G^T B^T (effs x att): back projection B^T of the efficiencies (and attenuation
    factors) of all LORs of the PETSIRD scanner (see backproject_efficiencies) followed
    by the adjoint of the image-based resolution model G.
    """
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


def constant_randoms_contamination(
    randoms_per_lor_and_tof_bin: float, num_events: int, num_tofbins: int, tof: bool
) -> np.ndarray:
    """additive contamination of all events for randoms that are constant for all LORs
    (with non-zero efficiency) and TOF bins

    The expected counts of an event in a LM reconstruction refer to its TOF bin (TOF) or
    its LOR (non-TOF), i.e. the randoms of a TOF bin or the sum of the randoms over all
    TOF bins of the LOR.
    """
    randoms_per_event = randoms_per_lor_and_tof_bin * (1 if tof else num_tofbins)
    return np.full(num_events, randoms_per_event, dtype=np.float32)


def em_update(x: Array, objective: C1Function, sens_img: Array, mask: Array | None = None) -> Array:
    """one EM update written as preconditioned gradient step

    x_new = x - (x / sens_img) * grad f(x)
          = (x / sens_img) * A^T (y / (A x + s))     for a negative Poisson log-likelihood f

    Works for any objective providing a gradient, e.g. NegPoissonLogLListmode (listmode)
    or C2AffineObjective(NegPoissonLogL) (histogrammed data).
    Voxels outside the mask (e.g. zero sensitivity) are set to 0.
    """
    xp = array_namespace(x)
    if mask is None:
        d = x / sens_img
    else:
        d = xp.where(mask, x / xp.where(mask, sens_img, 1), 0)
    # clip removes negative values due to floating point rounding
    return xp.clip(x - d * objective.gradient(x), 0, None)


def create_lm_subset_objectives(
    coords0: np.ndarray,
    coords1: np.ndarray,
    tof_bins: np.ndarray,
    effs: np.ndarray,
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    img_origin,
    res_model: GaussianFilterOperator,
    sens_img,
    num_subsets: int,
    eps: float,
    tof_params: TOFParameters | None = None,
    att_img: np.ndarray | None = None,
    contamination_list: np.ndarray | None = None,
    contamination_sum: float = 0.0,
    xp: ModuleType = np,
) -> list[NegPoissonLogLListmode]:
    """negative Poisson log-likelihoods of subsets of (pre-read) PETSIRD listmode events

    The events (from petsird_reader.read_listmode_events) are split into subsets
    by taking every num_subsets-th event. The forward model of the expected counts of
    an event e is
      ybar_e = eff_e * att_e * (P G x)_e + s_e
    with the detection efficiency eff_e, the attenuation factor att_e (non-TOF line
    integral through att_img), the (TOF) LM projector P, the image-based resolution
    model G and the additive contamination s_e (e.g. randoms, scatter; 0 if not given).
    The sensitivity image of a subset is approximated by sens_img / num_subsets.

    eps is added to the expected counts of all events (see
    parallelproj.functions.NegPoissonLogLListmode). It is only needed if the
    contamination is not strictly positive, to avoid divisions by zero for events with
    zero expected counts (e.g. when the image is 0 along their LOR); otherwise use
    eps = 0 (exact log-likelihood).
    """
    num_events = coords0.shape[0]
    if contamination_list is None:
        contamination_list = np.zeros(num_events, dtype=np.float32)
    contamination_list = xp.asarray(contamination_list, dtype=xp.float32)
    if eps <= 0 and float(xp.min(contamination_list)) <= 0:
        raise ValueError(
            "eps > 0 is required if the contamination is not strictly positive "
            "(expected counts of events can be 0)"
        )

    effs = xp.asarray(effs, dtype="float32")
    if att_img is not None:
        att_img = xp.asarray(att_img, dtype=xp.float32)

    objectives = []
    for k in range(num_subsets):
        sl = slice(k, None, num_subsets)
        lm_proj = ListmodePETProjector(
            xp.asarray(coords0[sl, :]).copy(),
            xp.asarray(coords1[sl, :]).copy(),
            img_shape,
            voxel_size,
            img_origin=img_origin,
        )

        # multiplicative factors of all events: efficiencies x attenuation factors
        # (non-TOF line integrals, i.e. computed before enabling TOF)
        mult_factors = effs[sl]
        if att_img is not None:
            mult_factors = mult_factors * xp.exp(-lm_proj(att_img))

        if tof_params is not None:
            lm_proj.tof_parameters = tof_params
            lm_proj.event_tofbins = xp.asarray(tof_bins[sl], dtype=xp.int16).copy()
            lm_proj.tof = True

        objectives.append(
            NegPoissonLogLListmode(
                CompositeLinearOperator(
                    [ElementwiseMultiplicationOperator(mult_factors), lm_proj, res_model]
                ),
                sensitivity_image=sens_img / num_subsets,
                contamination_list=contamination_list[sl],
                contamination_sinogram_sum=contamination_sum / num_subsets,
                eps=eps,
            )
        )
    return objectives


def lm_osem(
    subset_objectives: list[NegPoissonLogLListmode],
    sens_img,
    num_epochs: int,
    xp: ModuleType = np,
):
    """LM OSEM reconstruction of PETSIRD data (see create_lm_subset_objectives)

    Starts from a uniform image (0 outside the voxels with non-zero sensitivity)
    and cycles num_epochs times through all subsets using em_update with the
    subset sensitivity image sens_img / num_subsets.
    """
    num_subsets = len(subset_objectives)
    mask = sens_img > 0
    sens_subset = sens_img / num_subsets
    recon = xp.where(mask, xp.ones(sens_img.shape, dtype="float32"), 0)

    for i_epoch in range(num_epochs):
        for i_subset, objective in enumerate(subset_objectives):
            print(
                f"it {(i_epoch + 1):03} / {num_epochs:03}, ss {(i_subset + 1):03} / {num_subsets:03}",
                end="\r",
            )
            recon = em_update(recon, objective, sens_subset, mask)
    print("")
    return recon


def histogram_mlem(
    fwd_op,
    data: np.ndarray,
    sens_img: np.ndarray,
    num_epochs: int,
    contamination: np.ndarray | None = None,
) -> np.ndarray:
    """MLEM of histogrammed (sinogram) data with a parallelproj forward operator

    Used by the simulator as reference reconstruction with the exact simulation model
    (independent of PETSIRD). Uses the "safe epsilon" mode of NegPoissonLogL to handle
    bins with counts but zero expectation.
    """
    objective = C2AffineObjective(
        NegPoissonLogL(np.asarray(data, dtype=np.float32)), fwd_op, contamination
    )
    mask = sens_img > 0
    recon = np.where(mask, np.ones(fwd_op.in_shape, dtype=np.float32), 0)
    for i in range(num_epochs):
        print(f"MLEM epoch {(i + 1):03} / {num_epochs:03}", end="\r")
        recon = em_update(recon, objective, sens_img, mask)
    print("")
    return recon
