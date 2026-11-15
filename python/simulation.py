"""analytic simulation of (true, unscattered) TOF coincidences and writing them to PETSIRD"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import petsird
from parallelproj.operators import (
    CompositeLinearOperator,
    ElementwiseMultiplicationOperator,
    GaussianFilterOperator,
)
from parallelproj.pet_lors import EqualBlockPETLORDescriptor
from parallelproj.projectors import EqualBlockPETProjector
from parallelproj.tof import TOFParameters

from efficiencies import EfficiencyModel, to_petsird_detection_efficiencies
from scanner import ScannerConfig, create_petsird_scanner_geometry
from utils import FWHM_PER_SIGMA


def tof_parameters(
    tof_fwhm_mm: float, img_shape: tuple[int, int, int], voxel_size: tuple[float, float, float]
) -> TOFParameters:
    """TOF parameters: bin width = 0.8 sigma, odd number of bins covering the image diagonal"""
    sigma_tof = tof_fwhm_mm / FWHM_PER_SIGMA
    tofbin_width = 0.8 * sigma_tof
    num_tofbins = int(np.sqrt(2) * img_shape[0] * voxel_size[0] / tofbin_width)
    if num_tofbins % 2 == 0:
        num_tofbins += 1
    return TOFParameters(
        num_tofbins=num_tofbins, tofbin_width=tofbin_width, sigma_tof=sigma_tof, num_sigmas=3.0
    )


@dataclass
class ForwardModel:
    """expected TOF histogram of an activity image: diag(effs x att) P G

    G: image-based resolution model, P: TOF projector, effs: LOR efficiencies,
    att: attenuation factors exp(-int mu) of the LORs
    """

    proj: EqualBlockPETProjector
    res_model: GaussianFilterOperator
    lor_effs: np.ndarray
    att_factors: np.ndarray
    op: CompositeLinearOperator

    def sensitivity_image(self, with_attenuation: bool = True) -> np.ndarray:
        """back projection of ones (with or without the attenuation factors)"""
        if with_attenuation:
            return self.op.adjoint(np.ones(self.op.out_shape, dtype=np.float32))
        return self.res_model.adjoint(
            self.proj.adjoint(
                np.broadcast_to(self.lor_effs[..., None], self.proj.out_shape).astype(
                    np.float32
                )
            )
        )


def create_tof_projector(
    lor_desc: EqualBlockPETLORDescriptor,
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    tof_params: TOFParameters,
) -> EqualBlockPETProjector:
    """TOF sinogram projector (image centered at (0,0,0)) incl. an adjointness test

    Note: the adjointness test uses the global numpy random state.
    """
    proj = EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
    proj.tof_parameters = tof_params
    # the projectors require float32 arrays
    assert proj.adjointness_test(np, "cpu", dtype=np.float32, rtol=1e-4)
    return proj


def create_forward_model(
    proj: EqualBlockPETProjector,
    fwhm_mm: float,
    lor_effs: np.ndarray,
    att_img: np.ndarray,
) -> ForwardModel:
    """forward model of the simulation for a TOF projector (see create_tof_projector)"""
    lor_desc = proj.lor_descriptor
    img_shape = proj.in_shape
    voxel_size = tuple(float(v) for v in np.asarray(proj.voxel_size))

    res_model = GaussianFilterOperator(
        img_shape, sigma=fwhm_mm / (FWHM_PER_SIGMA * np.asarray(voxel_size))
    )

    # attenuation factors exp(-int mu) of all LORs (non-TOF line integrals)
    att_factors = np.exp(-EqualBlockPETProjector(lor_desc, img_shape, voxel_size)(att_img))

    # the non-TOF multiplicative factors are broadcasted along the TOF axis (zero-copy view)
    op = CompositeLinearOperator(
        [
            ElementwiseMultiplicationOperator(
                np.broadcast_to((lor_effs * att_factors)[..., None], proj.out_shape)
            ),
            proj,
            res_model,
        ]
    )
    return ForwardModel(proj, res_model, lor_effs, att_factors, op)


def histogram_mlem(fwd_op, data: np.ndarray, sens_img: np.ndarray, num_epochs: int) -> np.ndarray:
    """MLEM reconstruction of histogrammed data (reference for LM reconstructions)"""
    recon = np.ones(fwd_op.in_shape, dtype=np.float32)
    for i in range(num_epochs):
        print(f"{(i+1):03}/{num_epochs:03}", end="\r")
        exp = np.clip(fwd_op(recon), 1e-6, None)
        # data contains int64 Poisson counts -> cast back to float32
        grad = fwd_op.adjoint(((exp - data) / exp).astype(np.float32))
        recon -= (recon / sens_img) * grad
    print("")
    return recon


@dataclass
class ListmodeEvents:
    """listmode events in the parallelproj convention (start module < end module)"""

    start_module: np.ndarray
    start_el: np.ndarray
    end_module: np.ndarray
    end_el: np.ndarray
    # unsigned TOF bin (0 ... num_tofbins - 1) w.r.t. the direction start -> end
    tof_bin: np.ndarray

    @property
    def num_events(self) -> int:
        return self.start_module.shape[0]


def histogram_to_listmode(
    emission_data: np.ndarray, lor_desc: EqualBlockPETLORDescriptor, num_tofbins: int
) -> ListmodeEvents:
    """convert a TOF histogram (num_block_pairs, num_lors_per_block_pair, num_tofbins)
    of counts into (randomly shuffled) listmode events"""
    num_det_els = lor_desc.num_lorendpoints_per_block
    start_el_arr, end_el_arr = np.meshgrid(
        np.arange(num_det_els), np.arange(num_det_els), indexing="ij"
    )
    start_el_arr, end_el_arr = start_el_arr.ravel(), end_el_arr.ravel()

    num_events = emission_data.sum()
    start_module = np.zeros(num_events, dtype="uint32")
    start_el = np.zeros(num_events, dtype="uint32")
    end_module = np.zeros(num_events, dtype="uint32")
    end_el = np.zeros(num_events, dtype="uint32")
    signed_tof_bin = np.zeros(num_events, dtype="int32")

    i_event = 0
    for ibp, block_pair in enumerate(lor_desc.all_block_pairs):
        for it, tof_bin in enumerate(np.arange(num_tofbins) - num_tofbins // 2):
            counts = emission_data[ibp, :, it]
            n = counts.sum()
            inds = np.repeat(np.arange(counts.shape[0]), counts)
            sl = slice(i_event, i_event + n)
            start_module[sl] = block_pair[0]
            start_el[sl] = np.take(start_el_arr, inds)
            end_module[sl] = block_pair[1]
            end_el[sl] = np.take(end_el_arr, inds)
            signed_tof_bin[sl] = tof_bin
            i_event += n

    # shuffle the events
    inds = np.arange(num_events)
    np.random.shuffle(inds)

    return ListmodeEvents(
        start_module=start_module[inds],
        start_el=start_el[inds],
        end_module=end_module[inds],
        end_el=end_el[inds],
        tof_bin=np.asarray(signed_tof_bin[inds] + num_tofbins // 2, dtype="uint32"),
    )


def to_petsird_coincidences(
    events: ListmodeEvents, num_det_els: int, num_energy_bins: int, num_tofbins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PETSIRD detection bins and TOF indices of listmode events

    PETSIRD detection bin: energy_idx + (element_idx + module_idx * num_det_els) * num_energy_bins
    (all events are in energy bin 0).
    PETSIRD requires ordered coincidences: detection_bins[0] >= detection_bins[1]
    (for events within the same module type). For events where the detection bins are
    swapped, the TOF bin is mirrored, since PETSIRD stores (t1 - t2) * c / 2 where t1 / t2
    are the arrival times at the first / second detection bin.

    Returns (detection bin 0, detection bin 1, tof_idx)
    """
    start_det_bin = (events.start_module * num_det_els + events.start_el) * num_energy_bins
    end_det_bin = (events.end_module * num_det_els + events.end_el) * num_energy_bins

    swap = start_det_bin < end_det_bin
    det_bin_0 = np.where(swap, end_det_bin, start_det_bin)
    det_bin_1 = np.where(swap, start_det_bin, end_det_bin)
    tof_idx = np.where(swap, num_tofbins - 1 - events.tof_bin, events.tof_bin)
    return det_bin_0, det_bin_1, tof_idx


def create_petsird_header(
    config: ScannerConfig,
    efficiency_model: EfficiencyModel,
    tof_params: TOFParameters,
    tof_fwhm_mm: float,
    num_energy_bins: int = 1,
) -> petsird.Header:
    """PETSIRD header of the simulated scanner (one module type)"""
    tof_bin_edges = petsird.BinEdges(
        edges=np.linspace(
            -tof_params.num_tofbins * tof_params.tofbin_width / 2,
            tof_params.num_tofbins * tof_params.tofbin_width / 2,
            tof_params.num_tofbins + 1,
            dtype="float32",
        )
    )
    energy_bin_edges = petsird.BinEdges(
        edges=np.linspace(430, 650, num_energy_bins + 1, dtype="float32")
    )

    # all tables over module type pairs are lower-triangular (here: 1x1)
    scanner = petsird.ScannerInformation(
        model_name="PETSIRD_TEST",
        scanner_geometry=create_petsird_scanner_geometry(config),
        tof_bin_edges=[[tof_bin_edges]],
        tof_resolution=[[tof_fwhm_mm]],  # FWHM in mm
        event_energy_bin_edges=[energy_bin_edges],
        energy_resolution_at_511=[0.11],  # as fraction of 511
        detection_efficiencies=to_petsird_detection_efficiencies(
            efficiency_model, config, num_energy_bins
        ),
    )
    # we only store prompt (true) coincidences, all other event types (singles,
    # delayeds, triples, quadruples) keep the default policy NONE
    scanner.prompt_event_policy = petsird.CoincidencePolicy.REJECT_HIGHER_MULTIPLES

    exam = petsird.ExamInformation(
        modality="PT", patient=petsird.DICOMPatientInformation(patient_id="42")
    )
    return petsird.Header(exam=exam, scanner=scanner)


def write_petsird_file(
    fname: str | Path,
    header: petsird.Header,
    det_bin_0: np.ndarray,
    det_bin_1: np.ndarray,
    tof_idx: np.ndarray,
    num_time_blocks: int,
    time_block_duration_ms: int,
) -> None:
    """write the coincidences (split evenly into time blocks) into a PETSIRD file"""
    chunks = np.array_split(np.array([det_bin_0, det_bin_1, tof_idx]).T, num_time_blocks)

    with petsird.BinaryPETSIRDWriter(str(fname)) as writer:
        writer.write_header(header)
        for i_t, chunk in enumerate(chunks):
            print(f"Writing time block {i_t + 1}/{len(chunks)}")
            prompts = [
                petsird.CoincidenceEvent(detection_bins=[int(x[0]), int(x[1])], tof_idx=int(x[2]))
                for x in chunk
            ]
            writer.write_time_blocks(
                (
                    petsird.TimeBlock.EventTimeBlock(
                        petsird.EventTimeBlock(
                            prompt_events=[[prompts]],
                            time_interval=petsird.TimeInterval(
                                start=i_t * time_block_duration_ms,
                                stop=(i_t + 1) * time_block_duration_ms,
                            ),
                        )
                    ),
                )
            )


def plot_sensitivity_and_tof_profile(
    sens_img: np.ndarray,
    expected_counts: np.ndarray,
    lor_desc: EqualBlockPETLORDescriptor,
    voxel_size: tuple[float, float, float],
):
    """3 transaxial slices of the sensitivity image and the TOF profile of the LOR
    with the most expected counts (image centered at (0,0,0))"""
    import matplotlib.pyplot as plt

    img_shape = sens_img.shape
    origin = -(np.asarray(img_shape) - 1) / 2 * np.asarray(voxel_size)
    extent = [
        origin[0] - 0.5 * voxel_size[0],
        origin[0] + (img_shape[0] - 0.5) * voxel_size[0],
        origin[1] - 0.5 * voxel_size[1],
        origin[1] + (img_shape[1] - 0.5) * voxel_size[1],
    ]

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5), layout="constrained")
    vmin, vmax = float(np.min(sens_img)), float(np.max(sens_img))
    for i, sl in enumerate([img_shape[2] // 4, img_shape[2] // 2, 3 * img_shape[2] // 4]):
        im = ax[i].imshow(
            sens_img[:, :, sl].T, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap="Greys"
        )
        ax[i].set_title(
            f"sensitivity image, z = {origin[2] + sl * voxel_size[2]:.1f} mm", fontsize="small"
        )
        ax[i].set_xlabel("x (mm)")
        ax[i].set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax[:3], shrink=0.8)

    i_bp, i_lor = np.unravel_index(
        np.argmax(expected_counts.sum(-1)), expected_counts.shape[:-1]
    )
    ax[-1].plot(expected_counts[i_bp, i_lor, :], ".-")
    ax[-1].set_xlabel("TOF bin (parallelproj convention)")
    ax[-1].set_ylabel("expected counts")
    ax[-1].set_title(
        f"TOF profile of the LOR with the most counts\n"
        f"(block pair {tuple(int(b) for b in lor_desc.all_block_pairs[i_bp])}, LOR {i_lor})",
        fontsize="small",
    )
    return fig
