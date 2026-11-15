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
from scanner import ScannerConfig, create_petsird_scanner_geometry, crystal_bulk_material
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
    assert proj.adjointness_test(dtype=np.float32, rtol=1e-4)
    return proj


def attenuation_factors(
    lor_desc: EqualBlockPETLORDescriptor,
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    att_img: np.ndarray,
) -> np.ndarray:
    """attenuation factors exp(-int mu) of all LORs (non-TOF line integrals through the
    attenuation image in 1/mm, image centered at (0,0,0))

    Returns an array of shape (num_block_pairs, num_lors_per_block_pair).
    """
    return np.exp(-EqualBlockPETProjector(lor_desc, img_shape, voxel_size)(att_img))


def create_forward_model(
    proj: EqualBlockPETProjector,
    res_model: GaussianFilterOperator,
    mult_factors: np.ndarray,
) -> CompositeLinearOperator:
    """forward model diag(m) P G of the expected TOF histogram of an activity image

    G: image-based resolution model, P: (TOF) sinogram projector, m: non-TOF
    multiplicative factors of all LORs (e.g. efficiencies x attenuation factors),
    shape (num_block_pairs, num_lors_per_block_pair), which are broadcasted along the
    TOF axis (zero-copy view).
    The sensitivity image is the back projection of ones: op.adjoint(ones).
    """
    return CompositeLinearOperator(
        [
            ElementwiseMultiplicationOperator(
                np.broadcast_to(mult_factors[..., None], proj.out_shape)
            ),
            proj,
            res_model,
        ]
    )


def randoms_expectation(
    lor_effs: np.ndarray, num_tofbins: int, total_expected_randoms: float
) -> tuple[np.ndarray, float]:
    """noise-free randoms TOF histogram with an (expected) total of total_expected_randoms

    The randoms are constant for all LORs and TOF bins, except for LORs with a detecting
    element with zero efficiency ("dead" crystal, which can not detect any single) where
    they are 0.

    Parameters
    ----------
    lor_effs : efficiencies of all LORs, shape (num_block_pairs, num_lors_per_block_pair)
    num_tofbins : number of TOF bins
    total_expected_randoms : expected total number of randoms

    Returns
    -------
    expected_randoms : array of shape (num_block_pairs, num_lors_per_block_pair, num_tofbins)
    expected_randoms_per_bin : expected randoms per LOR (with non-zero efficiency) and TOF bin
    """
    valid = lor_effs > 0
    expected_randoms_per_bin = float(total_expected_randoms) / (np.count_nonzero(valid) * num_tofbins)
    expected_randoms = np.zeros(lor_effs.shape + (num_tofbins,), dtype=np.float32)
    expected_randoms[valid] = expected_randoms_per_bin
    return expected_randoms, expected_randoms_per_bin


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
    counts_histogram: np.ndarray, lor_desc: EqualBlockPETLORDescriptor, num_tofbins: int
) -> ListmodeEvents:
    """convert a TOF histogram (num_block_pairs, num_lors_per_block_pair, num_tofbins)
    of counts into (randomly shuffled) listmode events"""
    num_det_els = lor_desc.num_lorendpoints_per_block
    start_el_arr, end_el_arr = np.meshgrid(
        np.arange(num_det_els), np.arange(num_det_els), indexing="ij"
    )
    start_el_arr, end_el_arr = start_el_arr.ravel(), end_el_arr.ravel()

    num_events = counts_histogram.sum()
    start_module = np.zeros(num_events, dtype="uint32")
    start_el = np.zeros(num_events, dtype="uint32")
    end_module = np.zeros(num_events, dtype="uint32")
    end_el = np.zeros(num_events, dtype="uint32")
    signed_tof_bin = np.zeros(num_events, dtype="int32")

    i_event = 0
    for ibp, block_pair in enumerate(lor_desc.all_block_pairs):
        for it, tof_bin in enumerate(np.arange(num_tofbins) - num_tofbins // 2):
            counts = counts_histogram[ibp, :, it]
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
    store_delayeds: bool = False,
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
        # materials referenced by the (detecting) volumes of the scanner geometry
        bulk_materials=[crystal_bulk_material()],
        tof_bin_edges=[[tof_bin_edges]],
        tof_resolution=[[tof_fwhm_mm]],  # FWHM in mm
        event_energy_bin_edges=[energy_bin_edges],
        energy_resolution_at_511=[0.11],  # as fraction of 511
        detection_efficiencies=to_petsird_detection_efficiencies(
            efficiency_model, config, num_energy_bins
        ),
    )
    # we store prompt and (optionally) delayed coincidences, all other event types
    # (singles, triples, quadruples) keep the default policy NONE.
    # All pairs of singles within the coincidence window are coincidences
    # (multiples are stored as all pairs; the simulation itself creates no multiples)
    scanner.prompt_event_policy = petsird.CoincidencePolicy.MULTIPLES_AS_ALL_COINCIDENCES
    if store_delayeds:
        scanner.delayed_event_policy = petsird.CoincidencePolicy.MULTIPLES_AS_ALL_COINCIDENCES

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
    delayeds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """write the coincidences (split evenly into time blocks) into a PETSIRD file

    det_bin_0, det_bin_1, tof_idx: prompt coincidences (see to_petsird_coincidences)
    delayeds: (det_bin_0, det_bin_1, tof_idx) of the delayed coincidences (optional,
    the header must have a delayed_event_policy != NONE)
    """

    def to_events(chunk):
        return [
            petsird.CoincidenceEvent(detection_bins=[int(x[0]), int(x[1])], tof_idx=int(x[2]))
            for x in chunk
        ]

    chunks = np.array_split(np.array([det_bin_0, det_bin_1, tof_idx]).T, num_time_blocks)
    if delayeds is not None:
        delayed_chunks = np.array_split(np.array(delayeds).T, num_time_blocks)

    with petsird.BinaryPETSIRDWriter(str(fname)) as writer:
        writer.write_header(header)
        for i_t, chunk in enumerate(chunks):
            print(f"Writing time block {i_t + 1}/{len(chunks)}")
            delayed_events = [[to_events(delayed_chunks[i_t])]] if delayeds is not None else []
            writer.write_time_blocks(
                (
                    petsird.TimeBlock.EventTimeBlock(
                        petsird.EventTimeBlock(
                            prompt_events=[[to_events(chunk)]],
                            delayed_events=delayed_events,
                            time_interval=petsird.TimeInterval(
                                start=i_t * time_block_duration_ms,
                                stop=(i_t + 1) * time_block_duration_ms,
                            ),
                        )
                    ),
                )
            )
