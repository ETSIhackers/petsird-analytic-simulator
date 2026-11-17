"""TOF histo-images: events placed at (or within) their TOF bin along their LOR

For TOF data, the most likely emission position of an event is (approximately) the
center of its TOF bin along its LOR. Accumulating these positions of all events in an
image gives a "histo-image" (a TOF backprojection with a point-like TOF kernel), which
is useful as a quick look at the data and to check the TOF convention of a file.

PETSIRD TOF bins are defined for x = (t1 - t2) * c / 2 (mm), where t1 / t2 are the
arrival times at the first / second detection bin. A positive x means that the photon
arrived first at the second detection bin, i.e. that the emission happened closer to
the second detection bin. The position of an event with TOF bin center x is therefore

    p = (a0 + a1) / 2 + x * (a1 - a0) / |a1 - a0|

with the average detection points (ADPs) a0 / a1 of the first / second detection bin.

The histo-image is calculated with the TOF listmode back projector of parallelproj
(LOR from a0 to a1, which places TOF bin i at (i - (N - 1) / 2) * bin width from the LOR
center towards a1, as in the reconstruction) with a narrow TOF kernel:
- "center": point-like kernel at the TOF bin center, emulated by subdividing every TOF
  bin into K narrow bins (TOF bin i -> i * K) such that the bin centers are unchanged
- "uniform": uniform kernel over the TOF bin (box of one bin width)
Both kernels are convolved with a narrow Gaussian (default sigma: half a voxel). This
avoids that a kernel narrower than the distance of the image planes along the LOR (up
to sqrt(3) voxels) misses all planes (parallelproj normalizes the kernel over the
sampled planes).
"""

import math

import numpy as np
from array_api_compat import array_namespace, device
from parallelproj.projectors import ListmodePETProjector
from parallelproj.tof import TOFParameters


def tof_bin_centers(tof_bin_edges: np.ndarray) -> np.ndarray:
    """centers (mm) of TOF bins given by their edges"""
    edges = np.asarray(tof_bin_edges, dtype=np.float64)
    return 0.5 * (edges[1:] + edges[:-1])


def tof_bin_center_positions(
    coords0: np.ndarray,
    coords1: np.ndarray,
    tof_bins: np.ndarray,
    tof_bin_edges: np.ndarray,
) -> np.ndarray:
    """positions (num_events, 3) of the centers of the TOF bins of events along their LORs

    coords0, coords1: ADPs of the first / second detection bin of the events (PETSIRD
    order, see module docstring), tof_bins: PETSIRD tof_idx of the events
    (reference implementation of the PETSIRD TOF convention, used in the tests)
    """
    c0 = np.asarray(coords0, dtype=np.float64)
    c1 = np.asarray(coords1, dtype=np.float64)
    direction = c1 - c0
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    x = tof_bin_centers(tof_bin_edges)[np.asarray(tof_bins, dtype=np.int64)]
    return 0.5 * (c0 + c1) + x[:, None] * direction


def tof_histo_image(
    coords0,
    coords1,
    tof_bins,
    tof_parameters: TOFParameters,
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    img_origin,
    weights=None,
    tof_position: str = "center",
    sigma_mm: float | None = None,
):
    """TOF histo-image (array of the namespace of coords0) of listmode events

    coords0, coords1: ADPs of the first / second detection bin of the events (PETSIRD
    order), tof_bins: PETSIRD tof_idx of the events, tof_parameters: parallelproj TOF
    parameters of the data (petsird_reader.get_tof_parameters: equidistant, symmetric
    TOF bins), img_origin: center of voxel [0, 0, 0], weights: event weights (default 1)
    tof_position: "center" (point-like kernel at the TOF bin center) or "uniform"
    (uniform within the TOF bin), see module docstring
    sigma_mm: sigma of the Gaussian that the kernel is convolved with (default: half of
    the smallest voxel size)

    Every event inside of the image contributes its weight to the image (the image is
    in counts).
    """
    xp = array_namespace(coords0)
    dev = device(coords0)
    if tof_position not in ("center", "uniform"):
        raise ValueError(f"unknown tof_position {tof_position}")
    sigma = 0.5 * min(voxel_size) if sigma_mm is None else float(sigma_mm)
    width = float(tof_parameters.tofbin_width)
    # subdivision of the TOF bins (center): narrow bins of at most sigma / 4
    k = max(1, math.ceil(4 * width / sigma)) if tof_position == "center" else 1
    num_bins = (tof_parameters.num_tofbins - 1) * k + 1
    if num_bins > np.iinfo(np.int16).max:
        raise ValueError(f"too many subdivided TOF bins ({num_bins}), increase sigma_mm")

    proj = ListmodePETProjector(
        xp.asarray(coords0, dtype=xp.float32, device=dev),
        xp.asarray(coords1, dtype=xp.float32, device=dev),
        tuple(int(n) for n in img_shape),
        tuple(float(v) for v in voxel_size),
        img_origin=xp.asarray(img_origin, dtype=xp.float32, device=dev),
    )
    proj.tof_parameters = TOFParameters(
        num_tofbins=num_bins, tofbin_width=width / k, sigma_tof=sigma, num_sigmas=3.0
    )
    proj.event_tofbins = xp.astype(xp.asarray(tof_bins, device=dev), xp.int16) * k
    proj.tof = True
    if weights is None:
        weights = xp.ones(proj.num_events, dtype=xp.float32, device=dev)
    # the TOF LM back projection deposits (tof bin width) x weight per event
    return proj.adjoint(xp.asarray(weights, dtype=xp.float32, device=dev)) / (width / k)
