"""TOF histo-images: events placed at the center of their TOF bin along their LOR

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
"""

import numpy as np


def tof_bin_centers(tof_bin_edges: np.ndarray) -> np.ndarray:
    """centers (mm) of TOF bins given by their edges"""
    edges = np.asarray(tof_bin_edges, dtype=np.float64)
    return 0.5 * (edges[1:] + edges[:-1])


def tof_bin_center_positions(
    coords0: np.ndarray,
    coords1: np.ndarray,
    tof_bins: np.ndarray,
    tof_bin_edges: np.ndarray,
    uniform_jitter: np.random.Generator | None = None,
) -> np.ndarray:
    """positions (num_events, 3) of the centers of the TOF bins of events along their LORs

    coords0, coords1: ADPs of the first / second detection bin of the events (PETSIRD
    order, see module docstring), tof_bins: PETSIRD tof_idx of the events

    uniform_jitter: if a random generator is given, the events are placed at a uniformly
    distributed random position within their TOF bin instead of its center. The centers
    of all TOF bins of a LOR are anchored at the LOR midpoint, and on a ring scanner
    the midpoints of many LORs coincide (e.g. on the scanner axis). Exact centers
    therefore produce spikes / lines in the image (e.g. at the scanner center), which the
    jitter removes.
    """
    c0 = np.asarray(coords0, dtype=np.float64)
    c1 = np.asarray(coords1, dtype=np.float64)
    direction = c1 - c0
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    tof_bins = np.asarray(tof_bins, dtype=np.int64)
    x = tof_bin_centers(tof_bin_edges)[tof_bins]
    if uniform_jitter is not None:
        widths = np.diff(np.asarray(tof_bin_edges, dtype=np.float64))[tof_bins]
        x = x + uniform_jitter.uniform(-0.5, 0.5, x.shape[0]) * widths
    return 0.5 * (c0 + c1) + x[:, None] * direction


def splat(
    positions: np.ndarray,
    img_shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    img_origin: np.ndarray,
    weights: np.ndarray | None = None,
    interpolation: str = "trilinear",
) -> tuple[np.ndarray, int]:
    """accumulate (weighted) points into an image

    img_origin: center of voxel [0, 0, 0]
    interpolation: "nearest" (nearest voxel) or "trilinear" (weights distributed to the
    8 surrounding voxel centers, which reduces aliasing of the discrete TOF bin centers)

    Returns the image (float64) and the number of points (partly) outside of the image.
    """
    shape = np.asarray(img_shape)
    # continuous voxel coordinates (voxel centers at integers)
    u = (np.asarray(positions, dtype=np.float64) - np.asarray(img_origin)) / np.asarray(voxel_size)
    w = np.ones(u.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    image = np.zeros(int(np.prod(shape)), dtype=np.float64)

    if interpolation == "nearest":
        idx = np.rint(u).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < shape), axis=1)
        flat = np.ravel_multi_index(idx[inside].T, img_shape)
        image += np.bincount(flat, weights=w[inside], minlength=image.size)
        return image.reshape(img_shape), int(np.count_nonzero(~inside))

    if interpolation != "trilinear":
        raise ValueError(f"unknown interpolation {interpolation}")
    i0 = np.floor(u).astype(np.int64)
    f = u - i0
    # per axis: weights and validity of the lower / upper neighbor
    lower_ok = (i0 >= 0) & (i0 < shape)
    upper_ok = (i0 + 1 >= 0) & (i0 + 1 < shape)
    touches = np.all(lower_ok | upper_ok, axis=1)
    # points (partly) outside: a neighbor with non-zero weight is outside of the image
    num_outside = int(
        np.count_nonzero(np.any((~lower_ok & (f < 1)) | (~upper_ok & (f > 0)), axis=1))
    )
    # only points with at least one of their 8 neighbors inside of the image
    i0, f, w = i0[touches], f[touches], w[touches]
    lower_ok, upper_ok = lower_ok[touches], upper_ok[touches]
    axis_weights = [(w * (1 - f[:, 0]), w * f[:, 0]), (1 - f[:, 1], f[:, 1]), (1 - f[:, 2], f[:, 2])]
    axis_ok = [(lower_ok[:, k], upper_ok[:, k]) for k in range(3)]
    strides = (shape[1] * shape[2], shape[2], 1)
    base = i0 @ np.asarray(strides)
    for cx, cy, cz in np.ndindex(2, 2, 2):
        ok = axis_ok[0][cx] & axis_ok[1][cy] & axis_ok[2][cz]
        cw = axis_weights[0][cx] * axis_weights[1][cy] * axis_weights[2][cz]
        flat = base + (cx * strides[0] + cy * strides[1] + cz * strides[2])
        image += np.bincount(flat[ok], weights=cw[ok], minlength=image.size)
    return image.reshape(img_shape), num_outside
