"""TOF histo-image of a PETSIRD listmode file

All events of the prompt or delayed stream are placed at the center of their TOF bin
along their LOR (most likely emission position; or uniformly within their TOF bin with
--tof_position uniform) and accumulated in an image, using the TOF listmode back
projector of parallelproj with a narrow TOF kernel (see histo_images.py). This is a
fast TOF backprojection, useful as a quick look at the data and to check the TOF
convention of a file. Non-TOF data (a single TOF bin) are not supported, and the TOF
bins have to be equidistant and symmetric around 0 (as for the reconstruction).

The image is written next to the listmode file
(<stem>_tof_histo_image_<stream>_<fingerprint>[_t<start>-<stop>ms].nii, NIfTI with the
grid in the scanner coordinate system of the PETSIRD file, see image_io.py) together
with a figure of three orthogonal cuts.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import petsird

from histo_images import tof_histo_image
from petsird_reader import (
    check_header_support,
    get_all_detector_centers,
    get_tof_parameters,
    read_listmode_events,
)
from image_io import grid_from_affine, load_image, load_image_on_grid, save_image
from reconstruction import reconstruction_grid
from parallelproj.operators import GaussianFilterOperator
from parallelproj.projectors import ListmodePETProjector

from utils import FWHM_PER_SIGMA, PETSIRD_BACKENDS, lm_file_fingerprint, resolve_petsird_backend
from viewer import show_vol_cuts

parser = argparse.ArgumentParser(
    description="TOF histo-image (events at the center of their TOF bin) of a PETSIRD file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("fname", type=str, help="Path to the PETSIRD listmode file")
parser.add_argument(
    "--stream", choices=["prompts", "delayeds"], default="prompts", help="event stream to use"
)
parser.add_argument(
    "--img_shape", type=int, nargs=3, default=None, help="shape of the image (default: as 02)"
)
parser.add_argument(
    "--voxel_size",
    type=float,
    nargs=3,
    default=None,
    help="voxel size in mm (default 2 2 2; without --img_shape / --voxel_size, the grid of "
    "--attenuation_image is used if given, as in 02)",
)
parser.add_argument(
    "--tof_position",
    choices=["center", "uniform"],
    default="center",
    help="position of an event along its LOR: center of its TOF bin (point-like TOF "
    "kernel), or uniformly distributed within its TOF bin (box kernel; removes the spikes "
    "/ lines at positions where the midpoints of many LORs coincide, e.g. the scanner "
    "center). Both kernels are convolved with a Gaussian of --kernel_sigma_mm.",
)
parser.add_argument(
    "--kernel_sigma_mm",
    type=float,
    default=None,
    help="sigma of the Gaussian of the TOF kernel (default: half of the smallest voxel size)",
)
parser.add_argument(
    "--postfilter_fwhm_mm",
    type=float,
    default=0.0,
    help="FWHM (mm) of an isotropic Gaussian post filter of the histo-image (0: none); a "
    "fast way to reduce noise and the aliasing of the TOF bin centers",
)
parser.add_argument(
    "--efficiency_weighted",
    action="store_true",
    help="weight every event with 1 / detection efficiency (events with zero efficiency "
    "are ignored)",
)
parser.add_argument(
    "--time_range_ms",
    type=int,
    nargs=2,
    default=None,
    metavar=("START", "STOP"),
    help="only use the time blocks whose time interval lies within [START, STOP] (ms)",
)
parser.add_argument(
    "--petsird_backend",
    choices=PETSIRD_BACKENDS,
    default="auto",
    help="read the time blocks with petsird (yardl Python) or petsird_fast (C++)",
)
parser.add_argument(
    "--attenuation_image",
    type=str,
    default=None,
    help="attenuation image (1/mm, same grid as the histo-image) to correct every event for "
    "the attenuation of its LOR (weight 1 / exp(-line integral))",
)
parser.add_argument("--skip_plots", action="store_true", help="do not show the viewer")
args = parser.parse_args()

fname = Path(args.fname)

reader = petsird.BinaryPETSIRDReader(str(fname))
header: petsird.Header = reader.read_header()
check_header_support(header)
scanner_info = header.scanner

# (check_header_support() made sure that all module type pairs have the same TOF bins)
tof_bin_edges = np.asarray(scanner_info.tof_bin_edges[0][0].edges)
if tof_bin_edges.size - 1 < 2:
    raise ValueError("TOF histo-images require TOF data (more than one TOF bin)")

all_detector_centers = get_all_detector_centers(scanner_info.scanner_geometry)
# image grid as in 02: --img_shape / --voxel_size, else the grid of the attenuation
# image, else the default grid
att_grid = None
if args.attenuation_image is not None:
    att_data, att_affine = load_image(args.attenuation_image)
    try:
        att_grid = grid_from_affine(att_data.shape, att_affine)
    except ValueError:
        att_grid = None
grid = reconstruction_grid(
    all_detector_centers,
    tuple(args.img_shape) if args.img_shape else None,
    tuple(args.voxel_size) if args.voxel_size else None,
    att_grid,
)
img_shape, voxel_size, img_origin = grid.shape, grid.voxel_size, np.asarray(grid.origin)
print(f"Image grid: {grid}")
att_img = None
if args.attenuation_image is not None:
    att_img = load_image_on_grid(args.attenuation_image, grid, "attenuation image")

backend = resolve_petsird_backend(args.petsird_backend)
time_range = tuple(args.time_range_ms) if args.time_range_ms is not None else None
all_events = read_listmode_events(
    reader, header, all_detector_centers, fname=str(fname), backend=backend,
    time_range_ms=time_range,
)
events = all_events[args.stream]
num_events = events["coords0"].shape[0]
if num_events == 0:
    raise ValueError(f"no {args.stream} in the file (or in the selected time range)")

weights = None
if args.efficiency_weighted or args.attenuation_image is not None:
    weights = np.ones(events["coords0"].shape[0])
if att_img is not None:
    # attenuation factors exp(-int mu) of the (non-TOF) LORs of all events
    att_proj = ListmodePETProjector(
        np.ascontiguousarray(events["coords0"]),
        np.ascontiguousarray(events["coords1"]),
        img_shape,
        voxel_size,
        img_origin=np.asarray(img_origin, dtype=np.float32),
    )
    weights = weights / np.exp(-att_proj(att_img))
    print(f"Attenuation correction: mean correction factor {weights.mean():.3f}")
if args.efficiency_weighted:
    valid = events["effs"] > 0
    events = {k: v[valid] for k, v in events.items()}
    weights = weights[valid] / events["effs"]

t0 = time.perf_counter()
image = np.asarray(
    tof_histo_image(
        np.ascontiguousarray(events["coords0"]),
        np.ascontiguousarray(events["coords1"]),
        events["tof_bins"],
        get_tof_parameters(scanner_info, 0, 0),
        img_shape,
        voxel_size,
        img_origin,
        weights=weights,
        tof_position=args.tof_position,
        sigma_mm=args.kernel_sigma_mm,
    )
)
t_image = time.perf_counter() - t0
if args.postfilter_fwhm_mm > 0:
    image = GaussianFilterOperator(
        img_shape, sigma=args.postfilter_fwhm_mm / (FWHM_PER_SIGMA * np.asarray(voxel_size))
    )(image)
print(
    f"TOF histo-image of {num_events:,} {args.stream} in {t_image:.1f} s (sum of the image "
    f"{image.sum():,.0f}; the rest is outside of the image {img_shape} with voxel size "
    f"{voxel_size} mm)"
)

tag = lm_file_fingerprint(fname)
if args.tof_position == "uniform":
    tag += "_uniform"
if args.postfilter_fwhm_mm > 0:
    tag += f"_fwhm{args.postfilter_fwhm_mm:g}mm"
if args.attenuation_image is not None:
    tag += "_ac"
if args.efficiency_weighted:
    tag += "_effs"
if time_range is not None:
    tag += f"_t{time_range[0]}-{time_range[1]}ms"
opath = fname.with_name(f"{fname.stem}_tof_histo_image_{args.stream}_{tag}.nii")
save_image(opath, image, grid, f"TOF histo-image ({args.stream})")
print(f"TOF histo-image saved to {opath}")

# three orthogonal cuts through the center of the image
fig, axs = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
center = [s // 2 for s in img_shape]
for ax, cut, label in zip(
    axs,
    (image[:, :, center[2]].T, image[:, center[1], :].T, image[center[0], :, :].T),
    ("z", "y", "x"),
):
    im = ax.imshow(cut, origin="lower", cmap="Greys")
    ax.set_title(f"central {label} slice")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(
    f"TOF histo-image ({args.stream}, {args.tof_position} TOF position"
    + (f", {args.postfilter_fwhm_mm:g} mm FWHM post filter" if args.postfilter_fwhm_mm > 0 else "")
    + f") of {fname.name}"
)
fig.savefig(opath.with_suffix(".png"), dpi=150)

if not args.skip_plots:
    viewer = show_vol_cuts(image, voxel_size=voxel_size, fig_title="TOF histo-image")
    plt.show()
