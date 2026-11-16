"""listmode OSEM reconstruction of PETSIRD (TOF) listmode data with parallelproj

The building blocks are defined in
- petsird_reader.py: reading and interpreting the PETSIRD data
- reconstruction.py: image grid, sensitivity image, LM OSEM
- randoms.py: randoms estimation from the delayed coincidences
All derived files (cached sensitivity image, backprojections, reconstructions, figures)
are written next to the listmode file and are tagged with a fingerprint of the listmode
file and a hash of the reconstruction parameters. The events are read from the listmode
file in every run (fast if the optional package petsird_fast is installed).
"""

import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parallelproj
import petsird
from parallelproj.operators import GaussianFilterOperator
from parallelproj.projectors import ListmodePETProjector

from geometry_plots import plot_petsird_geometry, read_first_events
from petsird_reader import (
    check_header_support,
    get_all_detector_centers,
    get_num_data_bins,
    get_tof_parameters,
    petsird_fast_available,
    UnsupportedPETSIRDFeatureError,
    read_listmode_events,
)
from randoms import (
    casey_singles,
    coincidence_window_ns,
    randoms_from_singles_of_events,
    randoms_from_singles_total,
    delayeds_fan_sums,
    detection_bin_fans,
    ml_singles,
    randoms_of_events,
)
from reconstruction import (
    create_lm_subset_objectives,
    default_image_grid,
    lm_osem,
    recon_params_hash,
    sensitivity_image,
)
from utils import FWHM_PER_SIGMA, lm_file_fingerprint, load_or_compute
from viewer import show_vol_cuts

# for the parallelproj recon we can use cupy as array backend if available
# otherwise we fall back to numpy
try:
    import cupy as xp
except ModuleNotFoundError:
    import numpy as xp

print(f"Using {xp.__name__} for parallelproj reconstructions")

# %%
# parse the command line

parser = argparse.ArgumentParser(
    description="LM OSEM reconstruction of PETSIRD listmode data",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("fname", type=str, help="Path to the PETSIRD listmode file")
parser.add_argument(
    "--img_shape", type=int, nargs=3, default=None, help="Shape of the image to be reconstructed"
)
parser.add_argument(
    "--voxel_size", type=float, nargs=3, default=[2.0, 2.0, 2.0], help="Voxel size in mm"
)
parser.add_argument(
    "--fwhm_mm",
    type=float,
    default=2.5,
    help="FWHM in mm for Gaussian filter for resolution model",
)
parser.add_argument("--num_epochs", type=int, default=5, help="Number of OSEM epochs")
parser.add_argument("--num_subsets", type=int, default=20, help="Number of OSEM subsets")
parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
parser.add_argument(
    "--unity_sens_img",
    action="store_true",
    help="Whether to skip sensitivity image calculation and use unity image",
)
parser.add_argument("--non-tof", action="store_true", help="Whether to disable TOF in recon")
parser.add_argument(
    "--attenuation_image", type=str, default=None, help="Path to attenuation image"
)
randoms_group = parser.add_mutually_exclusive_group()
randoms_group.add_argument(
    "--randoms_from_delayeds",
    choices=["ml", "casey"],
    default=None,
    help="randoms correction with the randoms estimated from the delayed coincidences: "
    "maximum likelihood (ml) or Casey-type fan-sum (casey) estimate of the singles",
)
randoms_group.add_argument(
    "--randoms_from_singles",
    action="store_true",
    help="randoms correction with the randoms calculated from the singles histograms "
    "(2 tau S_a S_b, the coincidence window 2 tau is assumed to be the range of the TOF bins)",
)
randoms_group.add_argument(
    "--randoms_expectation",
    type=str,
    default=None,
    help="Path to the true (noise-free) expected randoms of all prompts written by the "
    "simulator (prompts_randoms_expectation.npy) used for an 'oracle' randoms correction",
)
parser.add_argument(
    "--randoms_ml_iterations",
    type=int,
    default=10,
    help="number of iterations of the ML randoms (singles) estimation (initialized with "
    "the Casey-type estimate)",
)
args = parser.parse_args()

fname = Path(args.fname)
voxel_size = tuple(args.voxel_size)
tof = not args.non_tof

# %%
# read the PETSIRD header

reader = petsird.BinaryPETSIRDReader(str(fname))
header: petsird.Header = reader.read_header()
# raise errors / warnings for PETSIRD features that are not supported
check_header_support(header)
scanner_info: petsird.ScannerInformation = header.scanner
print(
    f"Scanner with {scanner_info.scanner_geometry.number_of_replicated_modules()} "
    "types of replicated modules."
)

# all cached / derived files are tagged with a fingerprint of the listmode file
# such that they are not reused for a different file with the same name
lm_fingerprint = lm_file_fingerprint(fname)
print(f"Listmode file fingerprint: {lm_fingerprint}")


def output_path(kind: str, suffix: str = ".npy", tag: str | None = None) -> Path:
    """path of a derived file next to the listmode file"""
    return fname.with_name(f"{fname.stem}_{kind}_{tag or lm_fingerprint}{suffix}")


# %%
# average detection points (ADPs) of all detecting elements

all_detector_centers = get_all_detector_centers(scanner_info.scanner_geometry)

# %%
# image grid, resolution model, attenuation image

img_shape, img_origin = default_image_grid(
    all_detector_centers,
    voxel_size,
    tuple(args.img_shape) if args.img_shape is not None else None,
)
img_origin = xp.asarray(img_origin)

if args.verbose:
    print(f"Image shape: {img_shape}, origin: {img_origin}, voxel size: {voxel_size}")

res_model = GaussianFilterOperator(
    img_shape, sigma=args.fwhm_mm / (FWHM_PER_SIGMA * np.asarray(voxel_size))
)

att_img: np.ndarray | None = None
if args.attenuation_image is not None:
    print(f"Loading attenuation image from {args.attenuation_image}")
    att_img = np.load(args.attenuation_image)
    if att_img.shape != img_shape:
        raise ValueError(
            f"Attenuation image shape {att_img.shape} does not match image shape {img_shape}"
        )

# tag of all files that depend on the reconstruction parameters
params_tag = (
    f"{lm_fingerprint}_{recon_params_hash(img_shape, voxel_size, tof, att_img, args.fwhm_mm)}"
)

# %%
# sensitivity image

if args.unity_sens_img:
    print("Using ones as sensitivity image ...")
    sens_img = xp.ones(img_shape, dtype="float32")
else:
    sens_img = load_or_compute(
        output_path("sens", tag=params_tag),
        compute=lambda: sensitivity_image(
            scanner_info,
            all_detector_centers,
            img_shape,
            voxel_size,
            img_origin,
            res_model,
            tof,
            xp=xp,
            attenuation_image=att_img,
            verbose=args.verbose,
        ),
        save=xp.save,
        load=xp.load,
        description="sensitivity image",
    )

# %%
# prompt and delayed events (LOR end points, TOF bins, efficiencies)

# the time blocks are read with the optional compiled package petsird_fast if installed
print(
    "Reading time blocks with "
    + ("petsird_fast (C++)" if petsird_fast_available() else "the yardl-generated Python code")
)
t_read = time.perf_counter()
all_events = read_listmode_events(
    reader, header, all_detector_centers, verbose=args.verbose, fname=str(fname)
)
print(f"Read all time blocks in {time.perf_counter() - t_read:.1f} s")
singles_hists = all_events["singles"]
events = all_events["prompts"]
num_delayeds = all_events["delayeds"]["coords0"].shape[0]
print(
    f"{events['coords0'].shape[0]} prompts, {num_delayeds} delayeds, "
    f"{singles_hists['counts'].shape[0]} singles histograms"
)

# fans (detection bins in coincidence) of all detection bins
fans = detection_bin_fans(scanner_info)

# TOF parameters (check_header_support() made sure that all module type pairs have
# the same TOF parameters)
tof_params = get_tof_parameters(scanner_info, 0, 0) if tof else None

# %%
# (unweighted) non-TOF and TOF backprojections of all events

proj = ListmodePETProjector(
    xp.asarray(events["coords0"]).copy(),
    xp.asarray(events["coords1"]).copy(),
    img_shape,
    voxel_size,
    img_origin=img_origin,
)
ones = xp.ones(events["coords0"].shape[0], dtype="float32")
xp.save(output_path("nontof_backproj", tag=params_tag), proj.adjoint(ones))

if tof:
    proj.tof_parameters = tof_params
    proj.event_tofbins = xp.asarray(events["tof_bins"], dtype=xp.int16).copy()
    proj.tof = True
    xp.save(output_path("tof_backproj", tag=params_tag), proj.adjoint(ones))
del proj

# %%
# LM OSEM reconstruction

# events with zero efficiency (e.g. due to errors in the encoding) can not be detected
valid = events["effs"] > 0
if not np.all(valid):
    print(f"Filtering out {np.count_nonzero(~valid)} events with zero efficiency ...")
    events = {k: v[valid] for k, v in events.items()}

# additive contamination (randoms) of all events
num_tofbins = scanner_info.tof_bin_edges[0][0].number_of_bins()
contamination_list, contamination_sum = None, 0.0
randoms_singles, randoms_tag = None, ""
if args.randoms_expectation is not None:
    # "oracle" randoms correction: the true expected randoms in the data bin (detection
    # bins and TOF bin) of every prompt, in the order of the prompts in the file
    if scanner_info.scanner_geometry.number_of_module_types() != 1:
        # read_listmode_events() groups the events by module type pairs
        raise ValueError("--randoms_expectation is only supported for one module type")
    prompts_randoms = np.load(args.randoms_expectation)
    if prompts_randoms.shape != valid.shape:
        raise ValueError(
            f"{prompts_randoms.shape[0]} expected randoms for {valid.shape[0]} prompts"
        )
    # non-TOF: the randoms of an event are the sum over all TOF bins of its LOR
    contamination_list = (prompts_randoms[valid] * (1 if tof else num_tofbins)).astype(
        np.float32
    )
    # the sum of the contamination over all data bins only enters the value of the
    # log-likelihood (not the EM update) and is not known here
    print(f"Oracle randoms correction, mean per event {contamination_list.mean():.3e}")
    randoms_tag = "_randoms_oracle"
elif args.randoms_from_delayeds is not None:
    delayeds = all_events["delayeds"]
    if num_delayeds == 0:
        raise ValueError("--randoms_from_delayeds requires delayed coincidences in the file")
    delayed_fan_sums = delayeds_fan_sums(fans, delayeds["det_bin0"], delayeds["det_bin1"])
    if args.randoms_from_delayeds == "casey":
        randoms_singles = casey_singles(fans, delayed_fan_sums)
        randoms_tag = "_randoms_casey"
    else:
        randoms_singles = ml_singles(
            fans, delayed_fan_sums, num_iterations=args.randoms_ml_iterations
        )
        randoms_tag = f"_randoms_ml{args.randoms_ml_iterations}"
    num_zero = np.count_nonzero(delayed_fan_sums == 0)
    print(
        f"Estimated singles from {num_delayeds} delayeds ({args.randoms_from_delayeds}), "
        f"{num_zero} of {fans.num_bins} detection bins without delayeds"
    )
    np.save(output_path("randoms_singles" + randoms_tag.removeprefix("_randoms")), randoms_singles)
elif args.randoms_from_singles:
    if singles_hists["counts"].shape[0] == 0:
        raise ValueError(
            "--randoms_from_singles requires singles histograms per detection bin in the file"
        )
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()
    if any(
        scanner_info.singles_histogram_energy_bin_edges[m].number_of_bins() != 1
        or scanner_info.event_energy_bin_edges[m].number_of_bins() != 1
        for m in range(num_module_types)
    ):
        raise UnsupportedPETSIRDFeatureError(
            "randoms from singles are only supported for one event and one singles energy bin"
        )
    window_ns = coincidence_window_ns(scanner_info.tof_bin_edges[0][0].edges)
    durations_ns = singles_hists["durations_ms"] * 1e6
    contamination_list = randoms_from_singles_of_events(
        singles_hists["counts"],
        durations_ns,
        window_ns,
        events["det_bin0"],
        events["det_bin1"],
        num_tofbins,
        tof,
    )
    contamination_sum = randoms_from_singles_total(
        fans, singles_hists["counts"], durations_ns, window_ns
    )
    print(
        f"Randoms from singles (coincidence window {window_ns:.3f} ns from the TOF bins): "
        f"{contamination_sum:.4e} expected randoms in total, "
        f"mean per event {contamination_list.mean():.3e}"
    )
    randoms_tag = "_randoms_singles"
elif num_delayeds > 0:
    warnings.warn(
        f"The file contains {num_delayeds} delayed coincidences, but randoms are not "
        "corrected (see --randoms_from_delayeds / --randoms_from_singles): the "
        "reconstruction is biased.",
        UserWarning,
    )

if randoms_singles is not None:
    # the expected randoms of a pair of detection bins (a, b) are s_a * s_b with the
    # (scaled) singles s (see randoms.py)
    if randoms_singles.shape != (fans.num_bins,):
        raise ValueError(
            f"singles of {randoms_singles.shape} detection bins do not match the "
            f"{fans.num_bins} detection bins of the scanner"
        )
    contamination_list = randoms_of_events(
        randoms_singles, events["det_bin0"], events["det_bin1"], num_tofbins, tof
    )
    # sum over all data bins (only needed for the value of the log-likelihood)
    contamination_sum = fans.pair_sum(randoms_singles)
    print(
        f"Randoms correction: {contamination_sum:.4e} expected randoms in total, "
        f"mean per event {contamination_list.mean():.3e}"
    )

# eps added to the expected counts of all events: only needed if the contamination is
# not strictly positive (e.g. without randoms correction) to avoid divisions by zero,
# then 1e-6 x mean counts per data bin (as recommended by parallelproj)
if contamination_list is not None and np.min(contamination_list) > 0:
    eps = 0.0
else:
    eps = 1e-6 * events["coords0"].shape[0] / get_num_data_bins(scanner_info, tof)
print(f"Starting parallelproj LM OSEM reconstruction (eps = {eps:.2e}) ...")

# negative Poisson log-likelihoods of all subsets
subset_objectives = create_lm_subset_objectives(
    events["coords0"],
    events["coords1"],
    events["tof_bins"],
    events["effs"],
    img_shape,
    voxel_size,
    img_origin,
    res_model,
    sens_img,
    args.num_subsets,
    eps,
    tof_params=tof_params,
    att_img=att_img,
    contamination_list=contamination_list,
    contamination_sum=contamination_sum,
    xp=xp,
)
recon = lm_osem(subset_objectives, sens_img, args.num_epochs, xp=xp)

opath = fname.with_name(
    f"lm_osem_{args.num_epochs}_{args.num_subsets}_{params_tag}{randoms_tag}.npy"
)
xp.save(opath, recon)
print(f"LM OSEM recon saved to {opath}")

# %%
# PETSIRD geometry and the first events (on top of the recon)

num_events_to_show = 8
geometry_reader = petsird.BinaryPETSIRDReader(str(fname))
geometry_reader.read_header()
fig_geom = plot_petsird_geometry(
    scanner_info,
    events=read_first_events(geometry_reader, num_events_to_show),
    background=(
        parallelproj.to_numpy_array(recon),
        voxel_size,
        parallelproj.to_numpy_array(img_origin),
    ),
    background_label="LM OSEM recon",
    title=f"PETSIRD scanner geometry and the first {num_events_to_show} events of {fname.name}",
)
geometry_fig_path = output_path("geometry_and_events", ".png", tag=params_tag)
fig_geom.savefig(geometry_fig_path, dpi=150)
print(f"Geometry figure saved to {geometry_fig_path}")

# %%
# show the recon

vols_to_show = {"LM OSEM recon": recon, "sensitivity image": sens_img}
if att_img is not None:
    vols_to_show["attenuation image"] = att_img

# keep references to the viewers (widgets) to keep them interactive
viewers = [
    show_vol_cuts(vol, voxel_size=voxel_size, fig_title=title)
    for title, vol in vols_to_show.items()
]
plt.show()
