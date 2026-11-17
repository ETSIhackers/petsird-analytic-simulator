"""analytic simulation of PETSIRD v0.13.3 listmode data for a block PET scanner

We simulate true (unscattered) TOF coincidences including the effect of attenuation
(of a water body), detecting element (crystal) efficiencies and module pair
(symmetry group) efficiencies, as well as random coincidences (stored in the prompts,
product of the singles of the detecting elements), an independent realization of
the randoms (stored as delayed coincidences) and the singles counts of all detecting
elements (stored as singles histograms). Next to the PETSIRD listmode file,
reference outputs (ground truth, attenuation and sensitivity images, true randoms
expectation, ...) are saved for testing readers / reconstructions of PETSIRD data.

The building blocks are defined in
- scanner.py: scanner geometry (parallelproj and PETSIRD)
- phantoms.py: activity and attenuation images
- efficiencies.py: detection efficiency model (parallelproj LORs and PETSIRD)
- simulation.py: forward model, listmode events, writing PETSIRD files
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from parallelproj.operators import GaussianFilterOperator
from parallelproj.projectors import ListmodePETProjector

from efficiencies import EFFICIENCY_MODELS, generate_efficiency_model, lor_efficiencies
from geometry_plots import plot_petsird_geometry
from phantoms import PHANTOMS, create_phantom
from randoms import coincidence_window_ns, randoms_of_events
from scanner import ScannerConfig, create_parallelproj_lor_descriptor
from reconstruction import histogram_mlem
from simulation import (
    attenuation_factors,
    create_forward_model,
    create_petsird_header,
    create_tof_projector,
    histogram_to_listmode,
    lor_fan_sums,
    randoms_expectation,
    singles_model,
    singles_rates,
    tof_parameters,
    to_petsird_coincidences,
    write_petsird_file,
)
from utils import (
    FWHM_PER_SIGMA,
    PETSIRD_BACKENDS,
    check_petsird_version,
    resolve_petsird_backend,
)

check_petsird_version()


def parse_int_tuple(arg):
    return tuple(map(int, arg.split(",")))


def parse_float_tuple(arg):
    return tuple(map(float, arg.split(",")))


def random_fraction(arg):
    value = float(arg)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("has to be in [0, 1)")
    return value


def positive_int(arg):
    value = int(arg)
    if value <= 0:
        raise argparse.ArgumentTypeError("has to be > 0")
    return value


# %%
# parse the command line

parser = argparse.ArgumentParser(
    description="Analytic simulation of PETSIRD v0.13.3 listmode data for a block PET scanner",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

output_group = parser.add_argument_group("Output Configuration")
output_group.add_argument(
    "--fname", type=str, default="simulated_petsird_lm_file.bin", help="name of the output LM file"
)
output_group.add_argument(
    "--output_dir", type=str, default=None, help="directory to save output files"
)
output_group.add_argument(
    "--skip_writing", action="store_true", help="skip writing the LM data to a file"
)

image_group = parser.add_argument_group("Image and Phantom Configuration")
image_group.add_argument(
    "--phantom",
    type=str,
    default="uniform_cylinder",
    choices=list(PHANTOMS),
    help="phantom to simulate",
)
image_group.add_argument(
    "--img_shape", type=parse_int_tuple, default=(55, 55, 21), help="shape of the image to simulate"
)
image_group.add_argument(
    "--voxel_size",
    type=parse_float_tuple,
    default=(2.0, 2.0, 2.0),
    help="voxel size in mm used in the simulation",
)

physics_group = parser.add_argument_group("Physics / simulation parameters")
physics_group.add_argument(
    "--fwhm_mm", type=float, default=2.5, help="FWHM of the image space resolution model in mm"
)
physics_group.add_argument(
    "--tof_fwhm_mm", type=float, default=10.0, help="FWHM of the TOF resolution model in mm"
)
physics_group.add_argument(
    "--num_true_counts",
    type=positive_int,
    default=int(4e6),
    help="(expected) number of true coincidences to simulate",
)
physics_group.add_argument(
    "--random_fraction",
    type=random_fraction,
    default=0.5,
    help="(expected) random fraction R / (T + R) of the prompts, where R / T are the "
    "total randoms / (attenuated) trues; the randoms are also stored as independent "
    "realization in the delayed coincidences (0 means no randoms and no delayeds)",
)

scanner_group = parser.add_argument_group("Scanner Geometry")
scanner_group.add_argument(
    "--modules_per_panel",
    type=int,
    choices=(1, 2),
    default=2,
    help="2: 6 flat panels of 2 modules around the scanner axis (only symmetric under a "
    "rotation by one panel, i.e. 60 deg), 1: 12 modules equally spaced around the axis",
)
scanner_group.add_argument(
    "--panel_gap", type=float, default=4.0, help="gap between the modules of a panel (mm)"
)
scanner_group.add_argument(
    "--num_axial_modules",
    type=int,
    default=3,
    help="number of axial module positions (the transaxial arrangement of 12 modules is "
    "repeated at each axial position)",
)
scanner_group.add_argument(
    "--axial_gap", type=float, default=2.0, help="gap between neighboring axial module positions (mm)"
)
scanner_group.add_argument(
    "--block_shape",
    type=parse_int_tuple,
    default=(10, 2, 3),
    help="number of detecting elements (4.5 x 10 x 4.5 mm) per module (tangential, radial, axial)",
)

efficiency_group = parser.add_argument_group("Detection Efficiency Parameters")
efficiency_group.add_argument(
    "--efficiency_model",
    choices=EFFICIENCY_MODELS,
    default="structured",
    help="structured: exaggerated, structured detecting element and module pair efficiencies "
    "(sectors, rings, axial profiles, a dead patch; clear artifacts if they are ignored or "
    "misinterpreted), random: random efficiencies (previous model)",
)
efficiency_group.add_argument(
    "--uniform_crystal_eff",
    action="store_true",
    help="use uniform crystal efficiencies, otherwise use a random distribution",
)
efficiency_group.add_argument(
    "--uniform_sg_eff",
    action="store_true",
    help="use uniform symmetry group (module pair) efficiencies, otherwise random non-symmetric module pair efficiency matrices are used",
)

time_group = parser.add_argument_group("Time Block Configuration")
time_group.add_argument(
    "--num_time_blocks", type=int, default=100, help="number of time blocks to split the data into"
)
time_group.add_argument(
    "--event_block_duration", type=int, default=100, help="duration of each time block in ms"
)

recon_group = parser.add_argument_group("Reconstruction and Analysis")
recon_group.add_argument(
    "--num_epochs_mlem",
    type=int,
    default=0,
    help="number of epochs for MLEM reconstruction of histogrammed data, 0 means no MLEM reconstruction",
)
recon_group.add_argument(
    "--save_lm_backprojection",
    action="store_true",
    help="save the TOF LM backprojection of the simulated events (before writing them to "
    "PETSIRD, used to check the written events)",
)

visual_group = parser.add_argument_group("Visualization and Debugging")
visual_group.add_argument(
    "--skip_plots", action="store_true", help="skip plotting the PETSIRD scanner geometry and events"
)

general_group = parser.add_argument_group("General Options")
general_group.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
general_group.add_argument(
    "--petsird_backend",
    choices=PETSIRD_BACKENDS,
    default="auto",
    help="write the time blocks with the yardl-generated Python code (petsird) or the "
    "optional compiled package petsird_fast (C++, much faster, byte-identical files); "
    "auto: petsird_fast if installed",
)
output_group.add_argument(
    "--petsird_summary",
    action="store_true",
    help="write a human-readable summary of the PETSIRD file (petsird_summary.txt, output "
    "of petsird.helpers.analysis, which reads all events with the yardl-generated Python "
    "code and is slow for large files)",
)

args = parser.parse_args()
# the backend that is used (e.g. auto -> petsird_fast), also stored in sim_parameters.json
args.petsird_backend = resolve_petsird_backend(args.petsird_backend)

img_shape = args.img_shape
voxel_size = args.voxel_size
# all events are in one energy bin (detection bin = detecting element)
num_energy_bins = 1

if args.output_dir is None:
    output_dir = (
        Path("data")
        / f"sim_{args.phantom}_{args.num_true_counts}_{args.seed}_{args.petsird_backend}"
    )
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "sim_parameters.json", "w", encoding="UTF-8") as f:
    json.dump(vars(args), f, indent=4)

np.random.seed(args.seed)

# %%
# scanner, phantom, efficiencies and forward model

scanner_config = ScannerConfig(
    modules_per_panel=args.modules_per_panel,
    panel_gap=args.panel_gap,
    num_axial_modules=args.num_axial_modules,
    axial_gap=args.axial_gap,
    block_shape=tuple(args.block_shape),
)
lor_desc = create_parallelproj_lor_descriptor(scanner_config)

img, att_img = create_phantom(args.phantom, img_shape, voxel_size)
np.save(output_dir / "attenuation_image.npy", att_img)

tof_params = tof_parameters(args.tof_fwhm_mm, img_shape, voxel_size)
tof_proj = create_tof_projector(lor_desc, img_shape, voxel_size, tof_params)

efficiency_model = generate_efficiency_model(
    scanner_config, args.uniform_crystal_eff, args.uniform_sg_eff, args.efficiency_model
)

res_model = GaussianFilterOperator(
    img_shape, sigma=args.fwhm_mm / (FWHM_PER_SIGMA * np.asarray(voxel_size))
)

# forward model of the data: diag(efficiencies x attenuation factors) P G
fwd_op = create_forward_model(
    tof_proj,
    res_model,
    lor_efficiencies(efficiency_model, lor_desc)
    * attenuation_factors(lor_desc, img_shape, voxel_size, att_img),
)

# sensitivity image of the forward model (incl. attenuation)
sens_img = fwd_op.adjoint(np.ones(fwd_op.out_shape, dtype=np.float32))
np.save(output_dir / "reference_sensitivity_image.npy", sens_img)

# %%
# expected trues and randoms, and Poisson realizations of the prompts and delayeds
# (TOF histograms)

expected_trues = fwd_op(img)
scale_fac = args.num_true_counts / expected_trues.sum()
img *= scale_fac
expected_trues *= scale_fac

# noise-free randoms with an expected random fraction R / (T + R): the randoms of a LOR
# are proportional to the product of the singles of its detecting elements and uniform
# over the TOF bins. The singles depend on the activity seen by the detecting elements
# (fan sums of the expected trues) and their efficiencies (0 for dead crystals).
total_expected_randoms = args.random_fraction / (1 - args.random_fraction) * args.num_true_counts
singles = singles_model(
    lor_fan_sums(expected_trues.sum(-1), lor_desc, scanner_config.num_modules),
    efficiency_model.det_el_effs,
)
expected_randoms, scaled_singles = randoms_expectation(
    singles, lor_desc, tof_params.num_tofbins, total_expected_randoms
)

# constant singles rates S of all detecting elements, such that the expected randoms
# 2 tau T S_a S_b of the acquisition (time T) are the simulated randoms. The coincidence
# window 2 tau is the range covered by the TOF bins.
window_ns = coincidence_window_ns(
    np.array([-0.5, 0.5]) * tof_params.num_tofbins * tof_params.tofbin_width
)
time_block_duration_ns = args.event_block_duration * 1e6
rates_per_ns = singles_rates(
    scaled_singles.ravel(), window_ns, args.num_time_blocks * time_block_duration_ns
)

# prompts: Poisson realization of trues + randoms
prompts_histogram = np.random.poisson(expected_trues + expected_randoms)

# true expectation of the randoms (e.g. to test randoms corrections)
with open(output_dir / "randoms_expectation.json", "w", encoding="UTF-8") as f:
    json.dump(
        {
            "random_fraction": args.random_fraction,
            "expected_trues": float(args.num_true_counts),
            "expected_randoms": float(total_expected_randoms),
            "num_tofbins": tof_params.num_tofbins,
            "coincidence_window_ns": window_ns,
            "acquisition_time_ns": args.num_time_blocks * time_block_duration_ns,
            # singles rates (per ns) of all detection bins
            "singles_rates_per_ns": rates_per_ns.tolist(),
            # one value per PETSIRD detection bin (= detecting element for one energy bin)
            "singles": scaled_singles.ravel().tolist(),
            "description": "the expected randoms of a pair of detection bins (a, b) in "
            "coincidence (summed over all TOF bins) are singles[a] * singles[b] and are "
            "uniformly distributed over the TOF bins (see python/randoms.py); "
            "singles[a] * singles[b] = coincidence_window_ns * acquisition_time_ns * "
            "singles_rates_per_ns[a] * singles_rates_per_ns[b] (constant singles rates)",
        },
        f,
        indent=4,
    )

# ground truth activity image (before the image-based resolution model, which is part
# of the forward model), scaled such that the forward model yields the expected counts
np.save(output_dir / "ground_truth_image.npy", img)

if args.num_epochs_mlem > 0:
    recon = histogram_mlem(
        fwd_op,
        prompts_histogram,
        sens_img,
        args.num_epochs_mlem,
        contamination=expected_randoms if args.random_fraction > 0 else None,
    )
    np.save(output_dir / f"reference_histogram_mlem_{args.num_epochs_mlem}_epochs.npy", recon)

# %%
# listmode events of the prompts and delayeds

prompt_events = histogram_to_listmode(prompts_histogram, lor_desc, tof_params.num_tofbins)

# delayeds: independent Poisson realization of the randoms
if args.random_fraction > 0:
    delayed_events = histogram_to_listmode(
        np.random.poisson(expected_randoms), lor_desc, tof_params.num_tofbins
    )

# singles histograms (ideal: same energy window as the coincidences, no dead time) of
# all detecting elements in every time block: Poisson realizations of the singles rates
if args.random_fraction > 0:
    singles_counts = np.random.poisson(
        np.broadcast_to(rates_per_ns * time_block_duration_ns, (args.num_time_blocks, rates_per_ns.size))
    )

if args.save_lm_backprojection:
    # TOF LM backprojection of the events before writing them to PETSIRD
    # (parallelproj uses unsigned TOF bins 0 ... num_tofbins - 1)
    scanner = lor_desc.scanner
    lm_proj = ListmodePETProjector(
        scanner.get_lor_endpoints(prompt_events.start_module, prompt_events.start_el),
        scanner.get_lor_endpoints(prompt_events.end_module, prompt_events.end_el),
        img_shape,
        tof_proj.voxel_size,
        img_origin=tof_proj.img_origin,
    )
    lm_proj.tof_parameters = tof_params
    lm_proj.event_tofbins = prompt_events.tof_bin.astype(np.int16)
    lm_proj.tof = True
    np.save(
        output_dir / "lm_backprojection_tof.npy",
        lm_proj.adjoint(np.ones(prompt_events.num_events, dtype=np.float32)),
    )

# %%
# PETSIRD header and coincidences

header = create_petsird_header(
    scanner_config,
    efficiency_model,
    tof_params,
    args.tof_fwhm_mm,
    num_energy_bins,
    store_delayeds=args.random_fraction > 0,
    store_singles=args.random_fraction > 0,
)
det_bin_0, det_bin_1, tof_idx = to_petsird_coincidences(
    prompt_events, scanner_config.num_det_els_per_module, num_energy_bins, tof_params.num_tofbins
)
petsird_delayeds = (
    to_petsird_coincidences(
        delayed_events,
        scanner_config.num_det_els_per_module,
        num_energy_bins,
        tof_params.num_tofbins,
    )
    if args.random_fraction > 0
    else None
)

# true (noise-free) expected randoms in the data bin (pair of detection bins and TOF bin)
# of every prompt, in the order of the prompts in the PETSIRD file (= additive
# contamination of a TOF LM reconstruction)
np.save(
    output_dir / "prompts_randoms_expectation.npy",
    randoms_of_events(
        scaled_singles.ravel(), det_bin_0, det_bin_1, tof_params.num_tofbins, tof=True
    ),
)

if not args.skip_plots:
    num_events_to_show = 8
    fig_geom = plot_petsird_geometry(
        header.scanner,
        events=np.stack(
            [
                np.zeros(num_events_to_show, dtype=np.int64),
                np.zeros(num_events_to_show, dtype=np.int64),
                det_bin_0[:num_events_to_show],
                det_bin_1[:num_events_to_show],
                tof_idx[:num_events_to_show],
            ],
            axis=1,
        ),
        background=(img, voxel_size, -(np.asarray(img_shape) - 1) / 2 * np.asarray(voxel_size)),
        background_label="ground truth activity",
        title=f"PETSIRD scanner geometry and the first {num_events_to_show} events written to {args.fname}",
    )
    fig_geom.savefig(output_dir / "petsird_geometry_and_events.png", dpi=150)
    fig_geom.show()

if not args.skip_writing:
    print(f"Writing LM file to {output_dir / args.fname}")
    write_petsird_file(
        output_dir / args.fname,
        header,
        det_bin_0,
        det_bin_1,
        tof_idx,
        args.num_time_blocks,
        args.event_block_duration,
        delayeds=petsird_delayeds,
        singles_counts=singles_counts if args.random_fraction > 0 else None,
        backend=args.petsird_backend,
    )

if args.petsird_summary and not args.skip_writing:
    # human-readable summary of the written file (output of petsird.helpers.analysis)
    summary = subprocess.run(
        [sys.executable, "-m", "petsird.helpers.analysis", "-i", str(output_dir / args.fname)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (output_dir / "petsird_summary.txt").write_text(summary)

if not args.skip_plots:
    plt.show()
