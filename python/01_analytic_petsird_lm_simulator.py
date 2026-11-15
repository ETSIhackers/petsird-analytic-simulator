"""analytic simulation of PETSIRD v0.13.1 listmode data for a block PET scanner

We simulate true (unscattered) TOF coincidences including the effect of attenuation
(of a water body), detecting element (crystal) efficiencies and module pair
(symmetry group) efficiencies. Next to the PETSIRD listmode file, reference outputs
(ground truth, attenuation and sensitivity images, ...) are saved for testing
readers / reconstructions of PETSIRD data.

The building blocks are defined in
- scanner.py: scanner geometry (parallelproj and PETSIRD)
- phantoms.py: activity and attenuation images
- efficiencies.py: detection efficiency model (parallelproj LORs and PETSIRD)
- simulation.py: forward model, listmode events, writing PETSIRD files
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from parallelproj.projectors import ListmodePETProjector

from efficiencies import generate_efficiency_model, lor_efficiencies
from geometry_plots import plot_petsird_geometry
from phantoms import PHANTOMS, create_phantom
from scanner import ScannerConfig, create_parallelproj_lor_descriptor
from simulation import (
    create_forward_model,
    create_petsird_header,
    create_tof_projector,
    histogram_mlem,
    histogram_to_listmode,
    plot_sensitivity_and_tof_profile,
    tof_parameters,
    to_petsird_coincidences,
    write_petsird_file,
)
from utils import check_petsird_version

check_petsird_version()


def parse_int_tuple(arg):
    return tuple(map(int, arg.split(",")))


def parse_float_tuple(arg):
    return tuple(map(float, arg.split(",")))


def positive_int(arg):
    value = int(arg)
    if value <= 0:
        raise argparse.ArgumentTypeError("has to be > 0")
    return value


# %%
# parse the command line

parser = argparse.ArgumentParser(
    description="Analytic simulation of PETSIRD v0.13.1 listmode data for a block PET scanner",
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
    "--img_shape", type=parse_int_tuple, default=(55, 55, 19), help="shape of the image to simulate"
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

efficiency_group = parser.add_argument_group("Detection Efficiency Parameters")
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
    "--num_time_blocks", type=int, default=3, help="number of time blocks to split the data into"
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
    "--check_backprojection",
    action="store_true",
    help="save TOF histogram and (TOF and non-TOF) LM backprojections of the simulated data",
)

visual_group = parser.add_argument_group("Visualization and Debugging")
visual_group.add_argument(
    "--skip_plots", action="store_true", help="skip plotting the scanner geometry and TOF profile"
)

general_group = parser.add_argument_group("General Options")
general_group.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")

args = parser.parse_args()

img_shape = args.img_shape
voxel_size = args.voxel_size
num_energy_bins = 1

if args.output_dir is None:
    output_dir = Path("data") / f"sim_{args.phantom}_{args.num_true_counts}_{args.seed}"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "sim_parameters.json", "w", encoding="UTF-8") as f:
    json.dump(vars(args), f, indent=4)

np.random.seed(args.seed)

# %%
# scanner, phantom, efficiencies and forward model

scanner_config = ScannerConfig()
lor_desc = create_parallelproj_lor_descriptor(scanner_config)

img, att_img = create_phantom(args.phantom, img_shape, voxel_size)
np.save(output_dir / "attenuation_image.npy", att_img)

tof_params = tof_parameters(args.tof_fwhm_mm, img_shape, voxel_size)
tof_proj = create_tof_projector(lor_desc, img_shape, voxel_size, tof_params)

efficiency_model = generate_efficiency_model(
    scanner_config, args.uniform_crystal_eff, args.uniform_sg_eff
)

fwd_model = create_forward_model(
    tof_proj, args.fwhm_mm, lor_efficiencies(efficiency_model, lor_desc), att_img
)

# sensitivity images: of the forward model of the data (with attenuation) and of a
# recon without attenuation correction
sens_img = fwd_model.sensitivity_image(with_attenuation=True)
np.save(output_dir / "reference_sensitivity_image.npy", sens_img)
np.save(
    output_dir / "reference_sensitivity_image_no_attenuation.npy",
    fwd_model.sensitivity_image(with_attenuation=False),
)

# %%
# expected and noisy TOF histogram (scaled to the requested number of counts)

expected_counts = fwd_model.op(img)
scale_fac = args.num_true_counts / expected_counts.sum()
img *= scale_fac
expected_counts *= scale_fac
emission_data = np.random.poisson(expected_counts)

np.save(output_dir / "ground_truth_image.npy", img)

if args.num_epochs_mlem > 0:
    recon = histogram_mlem(fwd_model.op, emission_data, sens_img, args.num_epochs_mlem)
    np.save(output_dir / f"reference_histogram_mlem_{args.num_epochs_mlem}_epochs.npy", recon)

# %%
# listmode events

events = histogram_to_listmode(emission_data, lor_desc, tof_params.num_tofbins)

if not args.skip_plots:
    fig = plot_sensitivity_and_tof_profile(sens_img, expected_counts, lor_desc, voxel_size)
    fig.savefig(output_dir / "tof_profile_and_sensitivity_image.png", dpi=300)
    fig.show()

if args.check_backprojection:
    np.save(
        output_dir / "histogram_backprojection_tof.npy",
        fwd_model.proj.adjoint(emission_data.astype(np.float32)),
    )

    scanner = lor_desc.scanner
    lm_proj = ListmodePETProjector(
        scanner.get_lor_endpoints(events.start_module, events.start_el),
        scanner.get_lor_endpoints(events.end_module, events.end_el),
        img_shape,
        fwd_model.proj.voxel_size,
        img_origin=fwd_model.proj.img_origin,
    )
    ones = np.ones(events.num_events, dtype=np.float32)
    np.save(output_dir / "lm_backprojection_non_tof.npy", lm_proj.adjoint(ones))

    # parallelproj uses unsigned TOF bins (0 ... num_tofbins - 1)
    lm_proj.tof_parameters = tof_params
    lm_proj.event_tofbins = events.tof_bin.astype(np.int16)
    lm_proj.tof = True
    np.save(output_dir / "lm_backprojection_tof.npy", lm_proj.adjoint(ones))

# %%
# PETSIRD header and coincidences

header = create_petsird_header(
    scanner_config, efficiency_model, tof_params, args.tof_fwhm_mm, num_energy_bins
)
det_bin_0, det_bin_1, tof_idx = to_petsird_coincidences(
    events, scanner_config.num_det_els_per_module, num_energy_bins, tof_params.num_tofbins
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
    )

if not args.skip_plots:
    plt.show()
