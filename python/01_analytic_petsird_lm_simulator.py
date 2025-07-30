"""analytic simulation of petsird v0.7.2 listmode data for a block PET scanner
we only simulate true events and ignore the effect of attenuation
however, we simulate the effect of crystal efficiencies and LOR symmetry group efficiencies
"""

# %%
from importlib.metadata import version

# raise an error if petsird version is not at least 0.7.2
petsird_version = tuple(map(int, version("petsird").split(".")))
if petsird_version < (0, 7, 2):
    raise ImportError(
        f"petsird version {petsird_version} is not supported, please install petsird >= 0.7.2"
    )


import numpy as np
import argparse
from array_api_compat import size
from itertools import combinations

import parallelproj
import matplotlib.pyplot as plt
import math
import json

from pathlib import Path

import petsird


# %%
def circular_distance(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    clockwise_distance = abs(i_mod_1 - i_mod_2)
    counterclockwise_distance = num_modules - clockwise_distance

    return min(clockwise_distance, counterclockwise_distance)


# %%
def sgid_from_module_pair(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    """a random mapping between two modules into a symmetry group"""

    return circular_distance(i_mod_1, i_mod_2, num_modules) % 3


# %%
def module_pair_eff_from_sgd(i_sgd: int, uniform: bool = False) -> float:
    """a random mapping from symmetry group id (sgid) to efficiency"""
    if uniform:
        res = 1.0
    else:
        res = 1 + 0.5 * ((-1) ** i_sgd)

    return res


def parse_int_tuple(arg):
    return tuple(map(int, arg.split(",")))


def parse_float_tuple(arg):
    return tuple(map(float, arg.split(",")))


################################################################################
################################################################################
################################################################################

# %%
# parse the command line for the input parameters below
parser = argparse.ArgumentParser(
    description="Analytic simulation of PETSIRD v0.7.2 listmode data for a block PET scanner",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Output configuration group
output_group = parser.add_argument_group("Output Configuration")
output_group.add_argument(
    "--fname",
    type=str,
    default="simulated_petsird_lm_file.bin",
    help="name of the output LM file",
)
output_group.add_argument(
    "--output_dir", type=str, default=None, help="directory to save output files"
)
output_group.add_argument(
    "--skip_writing",
    default=False,
    action="store_true",
    help="skip writing the LM data to a file",
)

# Image and phantom configuration group
image_group = parser.add_argument_group("Image and Phantom Configuration")
image_group.add_argument(
    "--phantom",
    type=str,
    default="cylinder",
    choices=["cylinder", "uniform_cylinder", "squares", "points"],
    help="phantom to simulate",
)
image_group.add_argument(
    "--img_shape",
    type=parse_int_tuple,
    default=(55, 55, 19),
    help="shape of the image to simulate",
)
image_group.add_argument(
    "--voxel_size",
    type=parse_float_tuple,
    default=(2.0, 2.0, 2.0),
    help="voxel size in mm used in the simulation",
)

# Physics and resolution parameters group
physics_group = parser.add_argument_group("Physics / simulation parameters")
physics_group.add_argument(
    "--fwhm_mm",
    type=float,
    default=2.5,
    help="FWHM of the image space resolution model in mm",
)
physics_group.add_argument(
    "--tof_fwhm_mm",
    type=float,
    default=20.0,
    help="FWHM of the TOF resolution model in mm",
)
physics_group.add_argument(
    "--num_true_counts",
    type=int,
    default=int(4e6),
    help="number of true coincidences to simulate",
)

# Efficiency parameters group
efficiency_group = parser.add_argument_group("Detection Efficiency Parameters")
efficiency_group.add_argument(
    "--uniform_crystal_eff",
    action="store_true",
    help="use uniform crystal efficiencies, otherwise use a random distribution",
)
efficiency_group.add_argument(
    "--uniform_sg_eff",
    action="store_true",
    help="use uniform symmetry group (module pair) efficiencies, otherwise pseudo random pattern is used",
)

# Time block configuration group
time_group = parser.add_argument_group("Time Block Configuration")
time_group.add_argument(
    "--num_time_blocks",
    type=int,
    default=3,
    help="number of time blocks to split the data into",
)
time_group.add_argument(
    "--event_block_duration",
    type=int,
    default=100,
    help="duration of each time block in ms",
)

# Reconstruction and analysis group
recon_group = parser.add_argument_group("Reconstruction and Analysis")
recon_group.add_argument(
    "--num_epochs_mlem",
    type=int,
    default=0,
    help="number of epochs for MLEM reconstruction of histogrammed data, 0 means no MLEM reconstruction",
)
recon_group.add_argument(
    "--check_backprojection",
    default=False,
    action="store_true",
    help="check the backprojection of the TOF histogram and LM data",
)

# Visualization and debugging group
visual_group = parser.add_argument_group("Visualization and Debugging")
visual_group.add_argument(
    "--skip_plots",
    action="store_true",
    help="skip plotting the scanner geometry and TOF profile",
)

# General options group
general_group = parser.add_argument_group("General Options")
general_group.add_argument(
    "--seed", type=int, default=0, help="random seed for reproducibility"
)

args = parser.parse_args()

fname = args.fname
skip_plots = args.skip_plots
check_backprojection = args.check_backprojection
num_true_counts = args.num_true_counts
skip_writing = args.skip_writing
num_epochs_mlem = args.num_epochs_mlem
fwhm_mm = args.fwhm_mm
tof_fwhm_mm = args.tof_fwhm_mm
seed = args.seed
phantom = args.phantom
uniform_crystal_eff = args.uniform_crystal_eff
uniform_sg_eff = args.uniform_sg_eff
img_shape = args.img_shape
voxel_size = args.voxel_size
num_time_blocks: int = args.num_time_blocks
event_block_duration: int = args.event_block_duration

if args.output_dir is None:
    output_dir = Path("data") / f"sim_{phantom}_{num_true_counts}_{seed}"
else:
    output_dir = Path(args.output_dir)

num_energy_bins: int = 1

if not output_dir.exists():
    output_dir.mkdir(parents=True)

# dump args into a json file output_dir / "args.json"
with open(output_dir / "sim_parameters.json", "w", encoding="UTF-8") as f:
    json.dump(vars(args), f, indent=4)

# %%
# "fixed" input parameters
np.random.seed(args.seed)

# %%
# input parameters related to the scanner geometry

# number of LOR endpoints per block module in all 3 directions
block_shape = (10, 2, 9)
# spacing between LOR endpoints in a block module in all three directions (mm)
block_spacing = (4.5, 10.0, 4.5)
# radius of the scanner - distance from the center to the block modules (mm)
scanner_radius = 100
# number of modules - we will have 12 block modules arranged in a "circle"
num_blocks = 12


# %%
# Setup of a modularized parallelproj PET scanner geometry

modules: list[parallelproj.BlockPETScannerModule] = []

# setup an affine transformation matrix to translate the block modules from the
# center to the radius of the scanner
aff_mat_trans = np.eye(4, dtype="float32")
aff_mat_trans[1, -1] = scanner_radius

module_transforms = []

for i, phi in enumerate(np.linspace(0, 2 * np.pi, num_blocks, endpoint=False)):
    # setup an affine transformation matrix to rotate the block modules around the center
    # (of the "2" axis)
    aff_mat_rot = np.asarray(
        [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype="float32",
    )

    module_transforms.append(aff_mat_rot @ aff_mat_trans)

    modules.append(
        parallelproj.BlockPETScannerModule(
            np,
            "cpu",
            block_shape,
            block_spacing,
            affine_transformation_matrix=module_transforms[i],
        )
    )

# create the scanner geometry from a list of identical block modules at
# different locations in space
scanner = parallelproj.ModularizedPETScannerGeometry(tuple(modules))

# %%
# Setup of a parllelproj LOR descriptor that connectes LOR endpoints in modules
# that are in coincidence

# all possible module pairs
all_combinations = combinations(range(num_blocks), 2)
# exclude module pairs that are too close to each other
block_pairs = [
    (v1, v2) for v1, v2 in all_combinations if circular_distance(v1, v2, num_blocks) > 2
]

lor_desc = parallelproj.EqualBlockPETLORDescriptor(
    scanner,
    np.asarray(block_pairs),
)

# %%
# setup of the ground truth image used for the data simulation

img = np.zeros(img_shape, dtype=np.float32)

if phantom == "uniform_cylinder":
    tmp = np.linspace(-1, 1, img_shape[0])
    X0, X1 = np.meshgrid(tmp, tmp, indexing="ij")
    disk = np.astype(np.sqrt(X0**2 + X1**2) < 0.7, "float32")
    for i in range(2, img_shape[2] - 2):
        img[..., i] = disk
elif phantom == "squares":
    img[2:-12, 32:-20, 2:-1] = 3
    img[24:-40, 36:-28, 4:-2] = 9
    img[76:78, 68:72, :-2] = 18
    img[14:20, 35:75, 5:-3] = 0
elif phantom == "points":
    img[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 2] = 8
    img[img_shape[0] // 2, img_shape[1] // 6, img_shape[2] // 2] = 4
    img[img_shape[0] // 4, img_shape[1] // 2, img_shape[2] // 2] = 6

    img[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 6] = 8
    img[img_shape[0] // 2, img_shape[1] // 6, img_shape[2] // 6] = 4
    img[img_shape[0] // 4, img_shape[1] // 2, img_shape[2] // 6] = 6

else:
    raise ValueError("Invalid phantom {phantom}")

# %%
# setup of a parallelproj TOF projector

# TOF parameters
sig_tof = tof_fwhm_mm / 2.35
tof_bin_width = 0.8 * sig_tof
# calculate the number of TOF bins
# we set it to twice the image diagonal divided by the tof bin width
# and make sure it is an odd number
num_tof_bins = int(np.sqrt(2) * img_shape[0] * voxel_size[0] / tof_bin_width)
if num_tof_bins % 2 == 0:
    num_tof_bins += 1

proj = parallelproj.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=num_tof_bins,
    tofbin_width=tof_bin_width,
    sigma_tof=sig_tof,
    num_sigmas=3.0,
)

# check if the projector passes the adjointness test
assert proj.adjointness_test(np, "cpu")


# %%
# setup a simple image space resolution model

sig = fwhm_mm / (2.35 * np.asarray(voxel_size))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

# %%
# setup the sensitivity sinogram consisting of the crystal efficiencies factors
# and the LOR symmetry group efficiencies

tmp = np.arange(proj.lor_descriptor.num_lorendpoints_per_block)
start_el, end_el = np.meshgrid(tmp, tmp, indexing="ij")
start_el_arr = np.reshape(start_el, (size(start_el),))
end_el_arr = np.reshape(end_el, (size(end_el),))

nontof_sens_histo = np.ones(proj.out_shape[:-1], dtype="float32")

if uniform_crystal_eff:
    # crystal efficiencies are all 1
    det_el_efficiencies = np.ones(
        (scanner.num_modules, lor_desc.num_lorendpoints_per_block), dtype="float32"
    )
else:
    # simulate random crystal eff. uniformly distributed between 0.2 - 2.2
    det_el_efficiencies = 0.2 + 2 * np.astype(
        np.random.rand(scanner.num_modules, lor_desc.num_lorendpoints_per_block),
        "float32",
    )
    # multiply the det el eff. of the first module by 3 to introduce more variation
    det_el_efficiencies[0, :] *= 3
    # divide the det el eff. of the last module by 3 to introduce more variation
    det_el_efficiencies[-1, :] /= 3
    # simulate a few dead crystals
    det_el_efficiencies[det_el_efficiencies < 0.21] = 0

for i, bp in enumerate(proj.lor_descriptor.all_block_pairs):
    sgid = sgid_from_module_pair(bp[0], bp[1], num_blocks)
    sg_eff = module_pair_eff_from_sgd(sgid, uniform=uniform_sg_eff)
    start_crystal_eff = det_el_efficiencies[bp[0], start_el_arr]
    end_crystal_eff = det_el_efficiencies[bp[1], end_el_arr]
    nontof_sens_histo[i, ...] = sg_eff * start_crystal_eff * end_crystal_eff

# %%
# setup the complete forward operator consisting of diag(s) P G
# where G is the image-based resolution model
# P is the (TOF) forward projector
# diag(s) is the elementwise multiplication with a non-TOF sensitivity histogram
fwd_op = parallelproj.CompositeLinearOperator(
    [
        parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            proj.out_shape, nontof_sens_histo
        ),
        proj,
        res_model,
    ]
)

# %%
# TOF forward project an image full of ones. The forward projection has the
# shape (num_block_pairs, num_lors_per_block_pair, num_tofbins)

img_fwd_tof = fwd_op(img)

# %%
# calculate the sensitivity image

sens_img = fwd_op.adjoint(np.ones(fwd_op.out_shape, dtype=np.float32))
np.save(output_dir / "reference_sensitivity_image.npy", sens_img)

# %%
# add poisson noise on the forward projection
if num_true_counts > 0:
    scale_fac = num_true_counts / img_fwd_tof.sum()
    img *= scale_fac
    img_fwd_tof *= scale_fac
    emission_data = np.random.poisson(img_fwd_tof)
else:
    emission_data = img_fwd_tof

# save the ground truth image
np.save(output_dir / "ground_truth_image.npy", img)

# %%
if num_epochs_mlem > 0:
    recon = np.ones(img_shape, dtype=np.float32)

    for i in range(num_epochs_mlem):
        print(f"{(i+1):03}/{num_epochs_mlem:03}", end="\r")

        exp = np.clip(fwd_op(recon), 1e-6, None)
        grad = fwd_op.adjoint((exp - emission_data) / exp)
        step = recon / sens_img
        recon -= step * grad

    print("")
    np.save(
        output_dir / f"reference_histogram_mlem_{num_epochs_mlem}_epochs.npy", recon
    )

# %%
# convert emission histogram to LM

if num_true_counts > 0:
    num_events = emission_data.sum()
    event_start_block = np.zeros(num_events, dtype="uint32")
    event_start_el = np.zeros(num_events, dtype="uint32")
    event_end_block = np.zeros(num_events, dtype="uint32")
    event_end_el = np.zeros(num_events, dtype="uint32")
    event_tof_bin = np.zeros(num_events, dtype="int32")

    event_counter = 0

    for ibp, block_pair in enumerate(proj.lor_descriptor.all_block_pairs):
        for it, tof_bin in enumerate(
            np.arange(proj.tof_parameters.num_tofbins)
            - proj.tof_parameters.num_tofbins // 2
        ):
            ss = emission_data[ibp, :, it]
            num_slice_events = ss.sum()
            inds = np.repeat(np.arange(ss.shape[0]), ss)

            # event start block
            event_start_block[event_counter : (event_counter + num_slice_events)] = (
                block_pair[0]
            )
            # event start element in block
            event_start_el[event_counter : (event_counter + num_slice_events)] = (
                np.take(start_el_arr, inds)
            )
            # event end module
            event_end_block[event_counter : (event_counter + num_slice_events)] = (
                block_pair[1]
            )
            # event end element in block
            event_end_el[event_counter : (event_counter + num_slice_events)] = np.take(
                end_el_arr, inds
            )
            # event TOF bin - starting at 0
            event_tof_bin[event_counter : (event_counter + num_slice_events)] = tof_bin

            event_counter += num_slice_events

    # shuffle lm_event_table along 0 axis
    inds = np.arange(num_events)
    np.random.shuffle(inds)

    event_start_block = event_start_block[inds]
    event_start_el = event_start_el[inds]
    event_end_block = event_end_block[inds]
    event_end_el = event_end_el[inds]
    event_tof_bin = event_tof_bin[inds]

    del inds

    # create the unsigned tof bin (the index to the tof bin edges) that we need to write
    unsigned_event_tof_bin = np.asarray(
        event_tof_bin + proj.tof_parameters.num_tofbins // 2, dtype="uint32"
    )

# %%
# Visualize the projector geometry and and the first 3 coincidences
# Visualize the TOF profile of one LOR of the noise free data and the sensitivity image

if not skip_plots:
    fig_geom = plt.figure(figsize=(4, 4), tight_layout=True)
    ax_geom = fig_geom.add_subplot(111, projection="3d")
    ax_geom.set_xlabel("x0")
    ax_geom.set_ylabel("x1")
    ax_geom.set_zlabel("x2")
    proj.show_geometry(ax_geom)

    if num_true_counts > 0:
        for i in range(3):
            event_start_coord = scanner.get_lor_endpoints(
                event_start_block[i : (i + 1)], event_start_el[i : (i + 1)]
            )[0]
            event_end_coord = scanner.get_lor_endpoints(
                event_end_block[i : (i + 1)], event_end_el[i : (i + 1)]
            )[0]

            ax_geom.plot(
                [event_start_coord[0], event_end_coord[0]],
                [event_start_coord[1], event_end_coord[1]],
                [event_start_coord[2], event_end_coord[2]],
            )

    fig_geom.savefig(output_dir / "scanner_geometry.png", dpi=300)
    fig_geom.show()

    fig2, ax2 = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
    vmin = float(np.min(sens_img))
    vmax = float(np.max(sens_img))
    for i, sl in enumerate(
        [img_shape[2] // 4, img_shape[2] // 2, 3 * img_shape[2] // 4]
    ):
        ax2[i].imshow(
            parallelproj.to_numpy_array(sens_img[:, :, sl]),
            vmin=vmin,
            vmax=vmax,
            cmap="Greys",
        )
        ax2[i].set_title(f"sens. img. sl {sl}", fontsize="small")

    ax2[-1].plot(parallelproj.to_numpy_array(img_fwd_tof[17, 0, :]), ".-")
    ax2[-1].set_xlabel("TOF bin")
    ax2[-1].set_title(
        f"TOF profile of LOR 0 in block pair {block_pairs[17]}", fontsize="small"
    )

    fig2.savefig(output_dir / "tof_profile_and_sensitivity_image.png", dpi=300)
    fig2.show()


# %%
# do a TOF histogram as well as TOF and nonTOF LM backprojection

if check_backprojection and (num_true_counts > 0):
    histo_back = proj.adjoint(emission_data)

    np.save(output_dir / "histogram_backprojection_tof.npy", histo_back)

    lm_back = parallelproj.joseph3d_back_tof_lm(
        xstart=scanner.get_lor_endpoints(event_start_block, event_start_el),
        xend=scanner.get_lor_endpoints(event_end_block, event_end_el),
        img_shape=img_shape,
        img_origin=proj.img_origin,
        voxsize=proj.voxel_size,
        img_fwd=np.ones(num_events, dtype=np.float32),
        tofbin_width=proj.tof_parameters.tofbin_width,
        sigma_tof=np.asarray([proj.tof_parameters.sigma_tof]),
        tofcenter_offset=np.asarray([proj.tof_parameters.tofcenter_offset]),
        nsigmas=proj.tof_parameters.num_sigmas,
        tofbin=event_tof_bin,
    )
    np.save(output_dir / "lm_backprojection_tof.npy", lm_back)

    lm_back_non_tof = parallelproj.joseph3d_back(
        xstart=scanner.get_lor_endpoints(event_start_block, event_start_el),
        xend=scanner.get_lor_endpoints(event_end_block, event_end_el),
        img_shape=img_shape,
        img_origin=proj.img_origin,
        voxsize=proj.voxel_size,
        img_fwd=np.ones(num_events, dtype=np.float32),
    )
    np.save(output_dir / "lm_backprojection_non_tof.npy", lm_back_non_tof)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

# %%
# create ScannerGeometry

# The top down hiearchy of the scanner geometry is as follows:
#   ScannerGeometry(list[ReplicatedDetectorModule])
#   ReplicatedDetectorModule(list[RigidTransformation], DetectorModule)
#   DetectorModule(ReplicatedBoxSolidVolume)
#   ReplicatedBoxSolidVolume(list[RigidTransformation], BoxSolidVolume)

crystal_centers = parallelproj.BlockPETScannerModule(
    np, "cpu", block_shape, block_spacing
).lor_endpoints

# crystal widths in all dimensions
cw0 = block_spacing[0]
cw1 = block_spacing[1]
cw2 = block_spacing[2]

crystal_shape = petsird.BoxShape(
    corners=[
        petsird.Coordinate(c=np.array((-cw0 / 2, -cw1 / 2, -cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((-cw0 / 2, -cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((-cw0 / 2, cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((-cw0 / 2, cw1 / 2, -cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((cw0 / 2, -cw1 / 2, -cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((cw0 / 2, -cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((cw0 / 2, cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=np.array((cw0 / 2, cw1 / 2, -cw2 / 2), dtype="float32")),
    ]
)
crystal = petsird.BoxSolidVolume(shape=crystal_shape, material_id=1)

# setup the petsird geometry of a module / block

rep_volume = petsird.ReplicatedBoxSolidVolume(object=crystal)

for i_c, crystal_center in enumerate(crystal_centers):
    translation_matrix = np.eye(4, dtype="float32")[:-1, :]
    for j in range(3):
        translation_matrix[j, -1] = crystal_center[j]
    transform = petsird.RigidTransformation(matrix=translation_matrix)

    rep_volume.transforms.append(transform)

detector_module = petsird.DetectorModule(detecting_elements=rep_volume)

# setup the PETSIRD scanner geometry
rep_module = petsird.ReplicatedDetectorModule(object=detector_module)

for i in range(num_blocks):
    transform = petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])

    rep_module.transforms.append(
        petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])
    )

scanner_geometry = petsird.ScannerGeometry(replicated_modules=[rep_module])

################################################################################
################################################################################
################################################################################

# %%
# setup of detection efficiencies

# The top down hierarchy of the detection efficiencies is as follows:
# petsird.DetectionEfficiencies(detection_bin_efficiencies = list[numpy.ndarray], -> one 2D table per module type
#                               module_pair_sgidlut = list[list[numpy.ndarray]] -> one 2D table per module type combination
#                               module_pair_efficiencies_vectors = list[list[list[ModulePairEfficiencies]]]) -> list of modulepair efficiency vectors per module type combination

# the following only works for a scanner with one module type
# if there are more module types, we need a list of DetectionEfficiencies
# and list of list of module_pair_sgidlut and list of list of list of ModulePairEfficiencies
assert scanner_geometry.number_of_replicated_modules() == 1

# setup the symmetry group ID LUT
# we only create one symmetry group ID (1) and set the group ID to -1 for block
# block pairs that are not in coincidence

module_pair_sgid_lut = np.full((num_blocks, num_blocks), -1, dtype="int32")

for bp in proj.lor_descriptor.all_block_pairs:
    # generate a random sgd
    sgid = sgid_from_module_pair(bp[0], bp[1], num_blocks)
    module_pair_sgid_lut[bp[0], bp[1]] = sgid

num_SGIDs = module_pair_sgid_lut.max() + 1

num_el_per_module = proj.lor_descriptor.scanner.num_lor_endpoints_per_module[0]

module_pair_efficiencies_shape = (
    num_el_per_module * num_energy_bins,
    num_el_per_module * num_energy_bins,
)

module_pair_efficiencies_vector = []

for sgid in range(num_SGIDs):
    eff = module_pair_eff_from_sgd(sgid, uniform=uniform_sg_eff)
    vals = np.full(module_pair_efficiencies_shape, eff, dtype="float32")

    module_pair_efficiencies_vector.append(
        petsird.ModulePairEfficiencies(values=vals, sgid=sgid)
    )

# only correct for scanner with one module type
det_effs = petsird.DetectionEfficiencies(
    detection_bin_efficiencies=[det_el_efficiencies.ravel()],
    module_pair_sgidlut=[[module_pair_sgid_lut]],
    module_pair_efficiencies_vectors=[[module_pair_efficiencies_vector]],
)

################################################################################
################################################################################
################################################################################

# %%
# setup ScannerInformation and Header

# TOF bin edges (in mm)
tofBinEdges = petsird.BinEdges(
    edges=np.linspace(
        -proj.tof_parameters.num_tofbins * proj.tof_parameters.tofbin_width / 2,
        proj.tof_parameters.num_tofbins * proj.tof_parameters.tofbin_width / 2,
        proj.tof_parameters.num_tofbins + 1,
        dtype="float32",
    )
)

energyBinEdges = petsird.BinEdges(
    edges=np.linspace(430, 650, num_energy_bins + 1, dtype="float32")
)

# num_total_elements = proj.lor_descriptor.scanner.num_lor_endpoints

# need energy bin info before being able to construct the detection efficiencies
# so we will construct a scanner without the efficiencies first
petsird_scanner = petsird.ScannerInformation(
    model_name="PETSIRD_TEST",
    scanner_geometry=scanner_geometry,
    tof_bin_edges=[[tofBinEdges]],  # list of list for all module type combinations
    tof_resolution=[
        [2.35 * proj.tof_parameters.sigma_tof]
    ],  # FWHM in mm, list of list for all module type combinations
    event_energy_bin_edges=[energyBinEdges],  # list for all module types
    energy_resolution_at_511=[0.11],  # as fraction of 511, list for all module types
    detection_efficiencies=det_effs,
)

petsird_scanner.coincidence_policy = petsird.CoincidencePolicy.REJECT_MULTIPLES
petsird_scanner.delayed_coincidences_are_stored = False
petsird_scanner.triple_events_are_stored = False

################################################################################
################################################################################
################################################################################

# %%
# create the petsird header

subject = petsird.Subject(id="42")
institution = petsird.Institution(
    name="Ministry of Silly Walks",
    address="42 Silly Walks Street, Silly Walks City",
)

header = petsird.Header(
    exam=petsird.ExamInformation(subject=subject, institution=institution),
    scanner=petsird_scanner,
)


################################################################################
################################################################################
################################################################################

# %%
# create petsird coincidence events - all in one timeblock without energy information

num_el_per_block = proj.lor_descriptor.num_lorendpoints_per_block

# split the data into chuncks such that we loop over chunks (time blocks)
# every chunk is an array of shape (num_events_per_chunk, 3)
# the first column is the start detector ID, the second column is the end detector ID,
# and the third column is the unsigned TOF bin index

chunked_data = np.array_split(
    np.array(
        [
            event_start_block * num_el_per_block + event_start_el,
            event_end_block * num_el_per_block + event_end_el,
            unsigned_event_tof_bin,
        ]
    ).T,
    num_time_blocks,
)

# %%
# write petsird data

if not skip_writing:
    print(f"Writing LM file to {str(output_dir / fname)}")
    with petsird.BinaryPETSIRDWriter(str(output_dir / fname)) as writer:
        writer.write_header(header)
        for i_t, data_chunk in enumerate(chunked_data):

            print(f"Writing time block {i_t + 1}/{len(chunked_data)}")
            print(
                "First 5 events (start / stop detection element, unsigned tofbin number):"
            )
            print(data_chunk[:5, :])
            print()

            time_block_prompt_events = [
                petsird.CoincidenceEvent(
                    detection_bins=[x[0], x[1]],
                    tof_idx=x[2],
                )
                for x in data_chunk
            ]

            # Normally we'd write multiple blocks, but here we have just one, so let's write a tuple with just one element
            writer.write_time_blocks(
                (
                    petsird.TimeBlock.EventTimeBlock(
                        petsird.EventTimeBlock(
                            prompt_events=[[time_block_prompt_events]],
                            time_interval=petsird.TimeInterval(
                                start=i_t * event_block_duration,
                                stop=(i_t + 1) * event_block_duration,
                            ),
                        )
                    ),
                )
            )
