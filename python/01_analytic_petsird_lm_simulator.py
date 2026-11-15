"""analytic simulation of petsird v0.13.1 listmode data for a block PET scanner
we only simulate true (unscattered) events including the effect of attenuation
(of a water cylinder), crystal efficiencies and LOR symmetry group efficiencies
"""

# %%
from importlib.metadata import version

# raise an error if petsird version is not at least 0.13.1
petsird_version = tuple(map(int, version("petsird").split(".")))
if petsird_version < (0, 13, 1):
    raise ImportError(
        f"petsird version {petsird_version} is not supported, please install petsird >= 0.13.1"
    )


import numpy as np
import argparse
from array_api_compat import size
from itertools import combinations

import parallelproj
from parallelproj.operators import (
    CompositeLinearOperator,
    ElementwiseMultiplicationOperator,
    GaussianFilterOperator,
)
from parallelproj.pet_lors import EqualBlockPETLORDescriptor
from parallelproj.pet_scanners import (
    BlockPETScannerModule,
    ModularizedPETScannerGeometry,
)
from parallelproj.projectors import EqualBlockPETProjector, ListmodePETProjector
from parallelproj.tof import TOFParameters
import matplotlib.pyplot as plt
import math
import json

from pathlib import Path

import petsird
from petsird.helpers.create import construct_lower_triangular_matrix


# %%
def circular_distance(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    clockwise_distance = abs(i_mod_1 - i_mod_2)
    counterclockwise_distance = num_modules - clockwise_distance

    return min(clockwise_distance, counterclockwise_distance)


# %%
def module_offset(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    """offset (i_mod_1 - i_mod_2) mod num_modules between the first and second module of a pair

    For our scanner with rotationally symmetric modules, two module pairs with the
    same offset are related by a rotation that maps the first (second) module of one
    pair onto the first (second) module of the other pair. Therefore, all module pairs
    with the same offset have the same module pair efficiencies, i.e. they form a
    PETSIRD symmetry group.
    Note: since PETSIRD stores module pairs (of the same type) in lower-triangular order
    (first module index >= second module index), a rotation can swap the order of the
    modules, which maps offset d to num_modules - d. The module pair efficiencies of
    offset num_modules - d are therefore the transpose of the ones of offset d.
    """
    return (i_mod_1 - i_mod_2) % num_modules


def generate_module_pair_efficiencies(
    offsets: list[int],
    num_modules: int,
    num_det_els: int,
    uniform: bool = False,
) -> dict[int, np.ndarray]:
    """(non-physical) module pair efficiency matrices for all module offsets

    Returns a dictionary offset -> matrix of shape (num_det_els, num_det_els) where
    matrix[i_el_1, i_el_2] is the efficiency for detecting element i_el_1 in the first
    and i_el_2 in the second module of a pair.
    The matrices are random and non-symmetric (to be able to detect transposition
    errors when reading the data), but consistent with the rotational symmetry of the
    scanner: matrix[num_modules - d] = matrix[d].T
    """
    effs = {}
    for d in sorted(offsets):
        if uniform:
            effs[d] = np.ones((num_det_els, num_det_els), dtype="float32")
        elif (num_modules - d) in effs:
            effs[d] = effs[num_modules - d].T.copy()
        else:
            # level alternating between offsets (same level for d and num_modules - d)
            level = 1 + 0.5 * (-1) ** d
            m = np.random.uniform(0.5, 1.5, (num_det_els, num_det_els))
            if 2 * d == num_modules:
                # opposite modules: matrix has to be symmetric
                m = 0.5 * (m + m.T)
            effs[d] = (level * m).astype("float32")
    return effs


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
    description="Analytic simulation of PETSIRD v0.13.1 listmode data for a block PET scanner",
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
    default="uniform_cylinder",
    choices=["uniform_cylinder", "offcenter_cylinder", "points"],
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
    default=10.0,
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
    help="use uniform symmetry group (module pair) efficiencies, otherwise random non-symmetric module pair efficiency matrices are used",
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

modules: list[BlockPETScannerModule] = []

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
        BlockPETScannerModule(
            np,
            "cpu",
            block_shape,
            block_spacing,
            affine_transformation_matrix=module_transforms[i],
        )
    )

# create the scanner geometry from a list of identical block modules at
# different locations in space
scanner = ModularizedPETScannerGeometry(tuple(modules))

# %%
# Setup of a parllelproj LOR descriptor that connectes LOR endpoints in modules
# that are in coincidence

# all possible module pairs
all_combinations = combinations(range(num_blocks), 2)
# exclude module pairs that are too close to each other
block_pairs = [
    (v1, v2) for v1, v2 in all_combinations if circular_distance(v1, v2, num_blocks) > 2
]

lor_desc = EqualBlockPETLORDescriptor(
    scanner,
    np.asarray(block_pairs),
)

# %%
# setup of the ground truth image used for the data simulation

# transaxial disk with a radius of 70% of the image half-width
# (used for the uniform cylinder phantom and the attenuating water cylinder)
tmp = np.linspace(-1, 1, img_shape[0])
X0, X1 = np.meshgrid(tmp, tmp, indexing="ij")
disk = np.astype(np.sqrt(X0**2 + X1**2) < 0.7, "float32")

# coordinates (mm) of all voxel centers (the image is centered at (0,0,0))
X, Y, Z = np.meshgrid(
    *[(np.arange(n) - (n - 1) / 2) * vs for n, vs in zip(img_shape, voxel_size)],
    indexing="ij",
)

img = np.zeros(img_shape, dtype=np.float32)

# attenuating water body (as boolean mask)
# (by default a centered water cylinder covering all slices)
water_body = np.broadcast_to(disk[..., None].astype(bool), img_shape)

if phantom == "uniform_cylinder":
    for i in range(2, img_shape[2] - 2):
        img[..., i] = disk
elif phantom == "offcenter_cylinder":
    # off-center (transaxially and axially) cylinder with 3 hot spheres of different
    # size, contrast and position, such that axis flips, swaps and mirroring (e.g. in
    # readers of the PETSIRD data) are visible in the reconstructions
    cylinder = (np.hypot(X - 10, Y + 5) <= 30) & (Z >= -12) & (Z <= 16)
    img[cylinder] = 1
    # (center x, y, z in mm, radius in mm, activity)
    for (cx, cy, cz), radius, activity in [
        ((22, -5, 6), 8, 3),
        ((5, 12, -4), 6, 2),
        ((0, -18, 10), 4, 3),
    ]:
        img[np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= radius] = activity
    water_body = cylinder
elif phantom == "points":
    img[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 2] = 8
    img[img_shape[0] // 2, img_shape[1] // 6, img_shape[2] // 2] = 4
    img[img_shape[0] // 4, img_shape[1] // 2, img_shape[2] // 2] = 6

    img[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 6] = 8
    img[img_shape[0] // 2, img_shape[1] // 6, img_shape[2] // 6] = 4
    img[img_shape[0] // 4, img_shape[1] // 2, img_shape[2] // 6] = 6

else:
    raise ValueError(f"Invalid phantom {phantom}")

# %%
# setup of the attenuation image (linear attenuation coefficients in mm^-1)
# all phantoms are embedded in a water body (see phantom setup)

mu_water = 0.0096  # linear attenuation coefficient of water at 511 keV (mm^-1)
att_img = np.where(water_body, mu_water, 0).astype(np.float32)

# save the attenuation image (can be used for attenuation correction in the recon)
np.save(output_dir / "attenuation_image.npy", att_img)

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

proj = EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
proj.tof_parameters = TOFParameters(
    num_tofbins=num_tof_bins,
    tofbin_width=tof_bin_width,
    sigma_tof=sig_tof,
    num_sigmas=3.0,
)

# check if the projector passes the adjointness test
# (the projectors require float32 arrays)
assert proj.adjointness_test(np, "cpu", dtype=np.float32, rtol=1e-4)


# %%
# setup a simple image space resolution model

sig = fwhm_mm / (2.35 * np.asarray(voxel_size))
res_model = GaussianFilterOperator(img_shape, sigma=sig)

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

# module pair efficiency matrices for all offsets between modules in coincidence
# (in the PETSIRD convention: first module = module with the higher index)
module_pair_effs = generate_module_pair_efficiencies(
    sorted({module_offset(max(bp), min(bp), num_blocks) for bp in block_pairs}),
    num_blocks,
    lor_desc.num_lorendpoints_per_block,
    uniform=uniform_sg_eff,
)

for i, bp in enumerate(proj.lor_descriptor.all_block_pairs):
    # the block pairs of the LOR descriptor have bp[0] < bp[1], i.e. the LOR end
    # module bp[1] is the first module of the pair in the PETSIRD convention
    mp_eff = module_pair_effs[module_offset(int(bp[1]), int(bp[0]), num_blocks)]
    start_crystal_eff = det_el_efficiencies[bp[0], start_el_arr]
    end_crystal_eff = det_el_efficiencies[bp[1], end_el_arr]
    nontof_sens_histo[i, ...] = (
        mp_eff[end_el_arr, start_el_arr] * start_crystal_eff * end_crystal_eff
    )

# %%
# attenuation factors exp(-int mu) of all LORs (non-TOF line integrals)
nontof_proj = EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
att_factors = np.exp(-nontof_proj(att_img))

# sensitivity histogram without attenuation (only used for reference outputs)
nontof_sens_histo_no_att = nontof_sens_histo.copy()
# multiplicative correction factors of the data: efficiencies x attenuation
nontof_sens_histo *= att_factors

# %%
# setup the complete forward operator consisting of diag(s) P G
# where G is the image-based resolution model
# P is the (TOF) forward projector
# diag(s) is the elementwise multiplication with a non-TOF sensitivity histogram
# (the non-TOF sensitivities are broadcasted along the TOF axis - zero-copy view)
fwd_op = CompositeLinearOperator(
    [
        ElementwiseMultiplicationOperator(
            np.broadcast_to(nontof_sens_histo[..., None], proj.out_shape)
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

# (including attenuation, i.e. the sensitivity image of the forward model of the data)
sens_img = fwd_op.adjoint(np.ones(fwd_op.out_shape, dtype=np.float32))
np.save(output_dir / "reference_sensitivity_image.npy", sens_img)

# sensitivity image without attenuation (= sensitivity image of a recon without AC)
sens_img_no_att = res_model.adjoint(
    proj.adjoint(
        np.broadcast_to(nontof_sens_histo_no_att[..., None], proj.out_shape).astype(
            np.float32
        )
    )
)
np.save(output_dir / "reference_sensitivity_image_no_attenuation.npy", sens_img_no_att)

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
        # emission_data contains int64 Poisson counts -> cast back to float32
        grad = fwd_op.adjoint(((exp - emission_data) / exp).astype(np.float32))
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
    histo_back = proj.adjoint(emission_data.astype(np.float32))

    np.save(output_dir / "histogram_backprojection_tof.npy", histo_back)

    lm_proj = ListmodePETProjector(
        scanner.get_lor_endpoints(event_start_block, event_start_el),
        scanner.get_lor_endpoints(event_end_block, event_end_el),
        img_shape,
        proj.voxel_size,
        img_origin=proj.img_origin,
    )

    # non-TOF LM backprojection
    lm_back_non_tof = lm_proj.adjoint(np.ones(num_events, dtype=np.float32))

    # TOF LM backprojection - parallelproj uses unsigned TOF bins (0 ... num_tofbins - 1)
    lm_proj.tof_parameters = proj.tof_parameters
    lm_proj.event_tofbins = unsigned_event_tof_bin.astype(np.int16)
    lm_proj.tof = True
    lm_back = lm_proj.adjoint(np.ones(num_events, dtype=np.float32))

    np.save(output_dir / "lm_backprojection_tof.npy", lm_back)
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

crystal_centers = BlockPETScannerModule(
    np, "cpu", block_shape, block_spacing
).lor_endpoints

# PETSIRD (>= 0.13) requires that the box of a detecting element is oriented along
# its local x-axis ("box axis", pointing away from the scanner center), that (0,0,0)
# is the average detection point (ADP), and a fixed corner order (see petsird.BoxShape).
# In the module frame of our block modules, the radial (depth) direction is the
# y-axis (index 1 of block_shape). We therefore define the crystal box with its depth
# along the local x-axis and rotate it into the module frame via the detecting element
# transforms. We place the ADP at the crystal center (= LOR endpoints used in the
# parallelproj simulation).

# crystal size along the local box axes (depth, width, axial)
crystal_size = (block_spacing[1], block_spacing[0], block_spacing[2])
L0, L1, L2 = crystal_size

crystal_shape = petsird.BoxShape(
    corners=[
        petsird.Coordinate(c=np.array(c, dtype="float32") - 0.5 * np.array(crystal_size, dtype="float32"))
        for c in (
            (0, 0, 0),
            (0, 0, L2),
            (0, L1, L2),
            (0, L1, 0),
            (L0, 0, 0),
            (L0, 0, L2),
            (L0, L1, L2),
            (L0, L1, 0),
        )
    ]
)
crystal = petsird.BoxSolidVolume(shape=crystal_shape, material_id=1)

# setup the petsird geometry of a module / block

rep_volume = petsird.ReplicatedBoxSolidVolume(object=crystal)

# rotation that maps the local box axis (x) onto the radial direction of the module (y)
crystal_rotation = np.array(
    [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    dtype="float32",
)

for i_c, crystal_center in enumerate(crystal_centers):
    crystal_matrix = np.zeros((3, 4), dtype="float32")
    crystal_matrix[:, :3] = crystal_rotation
    crystal_matrix[:, 3] = crystal_center
    rep_volume.transforms.append(petsird.RigidTransformation(matrix=crystal_matrix))

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
# petsird.DetectionEfficiencies(
#    detection_bin_efficiencies = list[list[float]] -> one list (over all detection bins) per module type
#    module_pair_sgidlut = list[list[list[list[int]]]] -> one (lower-triangular) table per module type pair
#    module_pair_efficiencies_vectors = list[list[list[ModulePairEfficiencies]]]) -> one vector per module type pair
# all "module type pair" tables are lower-triangular, i.e. [type1][type2] with type2 <= type1

# the following only works for a scanner with one module type
assert scanner_geometry.number_of_module_types() == 1

# setup the (lower-triangular) symmetry group ID LUT
# module pairs that are not in coincidence get SGID -1

module_pair_sgid_lut = construct_lower_triangular_matrix(num_blocks, dtype=int, value=-1)

# one symmetry group (SGID) per module offset (see module_offset())
sgid_of_offset = {d: sgid for sgid, d in enumerate(sorted(module_pair_effs))}

for bp in proj.lor_descriptor.all_block_pairs:
    # lower triangular: first module index >= second module index
    # (PETSIRD tables are plain python lists -> use python ints, not numpy ints)
    i_mod_1, i_mod_2 = int(max(bp)), int(min(bp))
    module_pair_sgid_lut[i_mod_1][i_mod_2] = sgid_of_offset[
        module_offset(i_mod_1, i_mod_2, num_blocks)
    ]

module_pair_efficiencies_vector = []

for d, sgid in sgid_of_offset.items():
    # values[detection bin in first module][detection bin in second module]
    # (the detection bin index runs over (element, energy bin) with energy as the fastest index)
    vals = np.kron(
        module_pair_effs[d], np.ones((num_energy_bins, num_energy_bins), dtype="float32")
    )
    module_pair_efficiencies_vector.append(
        petsird.ModulePairEfficiencies(values=vals.tolist(), sgid=sgid)
    )

# detection bin efficiencies: the detection bin index runs over
# (module, element, energy bin) with energy as the fastest index
detection_bin_efficiencies = np.repeat(det_el_efficiencies.ravel(), num_energy_bins)

det_effs = petsird.DetectionEfficiencies(
    method_description="simulated random crystal and symmetry group efficiencies (petsird-analytic-simulator)",
    calibration_factor=1.0,
    detection_bin_efficiencies=[detection_bin_efficiencies.tolist()],
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
    tof_bin_edges=[[tofBinEdges]],  # lower-triangular list of list for all module type pairs
    tof_resolution=[
        [2.35 * proj.tof_parameters.sigma_tof]
    ],  # FWHM in mm, lower-triangular list of list for all module type pairs
    event_energy_bin_edges=[energyBinEdges],  # list for all module types
    energy_resolution_at_511=[0.11],  # as fraction of 511, list for all module types
    detection_efficiencies=det_effs,
)

# we only store prompt (true) coincidences
# all other event types (singles, delayeds, triples, quadruples) keep the default policy NONE
petsird_scanner.prompt_event_policy = petsird.CoincidencePolicy.REJECT_HIGHER_MULTIPLES

################################################################################
################################################################################
################################################################################

# %%
# create the petsird header

exam = petsird.ExamInformation(
    modality="PT",
    patient=petsird.DICOMPatientInformation(patient_id="42"),
)

header = petsird.Header(
    exam=exam,
    scanner=petsird_scanner,
)


################################################################################
################################################################################
################################################################################

# %%
# create petsird coincidence events - all in one timeblock without energy information

num_el_per_block = proj.lor_descriptor.num_lorendpoints_per_block
num_tof_bins_petsird = proj.tof_parameters.num_tofbins

# PETSIRD detection bins: energy_idx + (element_idx + module_idx * num_el) * num_energy_bins
# (all events are in energy bin 0)
event_start_det_bin = (event_start_block * num_el_per_block + event_start_el) * num_energy_bins
event_end_det_bin = (event_end_block * num_el_per_block + event_end_el) * num_energy_bins

# PETSIRD requires ordered coincidences: detection_bins[0] >= detection_bins[1]
# (for events within the same module type). For events where we have to swap the
# detection bins, we also have to mirror the TOF bin, since PETSIRD stores
# (t1 - t2) * c / 2 where t1/t2 are the arrival times at the first/second detection bin.
swap = event_start_det_bin < event_end_det_bin

det_bin_0 = np.where(swap, event_end_det_bin, event_start_det_bin)
det_bin_1 = np.where(swap, event_start_det_bin, event_end_det_bin)
petsird_tof_idx = np.where(
    swap, num_tof_bins_petsird - 1 - unsigned_event_tof_bin, unsigned_event_tof_bin
)

# split the data into chuncks such that we loop over chunks (time blocks)
# every chunk is an array of shape (num_events_per_chunk, 3)
# the first column is the first detection bin, the second column is the second detection bin,
# and the third column is the unsigned TOF bin index

chunked_data = np.array_split(
    np.array([det_bin_0, det_bin_1, petsird_tof_idx]).T,
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
                "First 5 events (first / second detection bin, unsigned tofbin number):"
            )
            print(data_chunk[:5, :])
            print()

            time_block_prompt_events = [
                petsird.CoincidenceEvent(
                    detection_bins=[int(x[0]), int(x[1])],
                    tof_idx=int(x[2]),
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
