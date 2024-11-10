"""analytic simulation of petsird v0.2 listmode data for a block PET scanner
   we only simulate true events and ignore the effect of attenuation
   however, we simulate the effect of crystal efficiencies and LOR symmetry group efficiencies
"""

# %%
import array_api_compat.numpy as xp
import argparse
from array_api_compat import size
from itertools import combinations

import parallelproj
import petsird
import matplotlib.pyplot as plt
import math
import json

from pathlib import Path


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
parser = argparse.ArgumentParser()

parser.add_argument("--fname", type=str, default="simulated_lm_file.bin")
parser.add_argument("--output_dir", type=str, default="my_lm_sim")
parser.add_argument("--num_true_counts", type=int, default=int(4e6))
parser.add_argument("--skip_plots", action="store_true")
parser.add_argument("--check_backprojection", default=False, action="store_true")
parser.add_argument("--num_epochs_mlem", type=int, default=0)
parser.add_argument("--skip_writing", default=False, action="store_true")
parser.add_argument("--fwhm_mm", type=float, default=1.5)
parser.add_argument("--tof_fwhm_mm", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--uniform_crystal_eff", action="store_true")
parser.add_argument("--uniform_sg_eff", action="store_true")
parser.add_argument("--img_shape", type=parse_int_tuple, default=(100, 100, 11))
parser.add_argument("--voxel_size", type=parse_float_tuple, default=(1.0, 1.0, 1.0))
parser.add_argument(
    "--phantom",
    type=str,
    default="squares",
    choices=["uniform_cylinder", "squares"],
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
output_dir = Path(args.output_dir)
uniform_crystal_eff = args.uniform_crystal_eff
uniform_sg_eff = args.uniform_sg_eff
img_shape = args.img_shape
voxel_size = args.voxel_size

if not output_dir.exists():
    output_dir.mkdir(parents=True)

# dump args into a json file output_dir / "args.json"
with open(output_dir / "sim_parameters.json", "w", encoding="UTF-8") as f:
    json.dump(vars(args), f, indent=4)

# %%
# "fixed" input parameters
dev = "cpu"
xp.random.seed(args.seed)

# %%
# input parameters related to the scanner geometry

# number of LOR endpoints per block module in all 3 directions
block_shape = (10, 2, 3)
# spacing between LOR endpoints in a block module in all three directions (mm)
block_spacing = (4.5, 10.0, 4.5)
# radius of the scanner - distance from the center to the block modules (mm)
scanner_radius = 100
# number of modules - we will have 12 block modules arranged in a "circle"
num_blocks = 12


# %%
# Setup of a modularized parallelproj PET scanner geometry

modules = []

# setup an affine transformation matrix to translate the block modules from the
# center to the radius of the scanner
aff_mat_trans = xp.eye(4, dtype="float32", device=dev)
aff_mat_trans[1, -1] = scanner_radius

module_transforms = []

for i, phi in enumerate(xp.linspace(0, 2 * xp.pi, num_blocks, endpoint=False)):
    # setup an affine transformation matrix to rotate the block modules around the center
    # (of the "2" axis)
    aff_mat_rot = xp.asarray(
        [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype="float32",
        device=dev,
    )

    module_transforms.append(aff_mat_rot @ aff_mat_trans)

    modules.append(
        parallelproj.BlockPETScannerModule(
            xp,
            dev,
            block_shape,
            block_spacing,
            affine_transformation_matrix=module_transforms[i],
        )
    )

# create the scanner geometry from a list of identical block modules at
# different locations in space
scanner = parallelproj.ModularizedPETScannerGeometry(modules)

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
    xp.asarray(block_pairs, device=dev),
)

# %%
# setup of the ground truth image used for the data simulation

img = xp.zeros(img_shape, dtype=xp.float32, device=dev)

if phantom == "uniform_cylinder":
    tmp = xp.linspace(-1, 1, img_shape[0])
    X0, X1 = xp.meshgrid(tmp, tmp, indexing="ij")
    disk = xp.astype(xp.sqrt(X0**2 + X1**2) < 0.7, "float32")
    for i in range(img_shape[2]):
        img[..., i] = disk
elif phantom == "squares":
    img[2:-12, 32:-20, 2:-1] = 3
    img[24:-40, 36:-28, 4:-2] = 9
    img[76:78, 68:72, :-2] = 18
    img[14:20, 35:75, 5:-3] = 0
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
num_tof_bins = int(2 * xp.sqrt(2) * img_shape[0] * voxel_size[0] / tof_bin_width)
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
assert proj.adjointness_test(xp, dev)


# %%
# setup a simple image space resolution model

sig = fwhm_mm / (2.35 * xp.asarray(voxel_size, device=dev))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

# %%
# setup the sensitivity sinogram consisting of the crystal efficiencies factors
# and the LOR symmetry group efficiencies

tmp = xp.arange(proj.lor_descriptor.num_lorendpoints_per_block)
start_el, end_el = xp.meshgrid(tmp, tmp, indexing="ij")
start_el_arr = xp.reshape(start_el, (size(start_el),))
end_el_arr = xp.reshape(end_el, (size(end_el),))

nontof_sens_histo = xp.ones(proj.out_shape[:-1], dtype="float32", device=dev)

if uniform_crystal_eff:
    # crystal efficiencies are all 1
    det_el_efficiencies = xp.ones(
        scanner.num_modules, lor_desc.num_lorendpoints_per_block, dtype="float32"
    )
else:
    # simulate random crystal eff. uniformly distributed between 0.2 - 2.2
    det_el_efficiencies = 0.2 + 2 * xp.astype(
        xp.random.rand(scanner.num_modules, lor_desc.num_lorendpoints_per_block),
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

sens_img = fwd_op.adjoint(xp.ones(fwd_op.out_shape, dtype=xp.float32, device=dev))
xp.save(output_dir / "reference_sensitivity_image.npy", sens_img)

# %%
# add poisson noise on the forward projection
if num_true_counts > 0:
    scale_fac = num_true_counts / img_fwd_tof.sum()
    img *= scale_fac
    img_fwd_tof *= scale_fac
    emission_data = xp.random.poisson(img_fwd_tof)
else:
    emission_data = img_fwd_tof

# save the ground truth image
xp.save(output_dir / "ground_truth_image.npy", img)

# %%
if num_epochs_mlem > 0:
    recon = xp.ones(img_shape, dtype=xp.float32, device=dev)

    for i in range(num_epochs_mlem):
        print(f"{(i+1):03}/{num_epochs_mlem:03}", end="\r")

        exp = xp.clip(fwd_op(recon), 1e-6, None)
        grad = fwd_op.adjoint((exp - emission_data) / exp)
        step = recon / sens_img
        recon -= step * grad

    print("")
    xp.save(
        output_dir / f"reference_histogram_mlem_{num_epochs_mlem}_epochs.npy", recon
    )

# %%
# convert emission histogram to LM

if num_true_counts > 0:
    num_events = emission_data.sum()
    event_start_block = xp.zeros(num_events, dtype="uint32", device=dev)
    event_start_el = xp.zeros(num_events, dtype="uint32", device=dev)
    event_end_block = xp.zeros(num_events, dtype="uint32", device=dev)
    event_end_el = xp.zeros(num_events, dtype="uint32", device=dev)
    event_tof_bin = xp.zeros(num_events, dtype="int32", device=dev)

    event_counter = 0

    for ibp, block_pair in enumerate(proj.lor_descriptor.all_block_pairs):
        for it, tof_bin in enumerate(
            xp.arange(proj.tof_parameters.num_tofbins)
            - proj.tof_parameters.num_tofbins // 2
        ):
            ss = emission_data[ibp, :, it]
            num_slice_events = ss.sum()
            inds = xp.repeat(xp.arange(ss.shape[0]), ss)

            # event start block
            event_start_block[event_counter : (event_counter + num_slice_events)] = (
                block_pair[0]
            )
            # event start element in block
            event_start_el[event_counter : (event_counter + num_slice_events)] = (
                xp.take(start_el_arr, inds)
            )
            # event end module
            event_end_block[event_counter : (event_counter + num_slice_events)] = (
                block_pair[1]
            )
            # event end element in block
            event_end_el[event_counter : (event_counter + num_slice_events)] = xp.take(
                end_el_arr, inds
            )
            # event TOF bin - starting at 0
            event_tof_bin[event_counter : (event_counter + num_slice_events)] = tof_bin

            event_counter += num_slice_events

    # shuffle lm_event_table along 0 axis
    inds = xp.arange(num_events)
    xp.random.shuffle(inds)

    event_start_block = event_start_block[inds]
    event_start_el = event_start_el[inds]
    event_end_block = event_end_block[inds]
    event_end_el = event_end_el[inds]
    event_tof_bin = event_tof_bin[inds]

    del inds

    # create the unsigned tof bin (the index to the tof bin edges) that we need to write
    unsigned_event_tof_bin = xp.asarray(
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
    vmin = float(xp.min(sens_img))
    vmax = float(xp.max(sens_img))
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

    xp.save(output_dir / "histogram_backprojection_tof.npy", histo_back)

    lm_back = parallelproj.joseph3d_back_tof_lm(
        xstart=scanner.get_lor_endpoints(event_start_block, event_start_el),
        xend=scanner.get_lor_endpoints(event_end_block, event_end_el),
        img_shape=img_shape,
        img_origin=proj.img_origin,
        voxsize=proj.voxel_size,
        img_fwd=xp.ones(num_events, dtype=xp.float32, device=dev),
        tofbin_width=proj.tof_parameters.tofbin_width,
        sigma_tof=xp.asarray([proj.tof_parameters.sigma_tof]),
        tofcenter_offset=xp.asarray([proj.tof_parameters.tofcenter_offset]),
        nsigmas=proj.tof_parameters.num_sigmas,
        tofbin=event_tof_bin,
    )
    xp.save(output_dir / "lm_backprojection_tof.npy", lm_back)

    lm_back_non_tof = parallelproj.joseph3d_back(
        xstart=scanner.get_lor_endpoints(event_start_block, event_start_el),
        xend=scanner.get_lor_endpoints(event_end_block, event_end_el),
        img_shape=img_shape,
        img_origin=proj.img_origin,
        voxsize=proj.voxel_size,
        img_fwd=xp.ones(num_events, dtype=xp.float32, device=dev),
    )
    xp.save(output_dir / "lm_backprojection_non_tof.npy", lm_back_non_tof)

# %%
# create the petsird header

if num_true_counts > 0:
    subject = petsird.Subject(id="42")
    institution = petsird.Institution(
        name="Ministry of Silly Walks",
        address="42 Silly Walks Street, Silly Walks City",
    )

    # create non geometry related scanner information

    num_energy_bins = 1

    # TOF bin edges (in mm)
    tofBinEdges = xp.linspace(
        -proj.tof_parameters.num_tofbins * proj.tof_parameters.tofbin_width / 2,
        proj.tof_parameters.num_tofbins * proj.tof_parameters.tofbin_width / 2,
        proj.tof_parameters.num_tofbins + 1,
        dtype="float32",
    )

    energyBinEdges = xp.linspace(430, 650, num_energy_bins + 1, dtype="float32")

    num_total_elements = proj.lor_descriptor.scanner.num_lor_endpoints

    # setup the symmetry group ID LUT
    # we only create one symmetry group ID (1) and set the group ID to -1 for block
    # block pairs that are not in coincidence

    module_pair_sgid_lut = xp.full((num_blocks, num_blocks), -1, dtype="int32")

    for bp in proj.lor_descriptor.all_block_pairs:
        # generate a random sgd
        sgid = sgid_from_module_pair(bp[0], bp[1], num_blocks)
        module_pair_sgid_lut[bp[0], bp[1]] = sgid

    num_SGIDs = module_pair_sgid_lut.max() + 1

    num_el_per_module = proj.lor_descriptor.scanner.num_lor_endpoints_per_module[0]

    module_pair_efficiencies_shape = (
        num_el_per_module,
        num_energy_bins,
        num_el_per_module,
        num_energy_bins,
    )

    module_pair_efficiencies_vector = []

    for sgid in range(num_SGIDs):
        eff = module_pair_eff_from_sgd(sgid, uniform=uniform_sg_eff)
        vals = xp.full(module_pair_efficiencies_shape, eff, dtype="float32", device=dev)

        module_pair_efficiencies_vector.append(
            petsird.ModulePairEfficiencies(values=vals, sgid=sgid)
        )

    det_effs = petsird.DetectionEfficiencies(
        det_el_efficiencies=xp.reshape(
            det_el_efficiencies, (size(det_el_efficiencies), 1)
        ),
        module_pair_sgidlut=module_pair_sgid_lut,
        module_pair_efficiencies_vector=module_pair_efficiencies_vector,
    )

    # setup crystal box object

    crystal_centers = parallelproj.BlockPETScannerModule(
        xp, dev, block_shape, block_spacing
    ).lor_endpoints

    # crystal widths in all dimensions
    cw0 = block_spacing[0]
    cw1 = block_spacing[1]
    cw2 = block_spacing[2]

    crystal_shape = petsird.BoxShape(
        corners=[
            petsird.Coordinate(
                c=xp.asarray((-cw0 / 2, -cw1 / 2, -cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((-cw0 / 2, -cw1 / 2, cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((-cw0 / 2, cw1 / 2, cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((-cw0 / 2, cw1 / 2, -cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((cw0 / 2, -cw1 / 2, -cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((cw0 / 2, -cw1 / 2, cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((cw0 / 2, cw1 / 2, cw2 / 2), dtype="float32")
            ),
            petsird.Coordinate(
                c=xp.asarray((cw0 / 2, cw1 / 2, -cw2 / 2), dtype="float32")
            ),
        ]
    )
    crystal = petsird.BoxSolidVolume(shape=crystal_shape, material_id=1)

    # setup the petsird geometry of a module / block

    rep_volume = petsird.ReplicatedBoxSolidVolume(object=crystal)

    for i_c, crystal_center in enumerate(crystal_centers):
        translation_matrix = xp.eye(4, dtype="float32")[:-1, :]
        for j in range(3):
            translation_matrix[j, -1] = crystal_center[j]
        transform = petsird.RigidTransformation(matrix=translation_matrix)

        rep_volume.transforms.append(transform)
        rep_volume.ids.append(i_c)

    detector_module = petsird.DetectorModule(
        detecting_elements=[rep_volume], detecting_element_ids=[0]
    )

    # setup the PETSIRD scanner geometry
    rep_module = petsird.ReplicatedDetectorModule(object=detector_module)

    for i in range(num_blocks):
        transform = petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])

        rep_module.ids.append(i)
        rep_module.transforms.append(
            petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])
        )

    scanner_geometry = petsird.ScannerGeometry(replicated_modules=[rep_module], ids=[0])

    # need energy bin info before being able to construct the detection efficiencies
    # so we will construct a scanner without the efficiencies first
    petsird_scanner = petsird.ScannerInformation(
        model_name="PETSIRD_TEST",
        scanner_geometry=scanner_geometry,
        tof_bin_edges=tofBinEdges,
        tof_resolution=2.35 * proj.tof_parameters.sigma_tof,  # FWHM in mm
        energy_bin_edges=energyBinEdges,
        energy_resolution_at_511=0.11,  # as fraction of 511
        event_time_block_duration=1,  # ms
    )

    petsird_scanner.detection_efficiencies = det_effs

    header = petsird.Header(
        exam=petsird.ExamInformation(subject=subject, institution=institution),
        scanner=petsird_scanner,
    )

    # %%
    # create petsird coincidence events - all in one timeblock without energy information

    num_el_per_block = proj.lor_descriptor.num_lorendpoints_per_block

    det_ID_start = event_start_block * num_el_per_block + event_start_el
    det_ID_end = event_end_block * num_el_per_block + event_end_el

    # %%
    # write petsird data

    if not skip_writing:
        print(f"Writing LM file to {str(output_dir / fname)}")
        with petsird.BinaryPETSIRDWriter(str(output_dir / fname)) as writer:
            writer.write_header(header)
            for i_t in range(1):
                start = i_t * header.scanner.event_time_block_duration

                time_block_prompt_events = [
                    petsird.CoincidenceEvent(
                        detector_ids=[det_ID_start[i], det_ID_end[i]],
                        tof_idx=unsigned_event_tof_bin[i],
                        energy_indices=[0, 0],
                    )
                    for i in range(num_events)
                ]

                # Normally we'd write multiple blocks, but here we have just one, so let's write a tuple with just one element
                writer.write_time_blocks(
                    (
                        petsird.TimeBlock.EventTimeBlock(
                            petsird.EventTimeBlock(
                                start=start, prompt_events=time_block_prompt_events
                            )
                        ),
                    )
                )
