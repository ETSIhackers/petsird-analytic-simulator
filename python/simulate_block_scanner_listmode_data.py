# %%
import array_api_compat.numpy as xp
import pymirc.viewer as pv
import argparse
from array_api_compat import size

import parallelproj
import petsird
import matplotlib.pyplot as plt
import math


# %%
def sgid_from_module_pair(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    """a random mapping between two modules into a symmetry group"""
    clockwise_distance = abs(i_mod_1 - i_mod_2)
    counterclockwise_distance = num_modules - clockwise_distance

    if (i_mod_1 == 1 and i_mod_2 == 7) or (i_mod_1 == 7 and i_mod_2 == 1):
        sgid = 3
    else:
        sgid = (min(clockwise_distance, counterclockwise_distance) + 2) % 3

    return sgid


# %%
def module_pair_eff_from_sgd(i_sgd: int) -> float:
    """a random mapping from symmetry group id (sgid) to efficiency"""
    return float((i_sgd + 1) ** 1.5)


# %%
# parse the command line for the input parameters below
parser = argparse.ArgumentParser()

parser.add_argument("--fname", type=str, default="sim_lm.bin")
parser.add_argument("--num_true_counts", type=int, default=int(1e6))
parser.add_argument("--show_plots", default=False, action="store_true")
parser.add_argument("--check_backprojection", default=False, action="store_true")
parser.add_argument("--run_recon", default=False, action="store_true")
parser.add_argument("--num_iter", type=int, default=10)
parser.add_argument("--skip_writing", default=False, action="store_true")
parser.add_argument("--fwhm_mm", type=float, default=1.5)
parser.add_argument("--tof_fwhm_mm", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

fname = args.fname
show_plots = args.show_plots
check_backprojection = args.check_backprojection
run_recon = args.run_recon
num_true_counts = args.num_true_counts
skip_writing = args.skip_writing
num_iter = args.num_iter
fwhm_mm = args.fwhm_mm
tof_fwhm_mm = args.tof_fwhm_mm
seed = args.seed

dev = "cpu"
xp.random.seed(args.seed)

# %%
# input parameters

# grid shape of LOR endpoints forming a block module
block_shape = (10, 2, 3)
# spacing between LOR endpoints in a block module
block_spacing = (4.5, 10.0, 4.5)
# radius of the scanner
scanner_radius = 100
# number of modules
num_blocks = 12


# %%
# Setup of a modularized PET scanner geometry
# -------------------------------------------
#
# We define 7 block modules arranged in a circle with a radius of 10.
# The arangement follows a regular polygon with 12 sides, leaving some
# of the sides empty.
# Note that all block modules must be identical, but can be anywhere in space.
# The location of a block module can be changed using an affine transformation matrix.

mods = []

delta_phi = 2 * xp.pi / num_blocks

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

    mods.append(
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
scanner = parallelproj.ModularizedPETScannerGeometry(mods)

# %%
# Setup of a LOR descriptor consisting of block pairs
# ---------------------------------------------------
#
# Once the geometry of the LOR endpoints is defined, we can define the LORs
# by specifying which block pairs are in coincidence and for "valid" LORs.
# To do this, we have manually define a list containing pairs of block numbers.
# Here, we define 9 block pairs. Note that more pairs would be possible.

block_pairs = []

for j in range(num_blocks):
    block_pairs += [[j, (j + 3 + i) % num_blocks] for i in range(7)]

lor_desc = parallelproj.EqualBlockPETLORDescriptor(
    scanner,
    xp.asarray(block_pairs, device=dev),
)

# %%
# Setup of a non-TOF projector
# ----------------------------
#
# Now that the LOR descriptor is defined, we can setup the projector.

img_shape = (100, 100, 12)
voxel_size = (1.0, 1.0, 1.0)
img = xp.zeros(img_shape, dtype=xp.float32, device=dev)

if True:
    tmp = xp.linspace(-1, 1, img_shape[0])
    X0, X1 = xp.meshgrid(tmp, tmp, indexing="ij")
    disk = xp.astype(xp.sqrt(X0**2 + X1**2) < 0.7, "float32")
    for i in range(img_shape[2]):
        img[..., i] = disk
else:
    img[2:-12, 32:-20, 2:] = 3
    img[24:-40, 36:-28, 4:-2] = 9
    img[76:78, 68:72, :-2] = 18
    img[52:56, 38:42, :-2] = 0

# %%
# Setup of a TOF projector
# ------------------------
#
# Now that the LOR descriptor is defined, we can setup the projector.

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

assert proj.adjointness_test(xp, dev)


# %%
# setup a simple image space resolution model
sig = fwhm_mm / (2.35 * xp.asarray(voxel_size, device=dev))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

# setup the sensitivity sinogram
tmp = xp.arange(proj.lor_descriptor.num_lorendpoints_per_block)
start_el, end_el = xp.meshgrid(tmp, tmp, indexing="ij")
start_el_arr = xp.reshape(start_el, (size(start_el),))
end_el_arr = xp.reshape(end_el, (size(end_el),))

nontof_sens_sino = xp.ones(proj.out_shape[:-1], dtype="float32", device=dev)

# simulate random crystal eff. uniformly distributed between 0.2 - 2.2
det_el_efficiencies = 0.2 + 2 * xp.random.rand(
    lor_desc.num_block_pairs, lor_desc.num_lorendpoints_per_block
)
# simulate a few dead crystals
det_el_efficiencies[det_el_efficiencies < 0.25] = 0

for i, bp in enumerate(proj.lor_descriptor.all_block_pairs):
    sgid = sgid_from_module_pair(bp[0], bp[1], num_blocks)
    start_crystal_eff = det_el_efficiencies[bp[0], start_el_arr]
    end_crystal_eff = det_el_efficiencies[bp[1], end_el_arr]

    nontof_sens_sino[i, ...] = (
        module_pair_eff_from_sgd(sgid) * start_crystal_eff * end_crystal_eff
    )

# %%
# setup the complete forward operator consisting of diag(s) P G
# where G is the image-based resolution model
# P is the (TOF) forward projector
# diag(s) is the elementwise multiplication with a non-TOF sens. sinogram
fwd_op = parallelproj.CompositeLinearOperator(
    [
        parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            proj.out_shape, nontof_sens_sino
        ),
        proj,
        res_model,
    ]
)

# %%
# TOF forward project an image full of ones. The forward projection has the
# shape (num_block_pairs, num_lors_per_block_pair, num_tofbins)

img_fwd_tof = fwd_op(img)
scale_fac = num_true_counts / img_fwd_tof.sum()

img *= scale_fac
img_fwd_tof *= scale_fac

print(img_fwd_tof.shape)

# %%
# TOF backproject a "TOF histogram" full of ones ("sensitivity image" when attenuation
# and normalization are ignored)

ones_back_tof = fwd_op.adjoint(xp.ones(fwd_op.out_shape, dtype=xp.float32, device=dev))
print(ones_back_tof.shape)

# %%
# put poisson noise on the forward projection
emission_data = xp.random.poisson(img_fwd_tof)
# emission_data = xp.zeros(img_fwd_tof.shape, dtype=xp.int32, device=dev)
# emission_data[3, 1, 10] = 1

# %%
# convert emission histogram to LM


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
        event_start_el[event_counter : (event_counter + num_slice_events)] = xp.take(
            start_el_arr, inds
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

if show_plots:
    fig_geom = plt.figure(figsize=(4, 4), tight_layout=True)
    ax_geom = fig_geom.add_subplot(111, projection="3d")
    ax_geom.set_xlabel("x0")
    ax_geom.set_ylabel("x1")
    ax_geom.set_zlabel("x2")
    proj.show_geometry(ax_geom)

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

    fig_geom.show()

# %%
# Visualize the TOF profile of one LOR of the noise free data and the sensitivity image

if show_plots:
    fig6, ax6 = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
    vmin = float(xp.min(ones_back_tof))
    vmax = float(xp.max(ones_back_tof))
    for i, sl in enumerate(
        [img_shape[2] // 4, img_shape[2] // 2, 3 * img_shape[2] // 4]
    ):
        ax6[i].imshow(
            parallelproj.to_numpy_array(ones_back_tof[:, :, sl]),
            vmin=vmin,
            vmax=vmax,
            cmap="Greys",
        )
        ax6[i].set_title(f"sens. img. sl {sl}", fontsize="small")

    ax6[-1].plot(parallelproj.to_numpy_array(img_fwd_tof[17, 0, :]), ".-")
    ax6[-1].set_xlabel("TOF bin")
    ax6[-1].set_title(
        f"TOF profile of LOR 0 in block pair {block_pairs[17]}", fontsize="small"
    )

    fig6.show()


# %%
# do a sinogram and LM back projection of the emission data
# if the Poisson emission sinogram to LM conversion is correct, those should
# look very similar

if check_backprojection:
    histo_back = proj.adjoint(emission_data)
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

    lm_back_non_tof = parallelproj.joseph3d_back(
        xstart=scanner.get_lor_endpoints(event_start_block, event_start_el),
        xend=scanner.get_lor_endpoints(event_end_block, event_end_el),
        img_shape=img_shape,
        img_origin=proj.img_origin,
        voxsize=proj.voxel_size,
        img_fwd=xp.ones(num_events, dtype=xp.float32, device=dev),
    )

    vi = pv.ThreeAxisViewer([histo_back, lm_back, histo_back - lm_back])

# %%
if run_recon:
    recon = xp.ones(img_shape, dtype=xp.float32, device=dev)

    for i in range(num_iter):
        print(f"{(i+1):03}/{num_iter:03}", end="\r")

        exp = xp.clip(fwd_op(recon), 1e-6, None)
        grad = fwd_op.adjoint((exp - emission_data) / exp)
        step = recon / ones_back_tof
        recon -= step * grad

    print("")

# %%
# create header

subject = petsird.Subject(id="42")
institution = petsird.Institution(
    name="Ministry of Silly Walks",
    address="42 Silly Walks Street, Silly Walks City",
)

# %%
# create petsird scanner geometry

# scanner_geometry = get_scanner_geometry()

# TODO scanner_info.bulk_materials

# %%
# create non geometry related scanner information

num_energy_bins = 1

# TOF bin edges (in mm)
tofBinEdges = xp.linspace(
    -proj.tof_parameters.num_tofbins * proj.tof_parameters.tofbin_width / 2,
    proj.tof_parameters.num_tofbins * proj.tof_parameters.tofbin_width / 2,
    proj.tof_parameters.num_tofbins + 1,
    dtype="float32",
)

energyBinEdges = xp.linspace(430, 650, num_energy_bins, dtype="float32")

num_total_elements = proj.lor_descriptor.scanner.num_lor_endpoints

# create detectors efficiencies (all ones)
det_el_efficiencies = xp.ones(
    (num_total_elements, num_energy_bins), dtype="float32", device=dev
)

# setup the symmetry group ID LUT
# we only create one symmetry group ID (1) and set the group ID to -1 for block
# block pairs that are not in coincidence

num_SGIDs = 4

module_pair_sgid_lut = xp.full((num_blocks, num_blocks), -1, dtype="int32")

for bp in proj.lor_descriptor.all_block_pairs:
    # generate a random sgd
    sgid = sgid_from_module_pair(bp[0], bp[1], num_blocks)
    module_pair_sgid_lut[bp[0], bp[1]] = sgid
    module_pair_sgid_lut[bp[1], bp[0]] = sgid

num_el_per_module = proj.lor_descriptor.scanner.num_lor_endpoints_per_module[0]

module_pair_efficiencies_shape = (
    num_el_per_module,
    num_energy_bins,
    num_el_per_module,
    num_energy_bins,
)

module_pair_efficiencies_vector = []

for sgid in range(num_SGIDs):
    eff = module_pair_eff_from_sgd(sgid)
    vals = xp.full(module_pair_efficiencies_shape, eff, dtype="float32", device=dev)

    module_pair_efficiencies_vector.append(
        petsird.ModulePairEfficiencies(values=vals, sgid=sgid)
    )

det_effs = petsird.DetectionEfficiencies(
    det_el_efficiencies=det_el_efficiencies,
    module_pair_sgidlut=module_pair_sgid_lut,
    module_pair_efficiencies_vector=module_pair_efficiencies_vector,
)

# %%
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
        petsird.Coordinate(c=xp.asarray((-cw0 / 2, cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(
            c=xp.asarray((-cw0 / 2, cw1 / 2, -cw2 / 2), dtype="float32")
        ),
        petsird.Coordinate(
            c=xp.asarray((cw0 / 2, -cw1 / 2, -cw2 / 2), dtype="float32")
        ),
        petsird.Coordinate(c=xp.asarray((cw0 / 2, -cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=xp.asarray((cw0 / 2, cw1 / 2, cw2 / 2), dtype="float32")),
        petsird.Coordinate(c=xp.asarray((cw0 / 2, cw1 / 2, -cw2 / 2), dtype="float32")),
    ]
)
crystal = petsird.BoxSolidVolume(shape=crystal_shape, material_id=1)
# %%
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


# %%
# setup the PETSIRD scanner geometry
rep_module = petsird.ReplicatedDetectorModule(object=detector_module)

for i in range(num_blocks):
    transform = petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])

    rep_module.ids.append(i)
    rep_module.transforms.append(
        petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])
    )

scanner_geometry = petsird.ScannerGeometry(replicated_modules=[rep_module], ids=[0])
# %%

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
    print("Writing LM file")
    with petsird.BinaryPETSIRDWriter(fname) as writer:
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
