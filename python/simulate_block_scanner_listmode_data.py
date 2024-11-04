# %%
import array_api_compat.numpy as xp
import pymirc.viewer as pv
from array_api_compat import size

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
import petsird
import matplotlib.pyplot as plt
import math

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = "cuda"

show_plots = False
check_backprojection = False
run_recon = False

# %%
# input parameters

# grid shape of LOR endpoints forming a block module
block_shape = (10, 1, 3)
# spacing between LOR endpoints in a block module
block_spacing = (4.5, 4.5, 4.5)
# radius of the scanner
scanner_radius = 100
# number of modules
num_blocks = 12
# FWHM of the Gaussian res model in mm
fwhm_mm = 4.0


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
aff_mat_trans = xp.eye(4, device=dev)
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
        ]
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

img_shape = (50, 50, 6)
voxel_size = (2.0, 2.0, 2.0)
img = xp.full(img_shape, 0.01, dtype=xp.float32, device=dev)
img[4:-4, 4:-4, :] = 0.02
img[16:-16, 16:-16, 2:-2] = 0.04

# %%
# Setup of a TOF projector
# ------------------------
#
# Now that the LOR descriptor is defined, we can setup the projector.

proj = parallelproj.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
proj.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=21, tofbin_width=10.0, sigma_tof=12.0, num_sigmas=3.0
)

assert proj.adjointness_test(xp, dev)


# %%
# Visualize the projector geometry and all LORs

if show_plots:
    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")
    proj.show_geometry(ax0)
    proj.show_geometry(ax1)
    # lor_desc.show_block_pair_lors(
    #    ax1, block_pair_nums=xp.arange(7), color=plt.cm.tab10(0)
    # )
    fig.show()


# %%
sig = fwhm_mm / (2.35 * xp.asarray(voxel_size, device=dev))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

fwd_op = parallelproj.CompositeLinearOperator([proj, res_model])

# %%
# TOF forward project an image full of ones. The forward projection has the
# shape (num_block_pairs, num_lors_per_block_pair, num_tofbins)

img_fwd_tof = fwd_op(img)
print(img_fwd_tof.shape)

# %%
# TOF backproject a "TOF histogram" full of ones ("sensitivity image" when attenuation
# and normalization are ignored)

ones_back_tof = fwd_op.adjoint(xp.ones(fwd_op.out_shape, dtype=xp.float32, device=dev))
print(ones_back_tof.shape)

# %%
# Visualize the forward and backward projection results

if show_plots:
    fig5, ax5 = plt.subplots(figsize=(6, 3), tight_layout=True)
    ax5.plot(parallelproj.to_numpy_array(img_fwd_tof[3, 0, :]), ".-")
    ax5.set_xlabel("TOF bin")
    ax5.set_title("TOF profile of LOR 0 in block pair 3")
    fig5.show()

    fig6, ax6 = plt.subplots(1, 3, figsize=(7, 3), tight_layout=True)
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
    ax6[1].set_title("TOF back projection of ones")
    fig6.show()

# %%
# put poisson noise on the forward projection

xp.random.seed(0)
emission_data = xp.random.poisson(img_fwd_tof)
# emission_data = xp.zeros(img_fwd_tof.shape, dtype=xp.int32, device=dev)
# emission_data[3, 1, 10] = 1

# %%
# convert emission histogram to LM

tmp = xp.arange(proj.lor_descriptor.num_lorendpoints_per_block)
start_el, end_el = xp.meshgrid(tmp, tmp, indexing="ij")

start_el_arr = xp.reshape(start_el, (size(start_el),))
end_el_arr = xp.reshape(end_el, (size(end_el),))

num_events = emission_data.sum()
event_start_block = xp.zeros(num_events, dtype=xp.int16, device=dev)
event_start_el = xp.zeros(num_events, dtype=xp.int16, device=dev)
event_end_block = xp.zeros(num_events, dtype=xp.int16, device=dev)
event_end_el = xp.zeros(num_events, dtype=xp.int16, device=dev)
event_tof_bin = xp.zeros(num_events, dtype=xp.int16, device=dev)

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
        # event TOF bin
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

# %%
# do a sinogram and LM back projection of the emission data

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
    vi = pv.ThreeAxisViewer([histo_back, lm_back, histo_back - lm_back])

# %%
if run_recon:
    recon = xp.ones(img_shape, dtype=xp.float32, device=dev)
    num_iter = 20

    for i in range(num_iter):
        print(f"{(i+1):03}/{num_iter}", end="\r")
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

num_SGIDs = 1

module_pair_sgid_lut = xp.full((num_blocks, num_blocks), -1, dtype="int16")

for bp in proj.lor_descriptor.all_block_pairs:
    module_pair_sgid_lut[bp[0], bp[1]] = 0
    module_pair_sgid_lut[bp[1], bp[0]] = 0

num_el_per_module = proj.lor_descriptor.scanner.num_lor_endpoints_per_module[0]
module_pair_efficiencies = xp.ones(
    (num_el_per_module, num_energy_bins, num_el_per_module, num_energy_bins),
    dtype="float32",
    device=dev,
)

module_pair_efficiencies_vector = []

for i in range(num_SGIDs):
    module_pair_efficiencies_vector.append(
        petsird.ModulePairEfficiencies(values=module_pair_efficiencies, sgid=i)
    )

det_effs = petsird.DetectionEfficiencies(
    det_el_efficiencies=det_el_efficiencies,
    module_pair_sgidlut=module_pair_sgid_lut,
    module_pair_efficiencies_vector=module_pair_efficiencies_vector,
)


# %%
# setup the PETSIRD scanner geometry
rep_module = petsird.ReplicatedDetectorModule(object=detector_module)


for i in range(num_blocks):
    transform = petsird.RigidTransformation(matrix=module_transforms[i][:-1, :])

    rep_module.ids.append(i)
    rep_module.transforms.append(module_transforms[i])

# %%

# need energy bin info before being able to construct the detection efficiencies
# so we will construct a scanner without the efficiencies first
scanner = petsird.ScannerInformation(
    model_name="PETSIRD_TEST",
    scanner_geometry=scanner_geometry,
    tof_bin_edges=tofBinEdges,
    tof_resolution=2.35 * proj.tof_parameters.sigma_tof,  # FWHM in mm
    energy_bin_edges=energyBinEdges,
    energy_resolution_at_511=0.11,  # as fraction of 511
    event_time_block_duration=1,  # ms
)

scanner.detection_efficiencies = det_effs

# header = petsird.Header(
#        exam=petsird.ExamInformation(subject=subject, institution=institution),
#        scanner=get_scanner_info(),
#    )

# %%
# create petsird coincidence events - all in one timeblock without energy information


num_el_per_block = proj.lor_descriptor.num_lorendpoints_per_block

det_ID_start = event_start_block * num_el_per_block + event_start_el
det_ID_end = event_end_block * num_el_per_block + event_end_el

time_block_coinc_events = [
    petsird.CoincidenceEvent(
        detector_ids=[det_ID_start[i], det_ID_end[i]],
        tof_idx=event_tof_bin[i],
        energy_indices=[0, 0],
    )
    for i in range(num_events)
]
