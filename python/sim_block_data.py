# %%
import array_api_compat.numpy as xp
from array_api_compat import size

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
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
run_recon = False

# %%
# input paraters

# grid shape of LOR endpoints forming a block module
block_shape = (10, 1, 3)
# spacing between LOR endpoints in a block module
block_spacing = (4.5, 4.5, 4.5)
# radius of the scanner
scanner_radius = 100
# number of modules
num_modules = 12
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

delta_phi = 2 * xp.pi / num_modules

# setup an affine transformation matrix to translate the block modules from the
# center to the radius of the scanner
aff_mat_trans = xp.eye(4, device=dev)
aff_mat_trans[1, -1] = scanner_radius

for phi in xp.linspace(0, 2 * xp.pi, num_modules, endpoint=False):
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
    mods.append(
        parallelproj.BlockPETScannerModule(
            xp,
            dev,
            block_shape,
            block_spacing,
            affine_transformation_matrix=(aff_mat_rot @ aff_mat_trans),
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

for j in range(num_modules):
    block_pairs += [[j, (j + 3 + i) % num_modules] for i in range(7)]

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

proj = parallelproj.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
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
# Setup of a TOF projector
# ------------------------
#
# Now that the LOR descriptor is defined, we can setup the projector.

proj_tof = parallelproj.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
proj_tof.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=21, tofbin_width=10.0, sigma_tof=12.0, num_sigmas=3.0
)

assert proj_tof.adjointness_test(xp, dev)

# %%
sig = fwhm_mm / (2.35 * xp.asarray(voxel_size, device=dev))
res_model = parallelproj.GaussianFilterOperator(img_shape, sigma=sig)

fwd_op = parallelproj.CompositeLinearOperator([proj_tof, res_model])

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

# %%
# convert emission histogram to LM

tmp = xp.arange(proj_tof.lor_descriptor.num_lorendpoints_per_block)
start_el, end_el = xp.meshgrid(tmp, tmp, indexing="ij")

start_el_arr = xp.reshape(start_el, (size(start_el),))
end_el_arr = xp.reshape(end_el, (size(end_el),))

lm_event_table = xp.zeros((emission_data.sum(), 5), dtype=xp.int16)

event_counter = 0

for ibp, block_pair in enumerate(proj_tof.lor_descriptor.all_block_pairs):
    for it, tof_bin in enumerate(
        xp.arange(proj_tof.tof_parameters.num_tofbins)
        - proj_tof.tof_parameters.num_tofbins // 2
    ):
        ss = emission_data[ibp, :, it]
        num_events = ss.sum()
        inds = xp.repeat(xp.arange(ss.shape[0]), ss)

        # event start block
        lm_event_table[event_counter : (event_counter + num_events), 0] = block_pair[0]
        # event start element in block
        lm_event_table[event_counter : (event_counter + num_events), 1] = xp.take(
            start_el_arr, inds
        )
        # event end module
        lm_event_table[event_counter : (event_counter + num_events), 2] = block_pair[1]
        # event end element in block
        lm_event_table[event_counter : (event_counter + num_events), 3] = xp.take(
            end_el_arr, inds
        )
        # event TOF bin
        lm_event_table[event_counter : (event_counter + num_events), 4] = tof_bin

        event_counter += num_events

# shuffle lm_event_table along 0 axis
xp.random.shuffle(lm_event_table)


# %%
if run_recon:
    recon = xp.ones(img_shape, dtype=xp.float32, device=dev)
    num_iter = 20

    for i in range(num_iter):
        print(f"{(i+1):03}/{num_iter}", end="\r")
        exp = xp.clip(fwd_op(recon), 1e-6, None)
        grad = fwd_op.adjoint((exp - histogrammed_data) / exp)
        step = recon / ones_back_tof
        recon -= step * grad

    print("")

# %%
