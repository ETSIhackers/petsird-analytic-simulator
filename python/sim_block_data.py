# %%
import array_api_compat.numpy as xp

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

# %%
# input paraters

# grid shape of LOR endpoints forming a block module
block_shape = (3, 1, 2)
# spacing between LOR endpoints in a block module
block_spacing = (1.5, 1.2, 1.7)
# radius of the scanner
scanner_radius = 10

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

delta_phi = 2 * xp.pi / 12

# setup an affine transformation matrix to translate the block modules from the
# center to the radius of the scanner
aff_mat_trans = xp.eye(4, device=dev)
aff_mat_trans[1, -1] = scanner_radius

for phi in xp.linspace(0, 2 * xp.pi, 12, endpoint=False):
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

for j in range(12):
    block_pairs += [[j, (j + 3 + i) % 12] for i in range(7)]

lor_desc = parallelproj.EqualBlockPETLORDescriptor(
    scanner,
    xp.asarray(block_pairs, device=dev),
)

# %%
# Setup of a non-TOF projector
# ----------------------------
#
# Now that the LOR descriptor is defined, we can setup the projector.

img_shape = (28, 20, 3)
voxel_size = (0.5, 0.5, 1.0)
img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
img[4:-4, 4:-4, 1:2] = 50.0
img[8:-8, 8:-8, 1:2] = 100.0

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
    lor_desc.show_block_pair_lors(ax1, block_pair_nums=None, color=plt.cm.tab10(0))
    fig.show()


# %%
# Forward project an image full of ones. The forward projection has the
# shape (num_block_pairs, num_lors_per_block_pair)

img_fwd = proj(img)
print(img_fwd.shape)

# %%
# Backproject a "histogram" full of ones ("sensitivity image" when attenuation
# and normalization are ignored)

ones_back = proj.adjoint(xp.ones(proj.out_shape, dtype=xp.float32, device=dev))
print(ones_back.shape)

# %%
# Visualize the forward and backward projection results

if show_plots:
    fig3, ax3 = plt.subplots(figsize=(8, 2), tight_layout=True)
    ax3.imshow(parallelproj.to_numpy_array(img_fwd), cmap="Greys", aspect=3.0)
    ax3.set_xlabel("LOR number in block pair")
    ax3.set_ylabel("block pair")
    ax3.set_title("forward projection of ones")
    fig3.show()

    fig4, ax4 = plt.subplots(1, 3, figsize=(7, 3), tight_layout=True)
    vmin = float(xp.min(ones_back))
    vmax = float(xp.max(ones_back))
    for i in range(3):
        ax4[i].imshow(
            parallelproj.to_numpy_array(ones_back[:, :, i]),
            vmin=vmin,
            vmax=vmax,
            cmap="Greys",
        )
    ax4[1].set_title("back projection of ones")
    fig4.show()

# %%
# Setup of a TOF projector
# ------------------------
#
# Now that the LOR descriptor is defined, we can setup the projector.

proj_tof = parallelproj.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
proj_tof.tof_parameters = parallelproj.TOFParameters(
    num_tofbins=27, tofbin_width=0.8, sigma_tof=2.0, num_sigmas=3.0
)

assert proj_tof.adjointness_test(xp, dev)

# %%
# TOF forward project an image full of ones. The forward projection has the
# shape (num_block_pairs, num_lors_per_block_pair, num_tofbins)

img_fwd_tof = proj_tof(img)
print(img_fwd_tof.shape)

# %%
# TOF backproject a "TOF histogram" full of ones ("sensitivity image" when attenuation
# and normalization are ignored)

ones_back_tof = proj_tof.adjoint(
    xp.ones(proj_tof.out_shape, dtype=xp.float32, device=dev)
)
print(ones_back_tof.shape)

# %%
# Visualize the forward and backward projection results

if show_plots:
    fig5, ax5 = plt.subplots(figsize=(6, 3), tight_layout=True)
    ax5.plot(parallelproj.to_numpy_array(img_fwd_tof[0, 0, :]), ".-")
    ax5.set_xlabel("TOF bin")
    ax5.set_title("TOF profile of LOR 0 in block pair 0")
    fig5.show()

    fig6, ax6 = plt.subplots(1, 3, figsize=(7, 3), tight_layout=True)
    vmin = float(xp.min(ones_back_tof))
    vmax = float(xp.max(ones_back_tof))
    for i in range(3):
        ax6[i].imshow(
            parallelproj.to_numpy_array(ones_back_tof[:, :, i]),
            vmin=vmin,
            vmax=vmax,
            cmap="Greys",
        )
    ax6[1].set_title("TOF back projection of ones")
    fig6.show()

# %%
# put poisson noise on the forward projection

xp.random.seed(0)
histogrammed_contam = xp.full(
    img_fwd_tof.shape, img_fwd_tof.mean(), dtype=xp.float32, device=dev
)
histogrammed_data = xp.random.poisson(img_fwd_tof + histogrammed_contam)
# histogrammed_data = img_fwd_tof + histogrammed_contam

# %%
recon = xp.ones(img_shape, dtype=xp.float32, device=dev)
num_iter = 200

for i in range(num_iter):
    print(f"{(i+1):03}/{num_iter}", end="\r")
    exp = proj_tof(recon) + histogrammed_contam
    ratio = histogrammed_data / exp
    recon = recon * proj_tof.adjoint(ratio) / ones_back_tof

print("")
