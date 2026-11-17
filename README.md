# Analytic simulation of PETSIRD v0.13.3 data using parallelproj

This tool allows to "quickly" simulate and reconstruct PETSIRD TOF listmode data
using a simplified physics model (true unscattered coincidences including
attenuation, and random coincidences) and non-trivial
detection efficiencies.

One of its aims is to be able to generate data that can be used to test implementations
of software supporting the PETSIRD standard for listmode data.

Requirements:
- [PETSIRD](https://github.com/ETSInitiative/PETSIRD) v0.13.x (>= 0.13.3)
- [parallelproj](https://github.com/gschramm/parallelproj) v2.x (requires python >= 3.12)
- matplotlib (plots and the interactive 3-axis viewer in `python/viewer.py`)

## Create your conda / mamba environment

```
conda env create -f environment.yml
conda activate petsird-analytic-simulator
cd python
```

## Simulate PETSIRD LM data

```
python 01_analytic_petsird_lm_simulator.py
```

The simulation script creates a binary PETSIRD LM file and reference files for
testing readers / reconstructions of PETSIRD data, which are all stored in the
output directory (by default `data/sim_{phantom}_{num_true_counts}_{seed}_{backend}`,
where `backend` is `petsird` or `petsird_fast`, see `--petsird_backend`)

```
tree --charset=ascii data/sim_uniform_cylinder_4000000_0_petsird_fast

data/sim_uniform_cylinder_4000000_0_petsird_fast
|-- attenuation_image.nii
|-- ground_truth_image.nii
|-- petsird_geometry_and_events.png
|-- petsird_summary.txt
|-- prompts_randoms_expectation.npy
|-- randoms_expectation.json
|-- reference_sensitivity_image.nii
|-- sim_parameters.json
`-- simulated_petsird_lm_file.bin
```

| file | content |
|---|---|
| `simulated_petsird_lm_file.bin` | the PETSIRD listmode file |
| `sim_parameters.json` | the parameters of the simulation |
| `ground_truth_image.nii` | ground truth activity image, i.e. **before** the image-based resolution model (`--fwhm_mm`) which is part of the forward model of the simulation |
| `attenuation_image.nii` | attenuation image (1 / mm) of the water body of the phantom, to be used for attenuation correction |
| `reference_sensitivity_image.nii` | sensitivity image of the forward model of the simulation (efficiencies, attenuation, image-based resolution model) |
| `prompts_randoms_expectation.npy` | true (noise-free) expected randoms in the data bin (pair of detection bins and TOF bin) of every prompt, in the order of the prompts in the PETSIRD file: the additive contamination of a TOF LM reconstruction (multiply by the number of TOF bins for non-TOF) |
| `randoms_expectation.json` | true (noise-free) expectation of the randoms: random fraction, expected trues and randoms, (scaled) singles of all detection bins (to test randoms corrections and estimations) |
| `petsird_summary.txt` | (only with `--petsird_summary`) human-readable summary of the PETSIRD file (output of `petsird.helpers.analysis`, which reads all events with the yardl-generated Python code and is slow for large files) |
| `petsird_geometry_and_events.png` | visualization of the PETSIRD geometry and the first events (see below) |
| `lm_backprojection_tof.nii` | (only with `--save_lm_backprojection`) TOF LM backprojection of the events before writing them to PETSIRD |
| `reference_histogram_mlem_<n>_epochs.nii` | (only with `--num_epochs_mlem n`) MLEM reconstruction of the histogrammed data with the exact forward model of the simulation |

All images are NIfTI files (see [Image files](#image-files)), centered at (0,0,0)
with the voxel size `--voxel_size`.

**Randoms:** the prompts contain true coincidences and random coincidences with an
expected random fraction R / (T + R) given by `--random_fraction` (default 0.5, where
T are the (attenuated) trues and R the randoms). The expected randoms of a LOR (pair of
detection bins a, b of modules in coincidence, summed over all TOF bins) are the product
of the (scaled) singles `s_a * s_b`, and the randoms are uniformly distributed over the
TOF bins. The singles of a detecting element are its efficiency (0 for "dead" elements)
times 50% uniform + 50% flux from the activity in the FOV (the efficiency corrected fan
sum of its expected trues, normalized to mean 1), such that they vary with the
activity distribution and the efficiencies (see `singles_model` in
`python/simulation.py`). An independent Poisson realization of the randoms is stored as
delayed coincidences. The singles counts of all detecting elements in every time block
are stored as singles histograms (`singles_histogram_level` ALL, ideal singles: same
energy window as the coincidences, no dead time). They are Poisson realizations of
constant singles rates `S` such that the randoms are `2 tau T S_a S_b` (acquisition time
`T`), where the coincidence window `2 tau` is the range covered by the TOF bins
(`(t1 - t2)` range `= 2 x TOF bin range / c`, ≈ 1.02 ns for the defaults). The true
singles, singles rates and coincidence window are saved in `randoms_expectation.json`.
With `--random_fraction 0`, no randoms, no delayeds and no singles are simulated.

`petsird_geometry_and_events.png` visualizes (transaxial, sagittal and 3D views) the PETSIRD scanner geometry as written
to the file (detecting element boxes with front faces and box axes, average detection
points, module and detecting element indices) and the first events (detection bins and
the emission position decoded from the TOF bin following the PETSIRD definition) on top
of the ground truth activity (see `python/geometry_plots.py`).

The following phantoms are available (`--phantom`):

- `uniform_cylinder`: uniform cylinder centered in the field of view
- `offcenter_cylinder`: cylinder that is off-center (transaxially and axially) with
  3 hot spheres of different size (radius 8, 6, 4 mm), contrast (3:1, 2:1, 3:1) and
  position. Since the phantom has no symmetry, axis flips, swaps and mirroring
  (e.g. in readers of the PETSIRD data) are visible in the reconstructions.
- `points`: point sources (in a centered water cylinder)
- `uniform_cylinder_text`: `uniform_cylinder` with hidden low-contrast letters
  (`--phantom_text`, default `PET`, `--phantom_text_contrast`, default 0.15), see
  [Reconstruction challenge](#reconstruction-challenge)

`--phantom_image activity.nii` (and optionally `--phantom_attenuation_image mu.nii`,
default: water where the activity is > 0) simulates any activity image (NIfTI in the
scanner coordinate system, resampled onto the simulation grid if needed) instead.

All phantoms are embedded in a water body (linear attenuation coefficient
0.0096 / mm) whose attenuation is included in the simulated data. The attenuation
image (in 1 / mm) is saved as `attenuation_image.nii` and can be used for
attenuation correction in the reconstruction (see below).

The simulation can be customized in many ways (number of counts to simulate,
uniform or non-uniform efficiencies ...) via command line options.

These options can be listed via

```
python 01_analytic_petsird_lm_simulator.py -h

Output Configuration:
  --fname FNAME         name of the output LM file (default: simulated_petsird_lm_file.bin)
  --output_dir OUTPUT_DIR
                        directory to save output files (default: None)
  --skip_writing        skip writing the LM data to a file (default: False)
  --petsird_summary     write a human-readable summary of the PETSIRD file (petsird_summary.txt,
                        output of petsird.helpers.analysis, which reads all events with the
                        yardl-generated Python code and is slow for large files) (default: False)

Image and Phantom Configuration:
  --phantom {uniform_cylinder,offcenter_cylinder,points,uniform_cylinder_text}
                        phantom to simulate (ignored if --phantom_image is given) (default:
                        uniform_cylinder)
  --phantom_text PHANTOM_TEXT
                        letters of the phantom uniform_cylinder_text (one per axial section)
                        (default: PET)
  --phantom_text_contrast PHANTOM_TEXT_CONTRAST
                        contrast of the letters of the phantom uniform_cylinder_text (default:
                        0.15)
  --phantom_image PHANTOM_IMAGE
                        activity image (NIfTI, in the scanner coordinate system) to simulate
                        instead of --phantom, resampled onto the simulation grid if needed
                        (default: None)
  --phantom_attenuation_image PHANTOM_ATTENUATION_IMAGE
                        attenuation image (NIfTI, 1/mm) of --phantom_image (default: water where
                        the activity is > 0) (default: None)
  --img_shape IMG_SHAPE
                        shape of the image to simulate (default: (55, 55, 21))
  --voxel_size VOXEL_SIZE
                        voxel size in mm used in the simulation (default: (2.0, 2.0, 2.0))

Physics / simulation parameters:
  --fwhm_mm FWHM_MM     FWHM of the image space resolution model in mm (default: 2.5)
  --tof_fwhm_mm TOF_FWHM_MM
                        FWHM of the TOF resolution model in mm (default: 10.0)
  --num_true_counts NUM_TRUE_COUNTS
                        (expected) number of true coincidences to simulate (default: 4000000)
  --random_fraction RANDOM_FRACTION
                        (expected) random fraction R / (T + R) of the prompts, where R / T are the
                        total randoms / (attenuated) trues; the randoms are also stored as
                        independent realization in the delayed coincidences (0 means no randoms
                        and no delayeds) (default: 0.5)

Scanner Geometry:
  --modules_per_panel {1,2}
                        2: 6 flat panels of 2 modules around the scanner axis (only symmetric
                        under a rotation by one panel, i.e. 60 deg), 1: 12 modules equally spaced
                        around the axis (default: 2)
  --panel_gap PANEL_GAP
                        gap between the modules of a panel (mm) (default: 4.0)
  --num_axial_modules NUM_AXIAL_MODULES
                        number of axial module positions (the transaxial arrangement of 12
                        modules is repeated at each axial position) (default: 3)
  --axial_gap AXIAL_GAP
                        gap between neighboring axial module positions (mm) (default: 2.0)
  --block_shape BLOCK_SHAPE
                        number of detecting elements (4.5 x 10 x 4.5 mm) per module (tangential,
                        radial, axial) (default: (10, 2, 3))

Detection Efficiency Parameters:
  --efficiency_model {structured,random}
                        structured: exaggerated, structured detecting element and module pair
                        efficiencies (sectors, rings, axial profiles, a dead patch; clear
                        artifacts if they are ignored or misinterpreted), random: random
                        efficiencies (previous model) (default: structured)
  --uniform_crystal_eff
                        use uniform crystal efficiencies, otherwise use a random distribution
                        (default: False)
  --uniform_sg_eff      use uniform symmetry group (module pair) efficiencies, otherwise pseudo
                        random pattern is used (default: False)

Time Block Configuration:
  --num_time_blocks NUM_TIME_BLOCKS
                        number of time blocks to split the data into (default: 100)
  --event_block_duration EVENT_BLOCK_DURATION
                        duration of each time block in ms (default: 100)

Reconstruction and Analysis:
  --num_epochs_mlem NUM_EPOCHS_MLEM
                        number of epochs for MLEM reconstruction of histogrammed data, 0 means no
                        MLEM reconstruction (default: 0)
  --save_lm_backprojection
                        save the TOF LM backprojection of the simulated events (before writing them
                        to PETSIRD, used to check the written events) (default: False)

Visualization and Debugging:
  --skip_plots          skip plotting the PETSIRD scanner geometry and events (default: False)

General Options:
  --seed SEED           random seed for reproducibility (default: 0)
  --petsird_backend {auto,petsird,petsird_fast}
                        write the time blocks with the yardl-generated Python code (petsird) or
                        the optional compiled package petsird_fast (C++, much faster, byte-
                        identical files); auto: petsird_fast if installed (default: auto)
```

**Note:** The "reference" MLEM using histogrammed data is only run if a
value > 0 is given via `--num_epochs_mlem`. Otherwise it is skipped to save
time.

### Simulated scanner

36 modules of 10 x 2 x 3 (tangential x radial x axial) LYSO crystals (4.5 x 10 x 4.5 mm),
2160 crystals in total:
- transaxially, 12 modules form 6 flat panels of 2 modules each (`--modules_per_panel 2`,
  panel center 100 mm from the scanner axis, 4 mm gap between the modules of a panel)
- axially, this arrangement is repeated at 3 axial module positions
  (`--num_axial_modules 3`, 2 mm gap, axial extent 44.5 mm, centered at z = 0)

Module index = `12 * axial module index + transaxial module index`. The transaxial
module index increases counterclockwise (also within a panel), so transaxial modules
`2k` and `2k + 1` form panel `k`, rotated by `60 * k` degrees; the axial module index
increases along +z. Module pairs with a transaxial circular index distance of at least 3
are in coincidence, for all axial module offsets (0, 1, 2).

`--modules_per_panel 1 --num_axial_modules 1 --block_shape 10,2,9` gives the original
geometry (12 modules of 10 x 2 x 9 crystals equally spaced around the axis, facing the
scanner center).

`petsird_geometry_and_events.png` (written by the reconstruction) shows the geometry as
encoded in the PETSIRD file (transaxial, sagittal and 3D views).

The geometry is deliberately not very symmetric:
- it is only symmetric under a rotation by one panel (60 deg), not by one module
  (30 deg). Two module pairs with the same transaxial module offset are therefore in
  general not equivalent (e.g. transaxial modules 3 / 0 are 2 panels apart, 4 / 1 are 1
  panel apart, both have offset 3).
- module pairs (not the scanner) are in addition symmetric under an axial shift by one
  axial module position: the module pairs (12 + i, 12 + j) and (i, j) are equivalent.
- the axial mirror (z -> -z) is not used: it reverses the order of the detecting
  elements within a module, which is neither the same nor the transposed module pair
  efficiency matrix.

Readers that assume that the module pair efficiencies only depend on the module index
difference, or that the SGID LUT is one triangle, give wrong results.

### Notes on the PETSIRD encoding

- Coincidences are stored ordered, i.e. `detection_bins[0] >= detection_bins[1]`.
  The TOF index `tof_idx` refers to `(t1 - t2) * c / 2` where `t1` / `t2` are the
  arrival times at the first / second detection bin.
- All tables over pairs of module (types) (TOF bin edges and resolution, prompt events,
  module pair SGID LUT, module pair efficiencies) are lower-triangular.
- The detecting elements are boxes oriented along their local x-axis (pointing away
  from the scanner center, along the normal of their panel) with the average detection
  point at the crystal center.
  Their material (LYSO, `material_id` 1) is defined in `ScannerInformation.bulk_materials`.
- The detection efficiencies consist of detection bin (crystal) efficiencies,
  module pair (symmetry group) efficiencies and a calibration factor of 1.
  Module pairs are grouped into symmetry groups by their class `(p1, p2, d, da)` with
  the position `p1` / `p2` of the first / second module in its panel (transaxial module
  index mod 2), the transaxial module offset `d = (t1 - t2) mod 12` and the axial module
  offset `da = a1 - a2 >= 0` (`t`, `a`: transaxial / axial module index, first module =
  module with the higher index, i.e. never at a lower axial module position).
  A rotation by one panel maps transaxial module `t` to `t + 2` and an axial shift maps
  axial module `a` to `a + 1`, both keep the class. For modules at the same axial module
  position (`da = 0`), a rotation can swap the order of the modules of a pair, which maps
  class `(p1, p2, d, 0)` to `(p2, p1, 12 - d, 0)` with the transposed module pair
  efficiency matrix `values[bin in first module][bin in second module]`. For `da > 0`,
  the first module is always the one at the higher axial module position, so no
  symmetry swaps the order (no transposed SGIDs).
  The SGIDs are ordered by `(da, d, p1, p2)`, i.e. `SGID = 14 * da + 2 * (d - 3) + p1`,
  which gives 42 symmetry groups (7 with `--modules_per_panel 1 --num_axial_modules 1`,
  one per offset):

  | SGID (+ 14 `da`) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | class `(p1, p2, d)` | 0,1,3 | 1,0,3 | 0,0,4 | 1,1,4 | 0,1,5 | 1,0,5 | 0,0,6 | 1,1,6 | 0,1,7 | 1,0,7 | 0,0,8 | 1,1,8 | 0,1,9 | 1,0,9 |
  | transposed SGID (`da = 0` only) | 13 | 12 | 10 | 11 | 9 | 8 | 6 (symmetric) | 7 (symmetric) | 5 | 4 | 2 | 3 | 1 | 0 |

  The 36 x 36 SGID LUT consists of 3 x 3 blocks of 12 x 12 (axial module positions of
  the first / second module):

  | | second module at axial position 0 | 1 | 2 |
  |---|---|---|---|
  | **first module at axial position 0** | lower triangle, SGIDs 0-13 | | |
  | **1** | full block, SGIDs 14-27 | lower triangle, SGIDs 0-13 | |
  | **2** | full block, SGIDs 28-41 | full block, SGIDs 14-27 | lower triangle, SGIDs 0-13 |

  The lower triangles of the diagonal blocks (rows: first transaxial module, columns:
  second transaxial module, `.`: not in coincidence) are

  ```
       0  1  2  3  4  5  6  7  8
   3:  1
   4:  2  0
   5:  5  3  1
   6:  6  4  2  0
   7:  9  7  5  3  1
   8: 10  8  6  4  2  0
   9: 13 11  9  7  5  3  1
  10:  . 12 10  8  6  4  2  0
  11:  .  . 13 11  9  7  5  3  1
  ```

  and the full off-diagonal blocks contain `14 * da + 2 * (d - 3) + p1` for all
  transaxial module pairs with a circular distance of at least 3 (also above their
  diagonal), e.g. for `da = 1`, first transaxial module 0, second transaxial module 3:
  `d = 9`, SGID `14 + 12 + 0 = 26`.
- The efficiencies are non-physical and deliberately exaggerated and structured
  (`--efficiency_model structured`, default, see `python/efficiencies.py`), such that
  ignoring or misinterpreting them gives clear, recognizable artifacts, and with as few
  symmetries as possible, such that flipped / swapped indices or transposed matrices
  change the result:
  - crystal efficiency = module gain (sinusoidal over the transaxial module index x
    axial module gain 1.0 / 1.71 / 1.65, module 2 x 1.8) x tangential ramp (0.5 ... 1.3)
    x layer factor (front 1.4, back 0.6) x axial profile (0.96 / 1.16 / 0.36) x 10%
    jitter, and a dead patch of 3 x 3 crystals in the front layer of module 19
    (transaxial module 7 at the central axial module position)
  - module pair efficiency = level of the class (alternating with the transaxial module
    offset, with opposite phase for odd axial module offsets, different for the two
    classes of an offset, lower for larger axial module offsets) x a
    function of the tangential positions of both crystals (i.e. of the radial position
    of the LOR) x an axial factor (oblique LORs lower) x a strongly non-symmetric layer
    factor (front -> back 1.6, back -> front 0.4) x 10% jitter
  - `--efficiency_model random` gives the previous random efficiencies (crystal
    efficiencies uniform in [0.2, 2.2], x3 / x(1/3) in the first / last module, some
    dead crystals, random non-symmetric module pair matrices)

## Reconstructions of PETSIRD listmode files

The reconstruction is meant as a reference implementation of reading and
reconstructing PETSIRD data. It supports multiple module types, energy bins and time
blocks, the full detection efficiency model and TOF. It stops with an error for
features that would lead to wrong reconstructions:

- bed or gantry movement time blocks
- weighted event time blocks
- different TOF bin edges / TOF resolutions for different module type pairs
- non-equidistant or asymmetric TOF bins

and warns about information that is ignored:

- gantry alignment (the image is reconstructed in the PET gantry coordinate system)
- delayed events, if no randoms correction is used, and dead time time blocks (no dead time correction)
- single, triple and quadruple events, singles histograms per singles unit (level
  MODULE), external signals and non-detecting volumes

The time blocks are read with the yardl-generated Python code (about 3 µs per event).
If the optional compiled package [`petsird_fast`](petsird_fast/README.md) (prototype,
yardl-generated C++ code via `libpetsird`) is installed, it is used automatically
instead, which is about 30x faster end to end with identical results. Both simulator
and reconstruction have the option `--petsird_backend {auto,petsird,petsird_fast}` to
choose the backend explicitly (the simulator writes byte-identical files with both).

After the reconstruction, the same visualization of the PETSIRD geometry and the first
events is shown on top of the reconstructed image (and saved as
`my_petsird_file_geometry_and_events_<fingerprint>_<hash>.png`).

All files written by the reconstruction script (cached sensitivity images,
backprojections and reconstructions) are tagged with a fingerprint of the listmode file
(`<fingerprint>`, computed from the file size and its first and last 4 MiB) and a hash
of the reconstruction parameters (`<hash>`). Re-running a simulation into the same
directory therefore never reuses cached files of the previous listmode file. Cache files
are written atomically, and a cache file that can not be loaded is recomputed. The
events are not cached: they are read from the listmode file in every run.

### Time ranges

```
python 02_reconstruct_petsird.py my_petsird_file.bin --time_range_ms 0 500
```

only uses the time blocks (events, singles histograms, ...) whose time interval lies
within [0, 500] ms. The backprojections, reconstructions and figures are tagged with
`_t0-500ms`. With the oracle randoms correction (`--randoms_expectation`), the true
expected randoms of the selected prompts (over the whole acquisition) are scaled with the
fraction of the acquisition time that is selected (the simulated randoms are constant
over time).

### Detection efficiency factors

```
python 02_reconstruct_petsird.py my_petsird_file.bin --efficiencies {all,module_pairs,detection_bins,none}
```

selects which of the encoded detection efficiency factors are used, consistently in the
forward model of the events and in the sensitivity image:

| mode | factor of an event / LOR |
|---|---|
| `all` (default) | calibration factor x detection bin efficiencies x module pair (SGID) efficiency |
| `module_pairs` | module pair (SGID) efficiency only |
| `detection_bins` | detection bin efficiencies only |
| `none` | 1 (purely geometric sensitivity image) |

In all modes, only module pairs in coincidence (SGID >= 0 in the module pair SGID LUT)
are used. Reconstructions with only some (or none) of the factors show the artifacts of
ignoring the others, e.g. for the simulated (structured) efficiencies: a cold streak
from the dead patch (whose LORs are in the sensitivity image but have no events), sector
artifacts from the module gains, axial bands from the axial profile, and ring artifacts
from the module pair efficiencies.
This helps to check whether the encoded efficiencies are interpreted correctly. The
reconstructions are tagged with `_effs_<mode>`.

### NON-TOF NAC recon

```
python 02_reconstruct_petsird.py my_petsird_file.bin --non-tof
```

### TOF NAC recon

```
python 02_reconstruct_petsird.py my_petsird_file.bin
```

### Creation of a simple 2-class air/water attenuation image

To create a simple 2-class attenuation image, use

```
python 03_create_two_class_att_img.py my_petsird_file.bin
```

which uses the TOF backprojection (`my_petsird_file_tof_backproj_<fingerprint>_<hash>.nii`)
created by the TOF NAC recon script above. If several reconstructions with different
settings were run, pass the TOF backprojection `.nii` file of the TOF NAC recon instead.
If no `--threshold` is given, Otsu's method is used for the segmentation.
The attenuation image is saved as `my_petsird_file_att_img_<fingerprint>_<hash>.nii`.


### NON-TOF AC recon

```
python 02_reconstruct_petsird.py my_petsird_file.bin --non-tof --attenuation_image my_att_img.nii
```

### TOF AC recon

```
python 02_reconstruct_petsird.py my_petsird_file.bin --attenuation_image my_att_img.nii
```

For data created by the simulator, use the simulated attenuation image
(`attenuation_image.nii` in the output directory) for a quantitatively correct
attenuation correction. The 2-class attenuation image created by
`03_create_two_class_att_img.py` is only an approximation of it.

The attenuation image defines the image grid of the reconstruction (shape, voxel size
and position, from its NIfTI affine) unless `--img_shape` / `--voxel_size` are given.
In that case (or for rotated grids), the attenuation image is resampled onto the
reconstruction grid (centered at the scanner iso center) with trilinear interpolation,
so an attenuation image on any grid in the scanner coordinate system can be used.

## Reconstruction challenge

The phantom `uniform_cylinder_text` hides a 3-letter word in a uniform cylinder: one
letter (cap height 40 mm, 6 mm thick axially, 15% contrast) in each of 3 equally spaced
axial sections (z = -14, 0, +14 mm for the default image, about the centers of the 3
axial module positions), the first letter at the lowest z. The letters read along +x
and are upright when the transaxial slices are shown with x to the right and y
**downwards** (as in `viewer.py`).

The contrast is chosen such that the letters are only readable if the detection
efficiencies are decoded correctly: reconstructions that ignore the module pair
efficiencies or the detection bin efficiencies, or all of them (`02 --efficiencies`),
are dominated by artifacts (rings, sectors, a strong central blob) that hide the letters
completely. Mirrored letters or a wrong letter order reveal flipped axes or swapped
axial module positions.

Challenge file (20M trues, random fraction 0.2, 25M prompts + 5M delayeds, ~150 MB):

```
python 01_analytic_petsird_lm_simulator.py --phantom uniform_cylinder_text --num_true_counts 20000000 --random_fraction 0.2
```

Give participants the listmode file and the attenuation image. With this repository's
reconstruction (`02 --attenuation_image attenuation_image.nii --randoms_from_singles
--num_epochs 2 --num_subsets 10`), the letters are readable after a Gaussian post filter
of about 4 mm FWHM (box "FWHM [mm]" of the viewer).

## Image files

All images (simulator: ground truth, attenuation, sensitivity, LM back projection,
histogram MLEM; `02`: sensitivity images, back projections, reconstructions; `03`:
attenuation images; `04`: histo-images) are NIfTI-1 files (`.nii`, float32), written and
read with `python/image_io.py` (nibabel). Arrays that are not images (e.g.
`prompts_randoms_expectation.npy`, one value per prompt, and the estimated singles
`*_randoms_singles_*.npy`) are NumPy files.

Conventions:
- the NIfTI affine maps the voxel indices `(i, j, k)` of `img[i, j, k]` to the center of
  the voxel in the **scanner coordinate system of the PETSIRD file** (in mm, the
  coordinate system of the detecting elements / ADPs): `(x, y, z, 1) = affine @ (i, j, k, 1)`.
  All grids of this repository are axis-aligned:
  `affine = [[vx, 0, 0, ox], [0, vy, 0, oy], [0, 0, vz, oz], [0, 0, 0, 1]]` with the
  voxel size `(vx, vy, vz)` and the center `(ox, oy, oz)` of voxel `[0, 0, 0]`.
- `qform_code = sform_code = 1` ("scanner-based anatomical coordinates"), spatial unit
  mm. The PETSIRD scanner coordinate system is **not** a patient coordinate system: the
  orientation labels (L/R, A/P, S/I) that NIfTI viewers show have no meaning here.
- units of the values (not stored in the files): attenuation images in 1 / mm, activity
  (ground truth, reconstructions) in arbitrary units such that the forward model of the
  simulation (with the sensitivity image of the same grid) gives the expected counts,
  back projections and histo-images in counts (weighted counts with
  `--efficiency_weighted` / `--attenuation_image`).

Example (Python):

```python
import nibabel as nib
nii = nib.load("data/sim_offcenter_cylinder_4000000_0_petsird_fast/attenuation_image.nii")
img, affine = nii.get_fdata(), nii.affine   # img[i, j, k], voxel center = affine @ (i, j, k, 1)
```

### Randoms correction

```
python 02_reconstruct_petsird.py my_petsird_file.bin --attenuation_image my_att_img.nii --randoms_from_delayeds ml
```

(or `--randoms_from_singles`, see below) estimates the randoms from the delayed
coincidences in the PETSIRD file and uses them
as additive contamination in the LM OSEM reconstruction (see `python/randoms.py` and
the theory in [`docs/randoms_estimation.md`](docs/randoms_estimation.md)).
The expected randoms of a pair of detection bins (a, b) in coincidence are modeled as
`s_a * s_b` (uniform over the TOF bins), where the singles `s` are estimated from the
fan sums of the delayeds `F_a` (number of delayeds of detection bin `a`). The fans
(detection bins in coincidence) follow from the module pair SGID LUT of the PETSIRD
file. Two estimators are available:

- `ml` (default for `--randoms_from_delayeds`): maximum likelihood estimate (Poisson
  model of the delayeds), `--randoms_ml_iterations` (default 10) monotone iterations
  initialized with the Casey-type estimate
- `casey`: Casey-type fan-sum estimate `s_a = F_a / sqrt(sum_{b in fan(a)} F_b)`, which is
  only exact if all fan sums of the singles are equal (biased by up to 14% for module
  pairs of the simulated scanner, see `tests/study_randoms_estimation.py`)

The estimated singles are saved as
`my_petsird_file_randoms_singles_<method>_<fingerprint>.npy`. For simulated data,
`--randoms_from_singles` calculates the randoms from the singles histograms (per
detection bin, level ALL). For static scans (constant singles rates, the default
`--randoms_from_singles static`), the singles counts `C_a` are summed over all time
blocks and the randoms are `2 tau C_a C_b / T` (acquisition time `T`): one product per
event, like the estimates from the delayeds (about 100x faster than per time block for
100 time blocks). `--randoms_from_singles time_blocks` uses
`2 tau sum_k C_a,k C_b,k / dt_k` (singles counts `C` in time blocks `k` of duration
`dt_k`), which is also correct for time-varying singles rates (e.g. decay, for which the
static estimate is too low). PETSIRD has no field for the coincidence window
`2 tau`, which is therefore assumed to be the range covered by the TOF bins (true for the
simulated data; if TOF bins outside of the coincidence window are discarded, the randoms
have to be rescaled accordingly). Only one event and one singles energy bin are
supported.

`--randoms_expectation prompts_randoms_expectation.npy` uses the true expected randoms
of all prompts written by the simulator instead ("oracle" randoms correction). Without randoms correction, the reconstruction of data
with randoms is biased (a warning is shown). Detection bins without delayeds get
zero singles (which is the ML estimate), and the LM OSEM then uses a small eps in the
expected counts of all events (only needed for very low numbers of delayeds).
Reconstructions without attenuation correction are expected to be biased low
(and to show "cupping").

## TOF histo-images

```
python 04_tof_histo_image.py my_petsird_file.bin [--stream {prompts,delayeds}] [--tof_position {center,uniform}]
```

places every event of the prompt (or delayed) stream at the center of its TOF bin along
its LOR (most likely emission position, following the PETSIRD TOF definition) and
accumulates them in an image. This is a TOF backprojection with a point-like TOF kernel,
useful as a quick look at the data and to check the TOF convention of a file. The image
and a figure with three central cuts are saved next to the listmode file
(`<stem>_tof_histo_image_<stream>_<fingerprint>.nii / .png`).

The histo-image is calculated with the TOF listmode back projector of parallelproj (the
same as in the reconstruction, with the TOF bins of the file) and a narrow TOF kernel
(see `python/histo_images.py`):
- `--tof_position center` (default): point-like kernel at the TOF bin center (emulated
  by subdividing every TOF bin into narrow bins with the same centers)
- `--tof_position uniform`: uniform kernel over the TOF bin (box of one bin width)

Both kernels are convolved with a narrow Gaussian (`--kernel_sigma_mm`, default half a
voxel), because the projector normalizes the kernel over the image planes crossed by the
LOR (up to sqrt(3) voxels apart), which a narrower kernel can miss. The image is in
counts (every event inside of the image contributes its weight). The TOF bins have to be
equidistant and symmetric around 0 (as for the reconstruction).
For the 25M prompts of the challenge file (see above), the histo-image takes 4 s
(4 CPU cores; previously 10 s with a NumPy implementation), the complete script 7 s
including reading the file with petsird_fast; with CuPy arrays the projector runs on
a GPU.

The centers of all TOF bins of a LOR are anchored at the LOR midpoint, and for detectors
arranged around the scanner axis the midpoints of many LORs coincide (e.g. at the
scanner center and on the scanner axis). With point-like kernels, the image therefore
shows a spike / lines at these positions. `--tof_position uniform` reduces them (central
voxel / neighborhood 1.45 instead of 1.6 for the challenge file); a remainder is a
property of the back projector (LORs through the scanner center all sample the central
voxel), as in the TOF backprojection of the reconstruction. A Gaussian post filter of
the image (`--postfilter_fwhm_mm FWHM`) removes it.
`--attenuation_image att.nii` (1 / mm, resampled onto the grid of the histo-image if needed) weights every
event with the inverse attenuation factor `1 / exp(-int mu)` of its LOR. The histo-image
is then approximately the activity times the geometric sensitivity without attenuation
(it is not divided by the sensitivity).
Non-TOF data are not supported.
Options: `--efficiency_weighted` (weights 1 / efficiency), `--time_range_ms`,
`--petsird_backend`.

## Encoded detection efficiencies

```
python 05_show_efficiencies.py my_petsird_file.bin [--energy_bin 0] [--fan_bins 0 270]
```

visualizes the detection efficiencies encoded in any PETSIRD file (only the header is
read) and writes a reference table to compare other readers with (all files next to the
listmode file, `<stem>_efficiencies_<fingerprint>_*`):

| output | content |
|---|---|
| `crystals.png` | detection bin efficiencies as image module x detecting element (selected energy bin), efficiency 0 marked in red |
| `sgid_lut.png` | module pair SGID LUT (lower-triangular, white: not in coincidence) |
| `module_pairs.png` | module pair efficiency matrix `values[element of the first module][element of the second module]` of every SGID |
| `fans.png` | total efficiency of all LORs of some detection bins (`--fan_bins`), as image module x detecting element of the second detection bin |
| `reference.csv` | probe and random LORs: detection bins, their (module, element, energy) decomposition and ADPs, SGID, detection bin and module pair efficiencies, calibration factor and total efficiency, computed with the official `petsird.helpers` |

All plots are in index space (module and detecting element numbers, as a reader indexes
the encoded arrays), which works for any arrangement of the modules; the positions of
the detection bins (ADPs) are in `reference.csv`. To check your own reader, compute the
columns of `reference.csv` for the same LORs and compare: the first differing column
shows which step of the decoding is wrong.

For the simulated data (structured efficiency model, detecting element index
`= 6 * tangential + 3 * layer + axial` for the 10 x 2 x 3 elements of a module), correct
decoding shows:
- `crystals.png`: in every module (row), a period of 6 elements (3 brighter front layer
  elements followed by 3 darker back layer elements, each with the axial profile
  medium / high / low) whose brightness increases along the element index (tangential
  ramp); smoothly varying module rows within each axial module position (sinusoidal
  gains, module 2 brightest), the axial module positions 1 and 2 (modules 12-35) brighter
  than position 0; and the dead patch (red) in module 19 at elements 24-26, 30-32 and
  36-38
- `sgid_lut.png`: the 3 x 3 block structure above: lower triangles with SGIDs 0-13 on
  the diagonal blocks, with SGIDs alternating between two values along each diagonal;
  full blocks with SGIDs 14-27 / 28-41 below; blocks with the same axial module offset
  identical
- `module_pairs.png`: a checkerboard of 3 x 3 blocks (front -> back layer blocks much
  higher than back -> front), different levels between the SGIDs, symmetric matrices for
  opposite modules at the same axial module position (SGIDs 6 and 7) and transposed
  matrices for the transposed SGIDs of the table above (only SGIDs 0-13)
- `fans.png`: only the modules in coincidence with the module of the detection bin
  (at all axial module positions), with the same 6-element periodicity and the dead
  patch

## Simulated data from the CI

Every CI run uploads all outputs of the simulation and the reconstructions (the
PETSIRD listmode file, ground truth / attenuation / sensitivity images,
reconstructions, figures, `sim_parameters.json` and `provenance.json` with the
commit and package versions) as a workflow artifact
(`petsird-simulation-<phantom>-<counts>-<commit>`, kept for 90 days). They can be
downloaded from the "Actions" tab of the repository (requires a GitHub login).
Datasets with other phantoms / number of counts can be created by running the
CI workflow manually ("Run workflow").

## Code structure

The scripts in `python/` only parse the command line and call functions of a few
flat modules (no installation needed, the scripts import them directly):

| module | content |
|---|---|
| `scanner.py` | definition of the simulated scanner (`ScannerConfig`), converted into the parallelproj geometry (used for simulating) and the PETSIRD `ScannerGeometry` (written) |
| `phantoms.py` | activity and attenuation images of the phantoms (`PHANTOMS`) |
| `efficiencies.py` | detection efficiency model, converted into the efficiencies of the simulated LORs and the PETSIRD `DetectionEfficiencies` |
| `simulation.py` | forward model, histogram to listmode conversion, ordering of the coincidences, PETSIRD header and writing |
| `petsird_reader.py` | reading and interpreting PETSIRD data (supported features, ADPs, efficiencies, TOF, events) |
| `reconstruction.py` | image grid, sensitivity image and LM OSEM reconstruction |
| `image_io.py` | NIfTI image files (grid in the PETSIRD scanner coordinate system), resampling of images onto other grids |
| `histo_images.py` | TOF histo-images: TOF listmode back projection (parallelproj) with a point-like or uniform TOF kernel |
| `efficiency_plots.py` | visualization of the encoded detection efficiencies and reference table of decoded LOR efficiencies (05) |
| `randoms.py` | randoms model (product of singles), fans of the detection bins, Casey-type and ML estimation of the singles from the delayeds, randoms from singles histograms |
| `geometry_plots.py` | visualization of the PETSIRD geometry and of events |
| `viewer.py` | interactive 3-axis viewer (incl. a box to apply an isotropic Gaussian filter with a given FWHM in mm) |
| `utils.py` | small generic helpers (version check, FWHM / sigma, file fingerprint, caching) |

## Tests

Unit tests (using a tiny scanner, run in a few seconds) are in `tests/test_*.py`:

```
python -m pytest
```

`tests/study_randoms_estimation.py OUTPUT_DIR` is a (slower) statistical study of the
randoms estimators (bias, variance and convergence for many noise realizations).

## Consistency checks

`tests/check_simulation_and_recon.py` checks a simulated PETSIRD file (and its
reconstruction outputs, if present) for consistency with the simulator, using
only petsird and parallelproj (see the docstring of the script).
These checks are run in the CI:

```
cd python
python 01_analytic_petsird_lm_simulator.py --output_dir ../ci_output --phantom offcenter_cylinder --num_true_counts 200000 --save_lm_backprojection --skip_plots
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --img_shape 55 55 19 --num_epochs 1
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --num_epochs 1 --attenuation_image ../ci_output/attenuation_image.nii --randoms_from_delayeds ml
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --num_epochs 1 --attenuation_image ../ci_output/attenuation_image.nii --randoms_from_singles
cd ..
python tests/check_simulation_and_recon.py ci_output
```
