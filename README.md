# Analytic simulation of PETSIRD v0.13.1 data using parallelproj

This tool allows to "quickly" simulate and reconstruct PETSIRD TOF listmode data
using a simplified physics model (true unscattered coincidences only, including
attenuation) and non-trivial
detection efficiencies.

One of its aims is to be able to generate data that can be used to test implementations
of software supporting the PETSIRD standard for listmode data.

Requirements:
- [PETSIRD](https://github.com/ETSInitiative/PETSIRD) v0.13.x
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

The simulation script creates a binary PETSIRD LM file, but also many other
files (e.g. a reference sensitivity image) that are all stored in the output
directory (by default `data/sim_{phantom}_{num_true_counts}_{seed}`)

```
tree --charset=ascii data/sim_uniform_cylinder_4000000_0

data/sim_uniform_cylinder_4000000_0
|-- attenuation_image.npy
|-- ground_truth_image.npy
|-- petsird_geometry_and_events.png
|-- reference_sensitivity_image.npy
|-- reference_sensitivity_image_no_attenuation.npy
|-- sim_parameters.json
|-- simulated_petsird_lm_file.bin
`-- tof_profile_and_sensitivity_image.png
```

`petsird_geometry_and_events.png` visualizes the PETSIRD scanner geometry as written
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

All phantoms are embedded in a water body (linear attenuation coefficient
0.0096 / mm) whose attenuation is included in the simulated data. The attenuation
image (in 1 / mm) is saved as `attenuation_image.npy` and can be used for
attenuation correction in the reconstruction (see below).
`reference_sensitivity_image.npy` is the sensitivity image of the forward model
used for the simulation (including attenuation);
`reference_sensitivity_image_no_attenuation.npy` the one without attenuation
(i.e. the one of a reconstruction without attenuation correction).

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

Image and Phantom Configuration:
  --phantom {uniform_cylinder,offcenter_cylinder,points}
                        phantom to simulate (default: uniform_cylinder)
  --img_shape IMG_SHAPE
                        shape of the image to simulate (default: (55, 55, 19))
  --voxel_size VOXEL_SIZE
                        voxel size in mm used in the simulation (default: (2.0, 2.0, 2.0))

Physics / simulation parameters:
  --fwhm_mm FWHM_MM     FWHM of the image space resolution model in mm (default: 2.5)
  --tof_fwhm_mm TOF_FWHM_MM
                        FWHM of the TOF resolution model in mm (default: 10.0)
  --num_true_counts NUM_TRUE_COUNTS
                        number of true coincidences to simulate (default: 4000000)

Detection Efficiency Parameters:
  --uniform_crystal_eff
                        use uniform crystal efficiencies, otherwise use a random distribution
                        (default: False)
  --uniform_sg_eff      use uniform symmetry group (module pair) efficiencies, otherwise pseudo
                        random pattern is used (default: False)

Time Block Configuration:
  --num_time_blocks NUM_TIME_BLOCKS
                        number of time blocks to split the data into (default: 3)
  --event_block_duration EVENT_BLOCK_DURATION
                        duration of each time block in ms (default: 100)

Reconstruction and Analysis:
  --num_epochs_mlem NUM_EPOCHS_MLEM
                        number of epochs for MLEM reconstruction of histogrammed data, 0 means no
                        MLEM reconstruction (default: 0)
  --check_backprojection
                        check the backprojection of the TOF histogram and LM data (default: False)

Visualization and Debugging:
  --skip_plots          skip plotting the scanner geometry and TOF profile (default: False)

General Options:
  --seed SEED           random seed for reproducibility (default: 0)
```

**Note:** The "reference" MLEM using histogrammed data is only run if a
value > 0 is given via `--num_epochs_mlem`. Otherwise it is skipped to save
time.

### Notes on the PETSIRD encoding

- Coincidences are stored ordered, i.e. `detection_bins[0] >= detection_bins[1]`.
  The TOF index `tof_idx` refers to `(t1 - t2) * c / 2` where `t1` / `t2` are the
  arrival times at the first / second detection bin.
- All tables over pairs of module (types) (TOF bin edges and resolution, prompt events,
  module pair SGID LUT, module pair efficiencies) are lower-triangular.
- The detecting elements are boxes oriented along their local x-axis (pointing away
  from the scanner center) with the average detection point at the crystal center.
- The detection efficiencies consist of detection bin (crystal) efficiencies,
  module pair (symmetry group) efficiencies and a calibration factor of 1.
  The crystal efficiencies are random (including some "dead" crystals).
  Module pairs are grouped into symmetry groups by their offset
  `(i_mod_1 - i_mod_2) mod 12` (first module = module with the higher index).
  The module pair efficiency matrices `values[bin in first module][bin in second module]`
  are random and non-symmetric, such that readers that e.g. transpose them or mix up
  the module order produce wrong results. Consistent with the rotational symmetry of
  the scanner, the matrix of offset `12 - d` is the transpose of the one of offset `d`.

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
- delayed events (no randoms correction) and dead time time blocks (no dead time correction)
- single, triple and quadruple events, singles histograms, external signals and
  non-detecting volumes

After the reconstruction, the same visualization of the PETSIRD geometry and the first
events is shown on top of the reconstructed image (and saved as
`my_petsird_file_geometry_and_events_<fingerprint>_<hash>.png`).

All files written by the reconstruction script (cached detector positions, events
and sensitivity images, backprojections and reconstructions) are tagged with a
fingerprint of the listmode file (`<fingerprint>`, computed from the file size and
its first and last 4 MiB) and a hash of the reconstruction parameters (`<hash>`).
Re-running a simulation into the same directory therefore never reuses cached files
of the previous listmode file.

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

which uses the TOF backprojection (`my_petsird_file_tof_backproj_<fingerprint>_<hash>.npy`)
created by the TOF NAC recon script above. If several reconstructions with different
settings were run, pass the TOF backprojection `.npy` file of the TOF NAC recon instead.
If no `--threshold` is given, Otsu's method is used for the segmentation.
The attenuation image is saved as `my_petsird_file_att_img_<fingerprint>_<hash>.npy`.


### NON-TOF AC recon

```
python 02_reconstruct_petsird.py my_petsird_file.bin --non-tof --attenuation_image my_att_img.npy
```

### TOF AC recon

```
python 02_reconstruct_petsird.py my_petsird_file.bin --attenuation_image my_att_img.npy
```

For data created by the simulator, use the simulated attenuation image
(`attenuation_image.npy` in the output directory) for a quantitatively correct
attenuation correction. The 2-class attenuation image created by
`03_create_two_class_att_img.py` is only an approximation of it.
Reconstructions without attenuation correction are expected to be biased low
(and to show "cupping").

## Consistency checks

`tests/check_simulation_and_recon.py` checks a simulated PETSIRD file (and its
reconstruction outputs, if present) for consistency with the simulator, using
only petsird and parallelproj (see the docstring of the script).
These checks are run in the CI:

```
cd python
python 01_analytic_petsird_lm_simulator.py --output_dir ../ci_output --phantom offcenter_cylinder --num_true_counts 200000 --check_backprojection --skip_plots
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --img_shape 55 55 19 --num_epochs 1
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --img_shape 55 55 19 --num_epochs 1 --attenuation_image ../ci_output/attenuation_image.npy
cd ..
python tests/check_simulation_and_recon.py ci_output --require_all_references
```
