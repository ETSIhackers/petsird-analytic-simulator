# Analytic simulation of PETSIRD v0.13.1 data using parallelproj

This tool allows to "quickly" simulate and reconstruct PETSIRD TOF listmode data
using a simplified physics model (true coincidences only) and non-trivial
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
|-- ground_truth_image.npy
|-- reference_sensitivity_image.npy
|-- scanner_geometry.png
|-- sim_parameters.json
|-- simulated_petsird_lm_file.bin
`-- tof_profile_and_sensitivity_image.png
```

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
  --phantom {uniform_cylinder,squares,points}
                        phantom to simulate (default: uniform_cylinder)
  --img_shape IMG_SHAPE
                        shape of the image to simulate (default: (55, 55, 19))
  --voxel_size VOXEL_SIZE
                        voxel size in mm used in the simulation (default: (2.0, 2.0, 2.0))

Physics / simulation parameters:
  --fwhm_mm FWHM_MM     FWHM of the image space resolution model in mm (default: 2.5)
  --tof_fwhm_mm TOF_FWHM_MM
                        FWHM of the TOF resolution model in mm (default: 20.0)
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

## Reconstructions of PETSIRD listmode files

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

## Consistency checks

`tests/check_simulation_and_recon.py` checks a simulated PETSIRD file (and its
reconstruction outputs, if present) for consistency with the simulator, using
only petsird and parallelproj (see the docstring of the script).
These checks are run in the CI:

```
cd python
python 01_analytic_petsird_lm_simulator.py --output_dir ../ci_output --phantom squares --num_true_counts 200000 --check_backprojection --skip_plots
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --img_shape 55 55 19 --num_epochs 1
cd ..
python tests/check_simulation_and_recon.py ci_output
```
