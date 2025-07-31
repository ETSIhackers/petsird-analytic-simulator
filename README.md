# Analytic simulation of petsird v0.7.2 data using parallelproj

This tool allows to "quickly" simulate and recostruct PETSIRD TOF listmode data
using a simplified physics model (true coincidences only) and non-trivial
detection efficiencies.

One of it's aims is to be able to generate data that can be used to test implementation
of software supporting the PETSIRD standard for listmode data.

## Create your conda / mamba environment

```
conda env create -f environment.yml
conda activate petsird-analytic-simulator
cd python
```

## Simulate petsird LM data

```
python 01_analytic_petsird_lm_simulator.py
```

The simulation script creates a binary petsird LM file, but also many other
files (e.g. a reference sensitivity image) that are all stored in the output
directory

```
tree --charset=ascii data/sim_points_400000_0 

data/sim_points_400000_0
|-- ground_truth_image.npy
|-- reference_sensitivity_image.npy
|-- scanner_geometry.png
|-- sim_parameters.json
|-- simulated_petsird_lm_file.bin
|-- tof_profile_and_sensitivity_image.png
`-- zz.json
```

The simulation can be customized in many ways (number of counts to simulate,
uniform or non-uniform efficiencies ...) via command line options.

These option can be listed via

```
python 01_analytic_petsird_lm_simulator.py -h

[-h] [--fname FNAME] [--output_dir OUTPUT_DIR] [--num_true_counts NUM_TRUE_COUNTS] [--skip_plots]
[--check_backprojection] [--num_epochs_mlem NUM_EPOCHS_MLEM] [--skip_writing] [--fwhm_mm FWHM_MM]
[--tof_fwhm_mm TOF_FWHM_MM] [--seed SEED] [--uniform_crystal_eff] [--uniform_sg_eff] [--img_shape IMG_SHAPE]
[--voxel_size VOXEL_SIZE] [--phantom {uniform_cylinder,squares,points}] [--num_time_blocks NUM_TIME_BLOCKS]
[--event_block_duration EVENT_BLOCK_DURATION]

options:
  -h, --help            show this help message and exit
  --fname FNAME
  --output_dir OUTPUT_DIR
  --num_true_counts NUM_TRUE_COUNTS
  --skip_plots
  --check_backprojection
  --num_epochs_mlem NUM_EPOCHS_MLEM
  --skip_writing
  --fwhm_mm FWHM_MM
  --tof_fwhm_mm TOF_FWHM_MM
  --seed SEED
  --uniform_crystal_eff
  --uniform_sg_eff
  --img_shape IMG_SHAPE
  --voxel_size VOXEL_SIZE
  --phantom {uniform_cylinder,squares,points}
  --num_time_blocks NUM_TIME_BLOCKS
  --event_block_duration EVENT_BLOCK_DURATION
```

**Note:** The "reference" MLEM using histogrammed data is only run if a
value > 0 is given via `--num_epochs_mlem`. Otherwise it is skipped to save
time.

## NAC reconstructions a PETSIRD listmode file

**NON-TOF NAC recon**

```
python 02_reconstruct_petsird.py my_petsird_file.bin --non-tof
```

**TOF NAC recon**

```
python 02_reconstruct_petsird.py my_petsird_file.bin
```

## Creation of a simple 2-class air/water attenuation image

To create a simple 2-class attenuation image, use

```
python 03_create_two_class_att_img.py tof_back_proj_file.npy
```

which uses the TOF backprojection created by the TOF NAC recon script above.

## AC reconstructions a PETSIRD listmode file

**NON-TOF AC recon**

```
python 02_reconstruct_petsird.py my_petsird_file.bin --non-tof --attenuation_image my_att_img.npy
```

**TOF AC recon**

```
python 02_reconstruct_petsird.py my_petsird_file.bin --non-tof --attenuation_image my_att_img.npy
```



