# Analytic simulation of petsird v0.2 data using parallelproj

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
tree --charset=ascii my_lm_sim

my_lm_sim
|-- reference_histogram_mlem_50_epochs.npy
|-- reference_sensitivity_image.npy
|-- scanner_geometry.png
|-- sim_parameters.json
|-- simulated_lm_file.bin
`-- tof_profile_and_sensitivity_image.png
```

The simulation can be customized in many ways (number of counts to simulate,
uniform or non-uniform efficiencies ...) via command line options.

These option can be listed via

```
python 01_analytic_petsird_lm_simulator.py -h
```

**Note:** The "reference" MLEM using histogrammed data is only run if a
value > 0 is given via `--num_epochs_mlem`. Otherwise it is skipped to save
time.

## Run a listmode OSEM recon on the simulated

```
python 02_lm_osem_recon_simulated_data.py
```

Thes command line optione for the LM OSEM recon script can be listed via

```
python 02_lm_osem_recon_simulated_data.py -h
```
