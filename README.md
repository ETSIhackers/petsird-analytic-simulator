# Analytic simulation of petsird v0.2 data using parallelproj

## Create your conda / mamba environment

```
conda env create -f environment.yml
conda activate petsird-v0.2-parallelproj
cd python
```

## Simulate petsird LM data

```
python 01_analytic_petsird_lm_simulator.py
```

The simulation can be customized in many ways (number of counts to simulate,
uniform or non-uniform efficiencies ...) via command line options.

These option can be listed via
```
python 01_analytic_petsird_lm_simulator.py
```

## Run a listmode OSEM recon on the simulated

```
python 02_recon_block_scanner_listmode_data.py
```

Thes command line optione for the LM OSEM recon script can be listed via
```
python 02_recon_block_scanner_listmode_data.py -h
```

