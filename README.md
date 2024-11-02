# PETSIRD v0.2 - parallelproj repository

The purpose of this repo is to provide a starting point for developing software that uses PETSIRD.
Here we will focus on python usecases using parallelproj projection library.

## Create your conda / mamba environment

```
conda env create -f environment.yml
```

## Run a first PETSIRD example

```
conda activate petsird-v0.2-parallelproj
cd python
python petsird_generator.py > test.bin
python petsird_analysis.py < test.bin
```

## Projects to work on

### Read scanner geometry from a PETSIRD file and visualize it

**TODO**: 
- start from this [example here](https://parallelproj.readthedocs.io/en/stable/auto_examples/01_pet_geometry/03_run_block_scanner.html#sphx-glr-auto-examples-01-pet-geometry-03-run-block-scanner-py)

### Simulation of PETSIRD LM data using parallelproj

Start from this [parallelproj example](https://parallelproj.readthedocs.io/en/stable/auto_examples/06_listmode_algorithms/01_listmode_mlem.html#sphx-glr-auto-examples-06-listmode-algorithms-01-listmode-mlem-py).

**TODO**: 
- rewrite (simplify) util function that converts sinogram to listmode including event detectors IDs instead of coordinates. See [here](https://parallelproj.readthedocs.io/en/stable/_modules/parallelproj/projectors.html#RegularPolygonPETProjector.convert_sinogram_to_listmode)
- write scanner geometry to file




