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




