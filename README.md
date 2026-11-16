# Analytic simulation of PETSIRD v0.13.2 data using parallelproj

This tool allows to "quickly" simulate and reconstruct PETSIRD TOF listmode data
using a simplified physics model (true unscattered coincidences including
attenuation, and random coincidences) and non-trivial
detection efficiencies.

One of its aims is to be able to generate data that can be used to test implementations
of software supporting the PETSIRD standard for listmode data.

Requirements:
- [PETSIRD](https://github.com/ETSInitiative/PETSIRD) v0.13.x (>= 0.13.2)
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
|-- attenuation_image.npy
|-- ground_truth_image.npy
|-- petsird_geometry_and_events.png
|-- petsird_summary.txt
|-- prompts_randoms_expectation.npy
|-- randoms_expectation.json
|-- reference_sensitivity_image.npy
|-- sim_parameters.json
`-- simulated_petsird_lm_file.bin
```

| file | content |
|---|---|
| `simulated_petsird_lm_file.bin` | the PETSIRD listmode file |
| `sim_parameters.json` | the parameters of the simulation |
| `ground_truth_image.npy` | ground truth activity image, i.e. **before** the image-based resolution model (`--fwhm_mm`) which is part of the forward model of the simulation |
| `attenuation_image.npy` | attenuation image (1 / mm) of the water body of the phantom, to be used for attenuation correction |
| `reference_sensitivity_image.npy` | sensitivity image of the forward model of the simulation (efficiencies, attenuation, image-based resolution model) |
| `prompts_randoms_expectation.npy` | true (noise-free) expected randoms in the data bin (pair of detection bins and TOF bin) of every prompt, in the order of the prompts in the PETSIRD file: the additive contamination of a TOF LM reconstruction (multiply by the number of TOF bins for non-TOF) |
| `randoms_expectation.json` | true (noise-free) expectation of the randoms: random fraction, expected trues and randoms, (scaled) singles of all detection bins (to test randoms corrections and estimations) |
| `petsird_summary.txt` | (only with `--petsird_summary`) human-readable summary of the PETSIRD file (output of `petsird.helpers.analysis`, which reads all events with the yardl-generated Python code and is slow for large files) |
| `petsird_geometry_and_events.png` | visualization of the PETSIRD geometry and the first events (see below) |
| `lm_backprojection_tof.npy` | (only with `--save_lm_backprojection`) TOF LM backprojection of the events before writing them to PETSIRD |
| `reference_histogram_mlem_<n>_epochs.npy` | (only with `--num_epochs_mlem n`) MLEM reconstruction of the histogrammed data with the exact forward model of the simulation |

All images are centered at (0,0,0) and use the voxel size `--voxel_size`.

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

All phantoms are embedded in a water body (linear attenuation coefficient
0.0096 / mm) whose attenuation is included in the simulated data. The attenuation
image (in 1 / mm) is saved as `attenuation_image.npy` and can be used for
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
                        (expected) number of true coincidences to simulate (default: 4000000)
  --random_fraction RANDOM_FRACTION
                        (expected) random fraction R / (T + R) of the prompts, where R / T are the
                        total randoms / (attenuated) trues; the randoms are also stored as
                        independent realization in the delayed coincidences (0 means no randoms
                        and no delayeds) (default: 0.5)

Detection Efficiency Parameters:
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

### Notes on the PETSIRD encoding

- Coincidences are stored ordered, i.e. `detection_bins[0] >= detection_bins[1]`.
  The TOF index `tof_idx` refers to `(t1 - t2) * c / 2` where `t1` / `t2` are the
  arrival times at the first / second detection bin.
- All tables over pairs of module (types) (TOF bin edges and resolution, prompt events,
  module pair SGID LUT, module pair efficiencies) are lower-triangular.
- The detecting elements are boxes oriented along their local x-axis (pointing away
  from the scanner center) with the average detection point at the crystal center.
  Their material (LYSO, `material_id` 1) is defined in `ScannerInformation.bulk_materials`.
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

### Randoms correction

```
python 02_reconstruct_petsird.py my_petsird_file.bin --attenuation_image my_att_img.npy --randoms_from_delayeds ml
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
detection bin, level ALL) as `2 tau sum_k C_a,k C_b,k / dt_k` (singles counts `C` in
time blocks `k` of duration `dt_k`). PETSIRD has no field for the coincidence window
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
accumulates the positions in an image (trilinear or nearest-voxel splatting, see
`python/histo_images.py`). This is a fast TOF backprojection with a point-like TOF
kernel (8M events in a few seconds), useful as a quick look at the data and to check the
TOF convention of a file. The image and a figure with three central cuts are saved next
to the listmode file (`<stem>_tof_histo_image_<stream>_<fingerprint>.npy / .png`).

The centers of all TOF bins of a LOR are anchored at the LOR midpoint, and on a ring
scanner the midpoints of many LORs coincide (e.g. at the scanner center and on the
scanner axis). With exact TOF bin centers, the image therefore shows a spike / lines at
these positions. `--tof_position uniform` places the events uniformly within their TOF
bins instead, which removes most of this artifact. A (much faster) alternative that is
often good enough is a Gaussian post filter of the image, `--postfilter_fwhm_mm FWHM`.
Non-TOF data are not supported.
Options: `--efficiency_weighted` (weights 1 / efficiency), `--time_range_ms`,
`--petsird_backend`.

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
| `histo_images.py` | TOF histo-images: events at the center of their TOF bin, splatting into an image |
| `randoms.py` | randoms model (product of singles), fans of the detection bins, Casey-type and ML estimation of the singles from the delayeds, randoms from singles histograms |
| `geometry_plots.py` | visualization of the PETSIRD geometry and of events |
| `viewer.py` | interactive 3-axis viewer |
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
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --img_shape 55 55 19 --num_epochs 1 --attenuation_image ../ci_output/attenuation_image.npy --randoms_from_delayeds ml
python 02_reconstruct_petsird.py ../ci_output/simulated_petsird_lm_file.bin --img_shape 55 55 19 --num_epochs 1 --attenuation_image ../ci_output/attenuation_image.npy --randoms_from_singles
cd ..
python tests/check_simulation_and_recon.py ci_output
```
