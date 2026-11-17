"""consistency checks of a simulated PETSIRD file and (optionally) its reconstruction

Usage:
    python 01_analytic_petsird_lm_simulator.py --output_dir OUT --save_lm_backprojection ...
    python 02_reconstruct_petsird.py OUT/simulated_petsird_lm_file.bin ...   (optional)
    python tests/check_simulation_and_recon.py OUT

The checks of the PETSIRD file only use petsird (and its helpers) and parallelproj,
but not the readers in python/utils.py, such that they are independent of the
reconstruction code:

- the detecting element box satisfies the PETSIRD constraints
- all materials referenced by the scanner geometry are defined in bulk_materials
- all tables over module (type) pairs are lower-triangular
- all prompt and delayed coincidences are ordered (detection_bins[0] >= detection_bins[1])
- the numbers of prompts (trues + randoms) and delayeds (randoms) match their expectation
- the delayeds of all module pairs match the randoms expectation of the simulator
  (randoms_expectation.json: product of the singles of the detection bins) and are
  uniformly distributed over the TOF bins
- the expected randoms of all prompts (prompts_randoms_expectation.npy) are consistent
  with the singles of their detection bins (event order of the file)
- the singles histograms (one per time block, all detection bins) match the singles
  rates of the simulator, and the randoms from singles 2 tau sum_k C_a,k C_b,k / dt_k
  (coincidence window 2 tau = range of the TOF bins) match the true randoms per module
  pair and per prompt
- a TOF LM back projection of the events in the PETSIRD file matches the one
  calculated by the simulator before writing (checks detection bin swapping and
  TOF bin mirroring)
- the TOF information follows the PETSIRD definition: the emission positions decoded
  from the TOF bins (only using the PETSIRD specification, no parallelproj) agree with
  the positions expected from the ground truth activity along the LORs
- vectorized efficiencies match petsird.helpers.get_detection_efficiency

If reconstruction outputs exist, the LM back projections calculated from the PETSIRD
file are compared to the ones of the simulator, and the sensitivity image of a TOF
recon with attenuation correction (using the simulated attenuation image
attenuation_image.nii) has to match the simulator's (reference_sensitivity_image.nii).
All images are NIfTI files whose grids (affines) are checked for consistency.

If a recon estimated the randoms from the delayeds (--randoms_from_delayeds), the
module pair randoms of the ML estimate have to match the expectation within the Poisson
noise and their total the number of delayeds (Casey-type estimates are only reported).

The script exits with a non-zero exit code if any check fails.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import petsird
from parallelproj.projectors import ListmodePETProjector
from parallelproj.tof import TOFParameters
from petsird.helpers import (
    are_detections_ordered,
    expand_detection_bin,
    get_detection_efficiency,
)
from petsird.helpers.geometry import get_ADP
from petsird.validate.geometry import validate_detecting_element

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from image_io import grid_from_affine, load_image  # noqa: E402
from utils import FWHM_PER_SIGMA, lm_file_fingerprint  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("output_dir", type=Path, help="output directory of the simulator")
args = parser.parse_args()

out_dir: Path = args.output_dir
sim_params = json.loads((out_dir / "sim_parameters.json").read_text())
lm_file = out_dir / sim_params["fname"]

failed: list[str] = []


def check(condition: bool, msg: str) -> None:
    print(f"[{'OK' if condition else 'FAILED'}] {msg}")
    if not condition:
        failed.append(msg)


def rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max() / np.abs(a).max())


# %%
# header checks

reader = petsird.BinaryPETSIRDReader(str(lm_file))
header = reader.read_header()
scanner = header.scanner
geom = scanner.scanner_geometry

check(geom.number_of_module_types() == 1, "scanner has one module type")
rep_module = geom.replicated_modules[0]
det_els = rep_module.object.detecting_elements

try:
    # petsird.validate currently prints debug output
    with contextlib.redirect_stdout(io.StringIO()):
        validate_detecting_element(det_els.object.shape)
    check(True, "detecting element box satisfies PETSIRD constraints")
except ValueError as e:
    check(False, f"detecting element box satisfies PETSIRD constraints ({e})")

num_modules = len(rep_module.transforms)
num_det_els = len(det_els.transforms)
num_energy_bins = scanner.event_energy_bin_edges[0].number_of_bins()
num_det_bins = num_modules * num_det_els * num_energy_bins
num_bins_per_module_type = num_det_els * num_energy_bins
det_effs = scanner.detection_efficiencies
sgid_lut = det_effs.module_pair_sgidlut[0][0]

check(
    [len(row) for row in sgid_lut] == list(range(1, num_modules + 1)),
    "module pair SGID LUT is lower-triangular",
)
check(
    [len(x) for x in scanner.tof_bin_edges] == [1]
    and [len(x) for x in scanner.tof_resolution] == [1],
    "TOF tables are lower-triangular",
)
check(
    len(det_effs.detection_bin_efficiencies[0]) == num_det_bins,
    "one detection bin efficiency per detection bin",
)
check(det_effs.calibration_factor == 1.0, "calibration factor is 1")

materials = {m.id: m for m in scanner.bulk_materials}
referenced_materials = {
    rm.object.detecting_elements.object.material_id for rm in geom.replicated_modules
} | {v.material_id for v in (geom.non_detecting_volumes or [])}
check(
    referenced_materials <= set(materials)
    and all(
        len(m.atoms) == len(m.mass_fractions) and abs(sum(m.mass_fractions) - 1) < 1e-6
        for m in materials.values()
    ),
    f"referenced materials {sorted(referenced_materials)} are defined in bulk_materials "
    f"({', '.join(m.name for m in materials.values())})",
)
random_fraction = sim_params.get("random_fraction", 0.0)
expected_delayed_policy = (
    petsird.CoincidencePolicy.MULTIPLES_AS_ALL_COINCIDENCES
    if random_fraction > 0
    else petsird.CoincidencePolicy.NONE
)
check(
    scanner.prompt_event_policy == petsird.CoincidencePolicy.MULTIPLES_AS_ALL_COINCIDENCES
    and scanner.delayed_event_policy == expected_delayed_policy,
    "event policies",
)
if random_fraction > 0:
    check(
        scanner.singles_histogram_level == petsird.SinglesHistogramLevelType.ALL
        and len(scanner.singles_histogram_energy_bin_edges) == 1
        and np.array_equal(
            scanner.singles_histogram_energy_bin_edges[0].edges,
            scanner.event_energy_bin_edges[0].edges,
        ),
        "singles histograms per detection bin with the energy window of the coincidences",
    )

# %%
# read all events

det_bin_0, det_bin_1, tof_idx = [], [], []
delayeds = []
singles_counts, singles_durations_ms = [], []
for time_block in reader.read_time_blocks():
    if isinstance(time_block, petsird.TimeBlock.SinglesHistogramTimeBlock):
        singles_counts.append(np.asarray(time_block.value.singles_per_detection_bin_histogram[0]))
        interval = time_block.value.time_interval
        singles_durations_ms.append(interval.stop - interval.start)
    if isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
        for event in time_block.value.prompt_events[0][0]:
            det_bin_0.append(event.detection_bins[0])
            det_bin_1.append(event.detection_bins[1])
            tof_idx.append(event.tof_idx)
        if len(time_block.value.delayed_events) > 0:
            delayeds += [
                (e.detection_bins[0], e.detection_bins[1], e.tof_idx)
                for e in time_block.value.delayed_events[0][0]
            ]
det_bin_0, det_bin_1, tof_idx = map(np.asarray, (det_bin_0, det_bin_1, tof_idx))
num_events = det_bin_0.size

type_pair = petsird.TypeOfModulePair((0, 0))
check(
    num_events > 0
    and all(
        are_detections_ordered(type_pair, int(b0), int(b1))
        for b0, b1 in zip(det_bin_0, det_bin_1)
    ),
    f"all {num_events} prompt coincidences are ordered",
)
check(
    all(are_detections_ordered(type_pair, int(b0), int(b1)) for b0, b1, _ in delayeds),
    f"all {len(delayeds)} delayed coincidences are ordered",
)

# number of prompts (trues + randoms) and delayeds (randoms) vs. their expectation
randoms_info = json.loads((out_dir / "randoms_expectation.json").read_text())
expected_prompts = randoms_info["expected_trues"] + randoms_info["expected_randoms"]
expected_delayeds = randoms_info["expected_randoms"]
check(
    abs(num_events - expected_prompts) < 5 * np.sqrt(expected_prompts),
    f"number of prompts {num_events} matches the expectation {expected_prompts:.0f}",
)
check(
    abs(len(delayeds) - expected_delayeds) < 5 * np.sqrt(expected_delayeds) + 1,
    f"number of delayeds {len(delayeds)} matches the expectation {expected_delayeds:.0f}",
)

# %%
# delayeds vs the randoms model of the simulator: the expected randoms of a pair of
# detection bins (a, b) in coincidence are singles[a] * singles[b], uniform over the TOF
# bins. Coincidences are defined on the module pair level (SGID >= 0), such that the
# expected delayeds of module pairs only depend on the module totals of the singles.

num_tofbins = scanner.tof_bin_edges[0][0].number_of_bins()
tof_bin_edges_all = np.asarray(scanner.tof_bin_edges[0][0].edges, dtype=np.float64)
module_coinc = np.zeros((num_modules, num_modules))
for i, row in enumerate(sgid_lut):
    for j, sg in enumerate(row):
        if sg >= 0:
            module_coinc[i, j] = module_coinc[j, i] = 1
true_singles = np.asarray(randoms_info["singles"])
check(true_singles.shape == (num_det_bins,), "randoms expectation has one singles value per detection bin")


def module_pair_randoms(singles: np.ndarray) -> np.ndarray:
    """expected randoms of all module pairs (lower triangle, i > j) from the singles"""
    totals = singles.reshape(num_modules, -1).sum(1)
    return np.tril(module_coinc * np.outer(totals, totals), -1)


# expected randoms of every prompt (in file order) written by the simulator
prompts_randoms = np.load(out_dir / "prompts_randoms_expectation.npy")
check(
    prompts_randoms.shape == (num_events,)
    and np.allclose(
        prompts_randoms, true_singles[det_bin_0] * true_singles[det_bin_1] / num_tofbins, rtol=1e-5
    ),
    "expected randoms of all prompts (prompts_randoms_expectation.npy) match the singles "
    "of their detection bins",
)
if random_fraction > 0:
    check(
        bool(np.all(prompts_randoms > 0)),
        f"all prompts have positive expected randoms (mean {prompts_randoms.mean():.2e})",
    )

if delayeds:
    delayed_arr = np.asarray(delayeds)
    delayed_mp = np.zeros((num_modules, num_modules))
    np.add.at(
        delayed_mp,
        (delayed_arr[:, 0] // num_bins_per_module_type, delayed_arr[:, 1] // num_bins_per_module_type),
        1,
    )
    exp_mp = module_pair_randoms(true_singles)
    in_coinc = exp_mp > 0
    z = (delayed_mp[in_coinc] - exp_mp[in_coinc]) / np.sqrt(exp_mp[in_coinc])
    check(
        np.all(delayed_mp[~in_coinc] == 0) and np.abs(z).max() < 5,
        f"delayeds of all module pairs match the randoms expectation (max |z| {np.abs(z).max():.1f})",
    )
    tof_counts = np.bincount(delayed_arr[:, 2], minlength=num_tofbins)
    exp_tof = len(delayeds) / num_tofbins
    z_tof = np.abs(tof_counts - exp_tof).max() / np.sqrt(exp_tof)
    check(z_tof < 5, f"delayeds are uniformly distributed over the TOF bins (max |z| {z_tof:.1f})")

# singles histograms vs the singles rates of the simulator, and randoms from singles
# 2 tau sum_k C_a,k C_b,k / dt_k (2 tau: range of the TOF bins in time) vs the true randoms
if singles_counts:
    singles_counts = np.asarray(singles_counts, dtype=np.float64)
    durations_ns = np.asarray(singles_durations_ms, dtype=np.float64) * 1e6
    rates = np.asarray(randoms_info["singles_rates_per_ns"])
    exp_counts = rates * durations_ns.sum()
    alive = exp_counts > 0
    z = (singles_counts.sum(0)[alive] - exp_counts[alive]) / np.sqrt(exp_counts[alive])
    check(
        singles_counts.shape == (num_time_blocks := len(durations_ns), num_det_bins)
        and np.all(singles_counts.sum(0)[~alive] == 0)
        and np.abs(z).max() < 6,
        f"singles counts of all detection bins in {num_time_blocks} time blocks match the "
        f"singles rates (max |z| {np.abs(z).max():.1f}, {singles_counts.sum():.3g} singles)",
    )
    window_ns = 2 * float(tof_bin_edges_all[-1] - tof_bin_edges_all[0]) / 299.792458
    check(
        np.isclose(window_ns, randoms_info["coincidence_window_ns"], rtol=1e-5),
        f"coincidence window ({window_ns:.4f} ns) = range of the TOF bins",
    )
    rfs_mp = window_ns * sum(
        module_pair_randoms(c) / t for c, t in zip(singles_counts, durations_ns)
    )
    exp_mp = module_pair_randoms(true_singles)
    in_coinc = exp_mp > 0
    rel = rfs_mp[in_coinc] / exp_mp[in_coinc] - 1
    check(
        np.abs(rel).max() < 0.02,
        f"module pair randoms from singles match the expectation (rel. diff "
        f"[{rel.min():+.4f}, {rel.max():+.4f}])",
    )
    rfs_prompts = window_ns * sum(
        c[det_bin_0] * c[det_bin_1] / t for c, t in zip(singles_counts, durations_ns)
    ) / num_tofbins
    rel = rfs_prompts / prompts_randoms - 1
    check(
        np.sqrt(np.mean(rel**2)) < 0.05,
        f"randoms from singles of all prompts match prompts_randoms_expectation.npy "
        f"(RMS rel. diff {np.sqrt(np.mean(rel**2)):.4f}, mean {rel.mean():+.4f})",
    )
elif random_fraction > 0:
    check(False, "singles histograms exist")

# %%
# TOF LM back projection of the events in the PETSIRD file vs simulator

adp = np.array(
    [
        get_ADP(scanner, 0, expand_detection_bin(scanner, 0, b)).c
        for b in range(num_det_bins)
    ],
    dtype="float32",
)

# image grid of the simulation (from the NIfTI affine of the ground truth image)
gt_img, gt_affine = load_image(out_dir / "ground_truth_image.nii")
sim_grid = grid_from_affine(gt_img.shape, gt_affine)
img_shape, voxel_size = sim_grid.shape, sim_grid.voxel_size
img_origin = np.asarray(sim_grid.origin, dtype="float32")
grid_center = np.asarray(sim_grid.origin) + 0.5 * (np.asarray(img_shape) - 1) * np.asarray(voxel_size)
check(
    sim_grid.shape == tuple(sim_params["img_shape"])
    and np.allclose(sim_grid.voxel_size, sim_params["voxel_size"])
    and np.allclose(grid_center, 0, atol=1e-3),
    f"ground truth image grid ({sim_grid}) = simulation parameters, centered at (0, 0, 0)",
)
sim_images = ["attenuation_image.nii", "reference_sensitivity_image.nii"] + [
    f.name for f in out_dir.glob("lm_backprojection_tof.nii")
] + [f.name for f in out_dir.glob("reference_histogram_mlem_*_epochs.nii")]
def image_grid(path):
    data, affine = load_image(path)
    return grid_from_affine(data.shape, affine)


grids_ok = all(image_grid(out_dir / f).matches(sim_grid) for f in sim_images)
check(grids_ok, f"all simulator images have the same grid ({', '.join(sim_images)})")

tof_bin_edges = scanner.tof_bin_edges[0][0].edges
lm_proj = ListmodePETProjector(
    adp[det_bin_0], adp[det_bin_1], img_shape, voxel_size, img_origin=img_origin
)
lm_proj.tof_parameters = TOFParameters(
    num_tofbins=tof_bin_edges.size - 1,
    tofbin_width=float(tof_bin_edges[1] - tof_bin_edges[0]),
    sigma_tof=float(scanner.tof_resolution[0][0]) / FWHM_PER_SIGMA,
)
lm_proj.event_tofbins = tof_idx.astype("int16")
lm_proj.tof = True
lm_back_tof = lm_proj.adjoint(np.ones(num_events, dtype="float32"))

sim_lm_back_tof_file = out_dir / "lm_backprojection_tof.nii"
if sim_lm_back_tof_file.exists():
    d = rel_diff(load_image(sim_lm_back_tof_file)[0], lm_back_tof)
    check(d < 1e-4, f"TOF LM back projection of PETSIRD events matches simulator ({d:.1e})")
else:
    check(False, "simulator TOF LM back projection exists (run with --save_lm_backprojection)")

# %%
# TOF sign / scale following the PETSIRD specification (independent of parallelproj)
#
# PETSIRD stores (t1 - t2) * c / 2 where t1 / t2 are the arrival times at the first /
# second detection bin. A positive value means that the photon arrived first at the
# second detection bin, i.e. that the emission happened closer to the second detection
# bin. We decode the emission position along each LOR (relative to the LOR center,
# positive towards the second detection bin) from the center of its TOF bin and compare
# it to the position expected from the ground truth activity along the LOR.
# (Attenuation and detection efficiencies are constant along a LOR and do not change
# the expected position.) For the correct convention, the slope of the regression of
# decoded vs expected positions is close to +1, for a flipped one close to -1.

rng = np.random.default_rng(0)
idx = rng.choice(num_events, min(20000, num_events), replace=False)

tof_bin_centers = 0.5 * (tof_bin_edges[1:] + tof_bin_edges[:-1]).astype("float64")
decoded_pos = tof_bin_centers[tof_idx[idx]]

p1 = adp[det_bin_0[idx]].astype("float64")
p2 = adp[det_bin_1[idx]].astype("float64")
lor_length = np.linalg.norm(p2 - p1, axis=1)
lor_dir = (p2 - p1) / lor_length[:, None]
lor_center = 0.5 * (p1 + p2)

# sample the ground truth image (nearest voxel) at 401 points along every LOR
t = np.linspace(-0.5, 0.5, 401)[None, :] * lor_length[:, None]
sample_pos = lor_center[:, None, :] + t[..., None] * lor_dir[:, None, :]
vox = np.rint((sample_pos - img_origin) / np.asarray(voxel_size)).astype(int)
inside = np.all((vox >= 0) & (vox < np.asarray(img_shape)), axis=-1)
vox = np.clip(vox, 0, np.asarray(img_shape) - 1)
act = np.where(inside, gt_img[vox[..., 0], vox[..., 1], vox[..., 2]], 0)

has_act = act.sum(1) > 0
expected_pos = (act * t).sum(1)[has_act] / act.sum(1)[has_act]
slope = np.polyfit(expected_pos, decoded_pos[has_act], 1)[0]
# (randoms have uniformly distributed TOF bins, which reduces the slope by ~ (1 - f))
check(
    0.5 * (1 - random_fraction) < slope < 1.5,
    f"TOF bins follow the PETSIRD convention: slope of decoded vs expected emission "
    f"positions {slope:+.2f} (flipped convention: -1)",
)

# %%
# vectorized efficiencies vs petsird.helpers.get_detection_efficiency

det_bin_effs = np.asarray(det_effs.detection_bin_efficiencies[0])
mp_effs = {
    x.sgid: np.asarray(x.values) for x in det_effs.module_pair_efficiencies_vectors[0][0]
}
num_bins_per_module = num_det_els * num_energy_bins
sgid = np.array(
    [
        sgid_lut[i][j]
        for i, j in zip(
            det_bin_0 // num_bins_per_module, det_bin_1 // num_bins_per_module
        )
    ]
)
check(bool(np.all(sgid >= 0)), "all events are in module pairs in coincidence")

idx = rng.choice(num_events, min(2000, num_events), replace=False)
effs = (
    det_effs.calibration_factor
    * det_bin_effs[det_bin_0[idx]]
    * det_bin_effs[det_bin_1[idx]]
    * np.array(
        [
            mp_effs[s][b0 % num_bins_per_module, b1 % num_bins_per_module]
            for s, b0, b1 in zip(sgid[idx], det_bin_0[idx], det_bin_1[idx])
        ]
    )
)
ref_effs = np.array(
    [
        get_detection_efficiency(scanner, type_pair, int(det_bin_0[i]), int(det_bin_1[i]))
        for i in idx
    ]
)
check(
    np.allclose(effs, ref_effs, rtol=1e-6) and bool(np.all(ref_effs > 0)),
    "efficiencies match petsird.helpers.get_detection_efficiency",
)

# %%
# reconstruction outputs (if present)

# only consider recon outputs of the current listmode file (tagged with its fingerprint)
stem = lm_file.stem
fingerprint = lm_file_fingerprint(lm_file)
def on_sim_grid(files):
    """images of files on the simulation grid (other grids, e.g. recons with
    --img_shape / --voxel_size, are skipped)"""
    images = {}
    for f in files:
        data, affine = load_image(f)
        if grid_from_affine(data.shape, affine).matches(sim_grid):
            images[f.name] = data
        else:
            print(f"       {f.name}: different image grid than the simulation - skipped")
    return images


recon_sens = on_sim_grid(sorted(out_dir.glob(f"{stem}_sens_{fingerprint}_*.nii")))
recon_tof_back = on_sim_grid(sorted(out_dir.glob(f"{stem}_tof_backproj_{fingerprint}_*.nii")))

if recon_sens and recon_tof_back:
    # there can be multiple recon outputs (e.g. TOF / non-TOF, with / without attenuation)
    # - the (unweighted) TOF LM back projections all have to match
    # - the sensitivity image of the TOF recon with attenuation correction (using the
    #   simulated attenuation image) has to match the simulator's. The simulator uses a
    #   TOF sinogram projector, the reconstruction a TOF LM projector, which differ
    #   slightly in the normalization of the TOF kernels.
    ref_sens = load_image(out_dir / "reference_sensitivity_image.nii")[0]
    diffs = {name: rel_diff(ref_sens, img) for name, img in recon_sens.items()}
    for name, d in diffs.items():
        print(f"       {name}: rel. diff to simulator sensitivity image {d:.1e}")
    best = min(diffs, key=diffs.get)
    check(
        diffs[best] < 2e-3,
        f"sensitivity image of a recon matches the simulator's ({best}, {diffs[best]:.1e}) "
        "(requires a TOF recon with --attenuation_image attenuation_image.nii)",
    )
    for name, img in recon_tof_back.items():
        d = rel_diff(lm_back_tof, img)
        check(d < 1e-4, f"recon TOF LM back projection {name} matches PETSIRD events ({d:.1e})")
else:
    print("no reconstruction outputs found - skipping reconstruction checks")

# randoms estimated from the delayeds by a recon (--randoms_from_delayeds): the ML
# estimate reproduces the number of delayeds and the module pair randoms agree with
# the expectation within the Poisson noise of the delayeds of a module pair
for f in sorted(out_dir.glob(f"{stem}_randoms_singles_*_{fingerprint}.npy")):
    est_singles = np.load(f)
    est_mp, exp_mp = module_pair_randoms(est_singles), module_pair_randoms(true_singles)
    in_coinc = exp_mp > 0
    z = (est_mp[in_coinc] - exp_mp[in_coinc]) / np.sqrt(exp_mp[in_coinc])
    rel = est_mp[in_coinc] / exp_mp[in_coinc] - 1
    msg = (
        f"{f.name}: total {est_mp.sum():.0f} ({len(delayeds)} delayeds, expectation "
        f"{exp_mp.sum():.0f}), module pair randoms vs expectation: rel. diff "
        f"[{rel.min():+.3f}, {rel.max():+.3f}], max |z| {np.abs(z).max():.1f}"
    )
    if "_ml" in f.name:
        check(
            abs(est_mp.sum() / len(delayeds) - 1) < 1e-4 and np.abs(z).max() < 5,
            f"ML randoms estimate {msg}",
        )
    else:
        print(f"       {msg}")

# %%
if failed:
    raise SystemExit(f"{len(failed)} check(s) failed")
print("all checks passed")
