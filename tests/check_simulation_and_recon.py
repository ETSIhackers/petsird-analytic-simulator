"""consistency checks of a simulated PETSIRD file and (optionally) its reconstruction

Usage:
    python 01_analytic_petsird_lm_simulator.py --output_dir OUT --check_backprojection ...
    python 02_reconstruct_petsird.py OUT/simulated_petsird_lm_file.bin ...   (optional)
    python tests/check_simulation_and_recon.py OUT

The checks of the PETSIRD file only use petsird (and its helpers) and parallelproj,
but not the readers in python/utils.py, such that they are independent of the
reconstruction code:

- the detecting element box satisfies the PETSIRD constraints
- all tables over module (type) pairs are lower-triangular
- all coincidences are ordered (detection_bins[0] >= detection_bins[1])
- a TOF LM back projection of the events in the PETSIRD file matches the one
  calculated by the simulator before writing (checks detection bin swapping and
  TOF bin mirroring)
- vectorized efficiencies match petsird.helpers.get_detection_efficiency

If the reconstruction outputs exist, the sensitivity image and the LM back projections
calculated from the PETSIRD file are compared to the ones of the simulator.

The script exits with a non-zero exit code if any check fails.
"""

import argparse
import contextlib
import io
import json
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
check(
    scanner.prompt_event_policy == petsird.CoincidencePolicy.REJECT_HIGHER_MULTIPLES
    and scanner.delayed_event_policy == petsird.CoincidencePolicy.NONE,
    "event policies",
)

# %%
# read all events

det_bin_0, det_bin_1, tof_idx = [], [], []
for time_block in reader.read_time_blocks():
    if isinstance(time_block, petsird.TimeBlock.EventTimeBlock):
        for event in time_block.value.prompt_events[0][0]:
            det_bin_0.append(event.detection_bins[0])
            det_bin_1.append(event.detection_bins[1])
            tof_idx.append(event.tof_idx)
det_bin_0, det_bin_1, tof_idx = map(np.asarray, (det_bin_0, det_bin_1, tof_idx))
num_events = det_bin_0.size

type_pair = petsird.TypeOfModulePair((0, 0))
check(
    num_events > 0
    and all(
        are_detections_ordered(type_pair, int(b0), int(b1))
        for b0, b1 in zip(det_bin_0, det_bin_1)
    ),
    f"all {num_events} coincidences are ordered",
)

# %%
# TOF LM back projection of the events in the PETSIRD file vs simulator

adp = np.array(
    [
        get_ADP(scanner, 0, expand_detection_bin(scanner, 0, b)).c
        for b in range(num_det_bins)
    ],
    dtype="float32",
)

img_shape = tuple(sim_params["img_shape"])
voxel_size = tuple(sim_params["voxel_size"])
# the simulator uses an image centered at (0,0,0)
img_origin = (-(np.asarray(img_shape) - 1) / 2 * np.asarray(voxel_size)).astype(
    "float32"
)

tof_bin_edges = scanner.tof_bin_edges[0][0].edges
lm_proj = ListmodePETProjector(
    adp[det_bin_0], adp[det_bin_1], img_shape, voxel_size, img_origin=img_origin
)
lm_proj.tof_parameters = TOFParameters(
    num_tofbins=tof_bin_edges.size - 1,
    tofbin_width=float(tof_bin_edges[1] - tof_bin_edges[0]),
    sigma_tof=float(scanner.tof_resolution[0][0]) / 2.35,
)
lm_proj.event_tofbins = tof_idx.astype("int16")
lm_proj.tof = True
lm_back_tof = lm_proj.adjoint(np.ones(num_events, dtype="float32"))

sim_lm_back_tof_file = out_dir / "lm_backprojection_tof.npy"
if sim_lm_back_tof_file.exists():
    d = rel_diff(np.load(sim_lm_back_tof_file), lm_back_tof)
    check(d < 1e-4, f"TOF LM back projection of PETSIRD events matches simulator ({d:.1e})")
else:
    check(False, "simulator TOF LM back projection exists (run with --check_backprojection)")

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

rng = np.random.default_rng(0)
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

stem = lm_file.stem
recon_sens = sorted(out_dir.glob(f"{stem}_sens_*.npy"))
recon_tof_back = sorted(out_dir.glob(f"{stem}_tof_backproj_*.npy"))

if recon_sens and recon_tof_back:
    # there can be multiple recon outputs (e.g. TOF / non-TOF, with / without attenuation)
    # - the (unweighted) TOF LM back projections all have to match
    # - at least one sensitivity image (TOF without attenuation) has to match the
    #   simulator's. The simulator's sensitivity image uses a TOF sinogram projector,
    #   the reconstruction a TOF LM projector, which differ slightly in the normalization
    #   of the TOF kernels.
    ref_sens = np.load(out_dir / "reference_sensitivity_image.npy")
    sens_diffs = {f.name: rel_diff(ref_sens, np.load(f)) for f in recon_sens}
    for name, d in sens_diffs.items():
        print(f"       {name}: rel. diff to simulator sensitivity image {d:.1e}")
    d = min(sens_diffs.values())
    check(d < 2e-3, f"a recon sensitivity image matches the simulator's ({d:.1e})")
    for f in recon_tof_back:
        d = rel_diff(lm_back_tof, np.load(f))
        check(d < 1e-4, f"recon TOF LM back projection {f.name} matches PETSIRD events ({d:.1e})")
else:
    print("no reconstruction outputs found - skipping reconstruction checks")

# %%
if failed:
    raise SystemExit(f"{len(failed)} check(s) failed")
print("all checks passed")
