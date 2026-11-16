"""statistical study of the randoms estimation from delayed coincidences

For the simulated scanner and phantom (same models as the simulator), we draw many
Poisson realizations of the delayeds (histogrammed over all LORs, no PETSIRD I/O) at
several count levels and compare the Casey-type and the ML estimates of the randoms
(randoms.py) with the true randoms expectation:

- convergence of the ML iterations (log-likelihood, difference to the converged ML
  estimate) when initialized with the Casey-type estimate
- bias and variance of the estimated randoms per LOR (Casey, ML) vs. the raw delayeds
- detection bins without delayeds

Usage:
    python tests/study_randoms_estimation.py OUTPUT_DIR [--num_realizations 50]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from efficiencies import generate_efficiency_model, lor_efficiencies  # noqa: E402
from phantoms import create_phantom  # noqa: E402
from randoms import (  # noqa: E402
    casey_singles,
    detection_bin_fans,
    ml_singles,
    randoms_log_likelihood,
)
from scanner import ScannerConfig, create_parallelproj_lor_descriptor  # noqa: E402
from simulation import (  # noqa: E402
    attenuation_factors,
    create_forward_model,
    create_petsird_header,
    create_tof_projector,
    lor_fan_sums,
    lor_products,
    randoms_expectation,
    singles_model,
    tof_parameters,
)
from parallelproj.operators import GaussianFilterOperator  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=Path)
parser.add_argument("--num_realizations", type=int, default=50)
parser.add_argument("--phantom", default="offcenter_cylinder")
parser.add_argument("--uniform_fraction", type=float, default=0.5)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

# %%
# true randoms expectation (same as 01_analytic_petsird_lm_simulator.py, seed 0)
np.random.seed(0)
img_shape, voxel_size = (55, 55, 19), (2.0, 2.0, 2.0)
config = ScannerConfig()
lor_desc = create_parallelproj_lor_descriptor(config)
img, att_img = create_phantom(args.phantom, img_shape, voxel_size)
tof_params = tof_parameters(10.0, img_shape, voxel_size)
proj = create_tof_projector(lor_desc, img_shape, voxel_size, tof_params)
eff_model = generate_efficiency_model(config)
res_model = GaussianFilterOperator(img_shape, sigma=2.5 / (2.35482 * np.asarray(voxel_size)))
fwd_op = create_forward_model(
    proj,
    res_model,
    lor_efficiencies(eff_model, lor_desc) * attenuation_factors(lor_desc, img_shape, voxel_size, att_img),
)
trues_lor = fwd_op(img).sum(-1)
singles = singles_model(
    lor_fan_sums(trues_lor, lor_desc, config.num_modules), eff_model.det_el_effs, args.uniform_fraction
)
_, true_singles = randoms_expectation(singles, lor_desc, 1, 1.0)  # unit total (rescaled below)
true_singles = true_singles.ravel()

header = create_petsird_header(config, eff_model, tof_params, 10.0)
fans = detection_bin_fans(header.scanner)

num_el = config.num_det_els_per_module
start_mod = np.asarray(lor_desc.all_block_pairs)[:, :1]
end_mod = np.asarray(lor_desc.all_block_pairs)[:, 1:]
el = np.arange(num_el**2)
bin_a = (start_mod * num_el + el // num_el).ravel()  # detection bins of all LORs
bin_b = (end_mod * num_el + el % num_el).ravel()
unit_randoms = lor_products(true_singles.reshape(config.num_modules, num_el), lor_desc).ravel()
alive_lor = unit_randoms > 0
dead_bins = int(np.count_nonzero(true_singles == 0))

print(f"{unit_randoms.size} LORs, {fans.num_bins} detection bins ({dead_bins} dead)")
print(
    "true singles (alive): min/max ratio "
    f"{true_singles[true_singles > 0].min() / true_singles.max():.3f}"
)
results = {
    "num_lors": int(unit_randoms.size),
    "num_detection_bins": fans.num_bins,
    "num_dead_bins": dead_bins,
    "phantom": args.phantom,
    "uniform_fraction": args.uniform_fraction,
    "levels": {},
}


def lor_randoms(s):
    return (s[bin_a] * s[bin_b])[alive_lor]


def fan_sums_of(d):
    return np.bincount(bin_a, weights=d, minlength=fans.num_bins) + np.bincount(
        bin_b, weights=d, minlength=fans.num_bins
    )


iterations = [0, 1, 2, 3, 5, 10, 20, 50]
rng = np.random.default_rng(1)
# expected delayeds: CI (200k trues, f=0.1), default simulation (4M trues, f=0.1), high
for total in [22_222.0, 444_444.0, 4_444_444.0]:
    true_r = total * unit_randoms[alive_lor]
    s_true = np.sqrt(total) * true_singles
    est = {"casey": [], "ml10": [], "ml_conv": []}
    conv = {k: [] for k in iterations}
    loglik = {k: [] for k in iterations}
    raw_var = []
    zero_alive = []
    t_ml = []
    for r in range(args.num_realizations):
        d = rng.poisson(total * unit_randoms).astype(np.float64)
        F = fan_sums_of(d)
        zero_alive.append(int(np.count_nonzero((F == 0) & (true_singles > 0))))
        s_ref = ml_singles(fans, F, num_iterations=2000)
        r_ref = lor_randoms(s_ref)
        s = casey_singles(fans, F)
        for k in range(max(iterations) + 1):
            if k in iterations:
                conv[k].append(float(np.sqrt(np.mean((lor_randoms(s) - r_ref) ** 2)) / true_r.mean()))
                loglik[k].append(randoms_log_likelihood(fans, F, s))
            s = ml_singles(fans, F, num_iterations=1, init=s)
        t0 = time.perf_counter()
        s10 = ml_singles(fans, F, num_iterations=10)
        t_ml.append(time.perf_counter() - t0)
        est["casey"].append(lor_randoms(casey_singles(fans, F)))
        est["ml10"].append(lor_randoms(s10))
        est["ml_conv"].append(r_ref)
        raw_var.append(d[alive_lor])
        loglik_ref = randoms_log_likelihood(fans, F, s_ref)
        for k in iterations:
            loglik[k][-1] = loglik_ref - loglik[k][-1]

    level = {
        "expected_delayeds": total,
        "mean_delayeds_per_lor": float(true_r.mean()),
        "alive_bins_without_delayeds_mean": float(np.mean(zero_alive)),
        "ml10_time_s": float(np.median(t_ml)),
        "ml_convergence_rms_diff_to_converged_rel_to_mean_randoms": {
            k: float(np.mean(v)) for k, v in conv.items()
        },
        "ml_loglik_gap_to_converged": {k: float(np.mean(v)) for k, v in loglik.items()},
        "estimators": {},
    }
    for name, values in list(est.items()) + [("raw_delayeds", raw_var)]:
        v = np.array(values)
        mean = v.mean(0)
        rel_bias = mean / true_r - 1
        rel_std = v.std(0) / true_r
        # bias aggregated per module pair (noise of the mean is small after averaging)
        lor_bp = np.repeat(np.arange(lor_desc.num_block_pairs), num_el**2)[alive_lor]
        bp_bias = np.bincount(lor_bp, weights=mean) / np.bincount(lor_bp, weights=true_r) - 1
        level["estimators"][name] = {
            "total_rel_bias": float(mean.sum() / true_r.sum() - 1),
            "module_pair_rel_bias_min": float(bp_bias.min()),
            "module_pair_rel_bias_max": float(bp_bias.max()),
            "lor_rel_bias_rms": float(np.sqrt(np.mean(rel_bias**2))),
            "lor_rel_bias_rms_noise_floor": float(
                np.sqrt(np.mean(rel_std**2) / args.num_realizations)
            ),
            "lor_rel_std_mean": float(rel_std.mean()),
            "lor_rel_rmse": float(np.sqrt(np.mean(((v - true_r) / true_r) ** 2))),
        }
    results["levels"][f"{total:.0f}"] = level
    print(json.dumps(level, indent=2))

(args.output_dir / "randoms_estimation_study.json").write_text(json.dumps(results, indent=2))
