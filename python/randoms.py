"""randoms model and estimation of the randoms from delayed coincidences

(theory: docs/randoms_estimation.md)

Model
-----
The expected number of random coincidences of a pair of detection bins (a, b) in
coincidence (summed over all TOF bins) is the product of their (scaled) singles rates

    E[R_ab] = s_a * s_b

(the scale of the singles includes the coincidence window), and the randoms are
uniformly distributed over the TOF bins. The delayed coincidences D_ab are an
independent Poisson realization of the randoms, E[D_ab] = s_a * s_b.

All functions use a global detection bin index over all module types
(see petsird_reader.detection_bin_offsets).

Fans
----
The fan of a detection bin a is the set of all detection bins b that are in coincidence
with a. In PETSIRD, coincidences are defined on the level of module pairs (SGID >= 0),
such that the fan of a contains all detection bins of all modules in coincidence with
the module of a. Fan sums sum_{b in fan(a)} v_b can therefore be computed from the
module totals of v (see DetectionBinFans.fan_sums) in
O(num_detection_bins + num_modules^2) operations.

Estimators of the singles s from the fan sums of the delayeds F_a = sum_{b in fan(a)} D_ab
-------------------------------------------------------------------------------------------
- casey_singles: closed form (Casey-type fan-sum estimate) s_a = F_a / sqrt(sum_{b in fan(a)} F_b),
  which is exact (in expectation) if all fan sums of the singles are equal
- ml_singles: maximum likelihood estimate (Poisson model of the delayeds), fixed-point
  iterations with exact scale maximization (safeguarded by MM steps, monotone),
  initialized with the Casey-type estimate
"""

from dataclasses import dataclass

import numpy as np
import petsird

from petsird_reader import detection_bin_offsets, get_module_pair_sgid_lut, module_type_info


@dataclass
class DetectionBinFans:
    """fan structure of all detection bins (global detection bin index)"""

    # module (global module index) of every detection bin, shape (num_detection_bins,)
    module_of_bin: np.ndarray
    # symmetric 0/1 matrix of all module pairs in coincidence, shape (num_modules, num_modules)
    module_coincidences: np.ndarray

    @property
    def num_bins(self) -> int:
        return self.module_of_bin.shape[0]

    def fan_sums(self, values: np.ndarray) -> np.ndarray:
        """sum_{b in fan(a)} values_b for all detection bins a

        computed from the module totals of values: modules in coincidence with the
        module of a (minus a itself if a module is in coincidence with itself).
        """
        values = np.asarray(values, dtype=np.float64)
        module_totals = np.bincount(
            self.module_of_bin, weights=values, minlength=self.module_coincidences.shape[0]
        )
        fan_totals = self.module_coincidences @ module_totals
        self_coinc = np.diag(self.module_coincidences)[self.module_of_bin]
        return fan_totals[self.module_of_bin] - self_coinc * values

    def pair_sum(self, values: np.ndarray) -> float:
        """sum over all (unordered) pairs (a, b) in coincidence of values_a * values_b"""
        values = np.asarray(values, dtype=np.float64)
        return 0.5 * float(np.sum(values * self.fan_sums(values)))


def detection_bin_fans(scanner_info: petsird.ScannerInformation) -> DetectionBinFans:
    """fan structure of all detection bins from the PETSIRD scanner information

    Two modules are in coincidence if their SGID (module_pair_sgidlut) is >= 0.
    """
    num_module_types = scanner_info.scanner_geometry.number_of_module_types()
    infos = [module_type_info(scanner_info, m) for m in range(num_module_types)]
    module_offsets = np.concatenate([[0], np.cumsum([info[0] for info in infos])])
    bin_offsets = detection_bin_offsets(scanner_info)

    module_of_bin = np.zeros(bin_offsets[-1], dtype=np.int64)
    for m, (num_modules, num_det_els, num_energy_bins) in enumerate(infos):
        module_of_bin[bin_offsets[m] : bin_offsets[m + 1]] = module_offsets[m] + np.repeat(
            np.arange(num_modules), num_det_els * num_energy_bins
        )

    coinc = np.zeros((module_offsets[-1], module_offsets[-1]), dtype=np.float64)
    for m1 in range(num_module_types):
        for m2 in range(m1 + 1):
            # dense LUT with -1 outside of the lower triangle (every module pair once)
            lut = get_module_pair_sgid_lut(scanner_info, m1, m2)
            i, j = np.nonzero(lut >= 0)
            coinc[module_offsets[m1] + i, module_offsets[m2] + j] = 1
            coinc[module_offsets[m2] + j, module_offsets[m1] + i] = 1

    return DetectionBinFans(module_of_bin=module_of_bin, module_coincidences=coinc)


def delayeds_fan_sums(
    fans: DetectionBinFans, det_bin0: np.ndarray, det_bin1: np.ndarray
) -> np.ndarray:
    """fan sums F_a = sum_{b in fan(a)} D_ab of the delayeds (counts of every detection bin)"""
    return (
        np.bincount(det_bin0, minlength=fans.num_bins)
        + np.bincount(det_bin1, minlength=fans.num_bins)
    ).astype(np.float64)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def casey_singles(fans: DetectionBinFans, delayeds_fan_sums: np.ndarray) -> np.ndarray:
    """Casey-type fan-sum estimate of the singles

    s_a = F_a / sqrt(sum_{b in fan(a)} F_b)

    With E[F_a] = s_a * G_a, where G_a = sum_{b in fan(a)} s_b, the denominator has the
    expectation sqrt(sum_{b in fan(a)} s_b G_b), such that the estimate is exact in
    expectation (up to the noise in the denominator) if all G_b are equal ("Casey
    assumption"). Detection bins without delayeds get s_a = 0.
    """
    return _safe_divide(delayeds_fan_sums, np.sqrt(fans.fan_sums(delayeds_fan_sums)))


def randoms_log_likelihood(
    fans: DetectionBinFans, delayeds_fan_sums: np.ndarray, singles: np.ndarray
) -> float:
    """Poisson log-likelihood of the delayeds (up to a constant independent of the singles)

    sum_{pairs (a,b)} D_ab log(s_a s_b) - s_a s_b = sum_a F_a log(s_a) - sum_{pairs} s_a s_b
    """
    pos = delayeds_fan_sums > 0
    if np.any(singles[pos] <= 0):
        return -np.inf
    return float(np.sum(delayeds_fan_sums[pos] * np.log(singles[pos]))) - fans.pair_sum(
        singles
    )


def _rescale_to_ml_scale(
    fans: DetectionBinFans, delayeds_fan_sums: np.ndarray, singles: np.ndarray
) -> np.ndarray:
    """scale c * s that maximizes the log-likelihood along the direction s

    L(c s) = sum_a F_a log(c s_a) - c^2 sum_{pairs} s_a s_b is maximal for
    c^2 = (sum_a F_a / 2) / sum_{pairs} s_a s_b, i.e. if the total of the estimated
    randoms equals the number of delayeds.
    """
    pair_sum = fans.pair_sum(singles)
    if pair_sum <= 0:
        return singles
    return singles * np.sqrt(0.5 * delayeds_fan_sums.sum() / pair_sum)


def ml_singles(
    fans: DetectionBinFans,
    delayeds_fan_sums: np.ndarray,
    num_iterations: int = 10,
    init: np.ndarray | None = None,
    history: list | None = None,
) -> np.ndarray:
    """maximum likelihood estimate of the singles from the fan sums of the delayeds

    The ML estimate satisfies s_a = F_a / G_a with G_a = sum_{b in fan(a)} s_b. The
    log-likelihood is concave in log(s), such that the ML solution is unique (up to a
    scaling that leaves all products s_a s_b of pairs in coincidence unchanged); the
    initialization (default: Casey-type estimate) only affects the convergence speed.

    Every iteration consists of
    1. the fixed-point (Jacobi) update s_a <- F_a / G_a
    2. a rescaling of s that exactly maximizes the log-likelihood along s (the plain
       fixed-point iteration oscillates in the overall scale: s -> c s is mapped to s / c)
    If this does not increase the log-likelihood, the step is replaced by the MM
    (minorize-maximize) update s_a <- sqrt(s_a * F_a / G_a) (followed by the rescaling),
    which follows from maximizing the separable minorizer obtained with
    s_a s_b <= (s_a^2 s_b^n / s_a^n + s_b^2 s_a^n / s_b^n) / 2 (n: current iterate).
    The log-likelihood therefore increases monotonically.
    Detection bins without delayeds get s_a = 0 (which is the ML estimate).

    If history is a list, (log-likelihood, update type) of every iteration is appended.
    """
    s = casey_singles(fans, delayeds_fan_sums) if init is None else np.asarray(init, np.float64)
    loglik = randoms_log_likelihood(fans, delayeds_fan_sums, s)
    for _ in range(num_iterations):
        fan_sums = fans.fan_sums(s)
        candidate = _rescale_to_ml_scale(
            fans, delayeds_fan_sums, _safe_divide(delayeds_fan_sums, fan_sums)
        )
        candidate_loglik = randoms_log_likelihood(fans, delayeds_fan_sums, candidate)
        update = "fixed-point"
        if candidate_loglik < loglik:
            candidate = _rescale_to_ml_scale(
                fans,
                delayeds_fan_sums,
                np.sqrt(_safe_divide(s * delayeds_fan_sums, fan_sums)),
            )
            candidate_loglik = randoms_log_likelihood(fans, delayeds_fan_sums, candidate)
            update = "MM"
        s, loglik = candidate, candidate_loglik
        if history is not None:
            history.append((loglik, update))
    return s


def randoms_of_events(
    singles: np.ndarray,
    det_bin0: np.ndarray,
    det_bin1: np.ndarray,
    num_tofbins: int,
    tof: bool,
) -> np.ndarray:
    """expected randoms of listmode events (additive contamination of a LM reconstruction)

    s_a * s_b / num_tofbins for TOF (randoms in the TOF bin of an event, uniform
    distribution over the TOF bins) and s_a * s_b for non-TOF (randoms of the LOR).
    """
    randoms = singles[det_bin0] * singles[det_bin1]
    if tof:
        randoms = randoms / num_tofbins
    return randoms.astype(np.float32)
