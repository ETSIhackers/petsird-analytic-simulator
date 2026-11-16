import numpy as np
import pytest

from randoms import (
    casey_singles,
    delayeds_fan_sums,
    detection_bin_fans,
    ml_singles,
    randoms_log_likelihood,
    randoms_of_events,
)
from scanner import create_parallelproj_lor_descriptor
from simulation import lor_fan_sums, lor_products


@pytest.fixture
def tiny_fans(tiny_header):
    return detection_bin_fans(tiny_header.scanner)


def lor_detection_bins(config):
    """detection bins (a, b) of all LORs of the parallelproj LOR descriptor"""
    lor_desc = create_parallelproj_lor_descriptor(config)
    n = config.num_det_els_per_module
    pairs = np.asarray(lor_desc.all_block_pairs)
    el = np.arange(n * n)
    a = (pairs[:, :1] * n + el // n).ravel()
    b = (pairs[:, 1:] * n + el % n).ravel()
    return lor_desc, a, b


def test_fans_match_simulated_lors(tiny_config, tiny_fans):
    """fans from the PETSIRD header == pairs of detection bins of the simulator's LORs"""
    lor_desc, a, b = lor_detection_bins(tiny_config)
    coinc = np.zeros((tiny_fans.num_bins, tiny_fans.num_bins))
    coinc[a, b] = coinc[b, a] = 1

    v = np.random.default_rng(0).uniform(0.5, 2, tiny_fans.num_bins)
    np.testing.assert_allclose(tiny_fans.fan_sums(v), coinc @ v)
    np.testing.assert_allclose(tiny_fans.pair_sum(v), (v[a] * v[b]).sum())

    # simulator (LOR descriptor) and PETSIRD fan sums of products of singles agree
    v2 = v.reshape(tiny_config.num_modules, -1)
    np.testing.assert_allclose(
        lor_fan_sums(lor_products(v2, lor_desc), lor_desc, tiny_config.num_modules).ravel(),
        v * tiny_fans.fan_sums(v),
    )


def test_delayeds_fan_sums(tiny_fans):
    F = delayeds_fan_sums(tiny_fans, np.array([5, 7, 7]), np.array([1, 2, 1]))
    assert F.shape == (tiny_fans.num_bins,)
    assert F[1] == 2 and F[7] == 2 and F[5] == 1 and F[2] == 1 and F.sum() == 6


def expected_fan_sums(fans, singles):
    return singles * fans.fan_sums(singles)


def test_casey_exact_for_equal_fan_sums(tiny_fans):
    """Casey-type estimate is exact (noise-free data) if all fan sums of the singles are equal"""
    singles = np.full(tiny_fans.num_bins, 0.3)
    np.testing.assert_allclose(casey_singles(tiny_fans, expected_fan_sums(tiny_fans, singles)), singles)


def test_ml_recovers_singles_from_noise_free_data(tiny_config, tiny_fans):
    """for noise-free delayeds, ML recovers all randoms s_a s_b, Casey is biased"""
    rng = np.random.default_rng(1)
    singles = rng.uniform(0.2, 2.0, tiny_fans.num_bins)
    singles[3] = 0  # dead detection bin
    singles[: tiny_config.num_det_els_per_module] *= 3  # "hot" module
    F = expected_fan_sums(tiny_fans, singles)
    _, a, b = lor_detection_bins(tiny_config)
    true_randoms = singles[a] * singles[b]

    s_ml = ml_singles(tiny_fans, F, num_iterations=100)
    np.testing.assert_allclose(s_ml[a] * s_ml[b], true_randoms, rtol=1e-6, atol=1e-12)
    assert s_ml[3] == 0

    s_casey = casey_singles(tiny_fans, F)
    assert np.abs(s_casey[a] * s_casey[b] / np.where(true_randoms > 0, true_randoms, 1) - 1).max() > 0.01


def test_ml_monotone_and_total(tiny_config, tiny_fans):
    """ML iterations increase the log-likelihood, the estimated randoms total equals the
    number of delayeds, detection bins without delayeds get zero singles"""
    rng = np.random.default_rng(2)
    singles = rng.uniform(0.05, 0.3, tiny_fans.num_bins)
    _, a, b = lor_detection_bins(tiny_config)
    d = rng.poisson(singles[a] * singles[b])
    F = delayeds_fan_sums(tiny_fans, np.repeat(a, d), np.repeat(b, d))
    assert np.any(F == 0)

    history = []
    s = ml_singles(tiny_fans, F, num_iterations=30, history=history)
    loglik = [randoms_log_likelihood(tiny_fans, F, casey_singles(tiny_fans, F))] + [
        h[0] for h in history
    ]
    assert np.all(np.diff(loglik) >= -1e-9 * abs(loglik[0]))
    np.testing.assert_allclose(tiny_fans.pair_sum(s), d.sum(), rtol=1e-6)
    assert np.all(s[F == 0] == 0) and np.all(s[F > 0] > 0)

    # ML fixed-point equations s_a = F_a / G_a hold at convergence
    pos = F > 0
    np.testing.assert_allclose(s[pos] * tiny_fans.fan_sums(s)[pos], F[pos], rtol=1e-4)


def test_randoms_of_events_units():
    """randoms per event refer to a TOF bin (TOF) or the whole LOR (non-TOF)"""
    singles = np.array([0.0, 0.5, 2.0])
    tof = randoms_of_events(singles, np.array([2, 2]), np.array([1, 0]), num_tofbins=4, tof=True)
    non_tof = randoms_of_events(singles, np.array([2, 2]), np.array([1, 0]), num_tofbins=4, tof=False)
    np.testing.assert_allclose(non_tof, [1.0, 0.0])
    np.testing.assert_allclose(tof, non_tof / 4)


def test_coincidence_window_from_tof_bin_edges():
    """TOF bin edges are (t1 - t2) c / 2 in mm: a range of L mm is 2 L / c in time"""
    from randoms import SPEED_OF_LIGHT_MM_PER_NS, coincidence_window_ns

    edges = np.linspace(-150, 150, 11)
    np.testing.assert_allclose(coincidence_window_ns(edges), 600 / SPEED_OF_LIGHT_MM_PER_NS)


def test_randoms_from_singles_of_events():
    """2 tau sum_k C_a,k C_b,k / dt_k, for noise-free counts of constant rates equal to
    the scaled-singles model s_a s_b (s = sqrt(2 tau T) S)"""
    from randoms import randoms_from_singles_of_events
    from simulation import singles_rates

    window, durations = 1.5, np.array([2e8, 1e8, 3e8])  # ns
    scaled_singles = np.array([0.0, 0.4, 1.2, 2.0])
    rates = singles_rates(scaled_singles, window, durations.sum())
    counts = rates[None, :] * durations[:, None]
    d0, d1 = np.array([3, 2, 3]), np.array([1, 0, 2])
    non_tof = randoms_from_singles_of_events(counts, durations, window, d0, d1, 5, tof=False)
    np.testing.assert_allclose(non_tof, scaled_singles[d0] * scaled_singles[d1], rtol=1e-6)
    tof = randoms_from_singles_of_events(counts, durations, window, d0, d1, 5, tof=True)
    np.testing.assert_allclose(tof, non_tof / 5, rtol=1e-6)


def test_randoms_from_singles_total(tiny_fans):
    from randoms import randoms_from_singles_total

    counts = np.random.default_rng(3).poisson(50.0, (2, tiny_fans.num_bins)).astype(float)
    durations = np.array([1e8, 2e8])
    ref = sum(tiny_fans.pair_sum(c) / t for c, t in zip(counts, durations)) * 0.7
    np.testing.assert_allclose(randoms_from_singles_total(tiny_fans, counts, durations, 0.7), ref)


def test_static_singles_equal_time_blocks_for_constant_rates():
    """for constant singles rates (noise-free counts), the static estimate (summed singles)
    equals the per-time-block estimate"""
    from randoms import randoms_from_singles_of_events, randoms_of_events, static_singles_from_singles_histograms

    rates = np.array([0.0, 1e-4, 3e-4, 2e-4])  # per ns
    durations = np.array([1e8, 2e8, 1e8])
    counts = rates[None, :] * durations[:, None]
    d0, d1 = np.array([3, 2, 3]), np.array([1, 0, 2])
    s = static_singles_from_singles_histograms(counts, durations, 1.02)
    np.testing.assert_allclose(
        randoms_of_events(s, d0, d1, 5, tof=True),
        randoms_from_singles_of_events(counts, durations, 1.02, d0, d1, 5, tof=True),
        rtol=1e-6,
    )
    # = 2 tau T S_a S_b
    np.testing.assert_allclose(s[3] * s[1], 1.02 * durations.sum() * rates[3] * rates[1])
