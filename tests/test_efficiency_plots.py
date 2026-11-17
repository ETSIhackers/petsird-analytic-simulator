import matplotlib

matplotlib.use("Agg")

import csv  # noqa: E402

import numpy as np  # noqa: E402
import petsird  # noqa: E402
from petsird.helpers import get_detection_efficiency  # noqa: E402

from efficiency_plots import (  # noqa: E402
    REFERENCE_COLUMNS,
    efficiency_reference_table,
    fan_efficiencies,
    plot_detection_bin_efficiencies,
    plot_fans,
    plot_module_pair_efficiencies,
    plot_sgid_luts,
    write_reference_table,
)


def test_reference_table_matches_petsird_helpers(tmp_path, tiny_header):
    rows = efficiency_reference_table(tiny_header.scanner, num_random_lors=200, seed=1)
    kinds = {r["lor_kind"] for r in rows}
    assert {"probe: first detection bin", "probe: last detection bin", "probe: opposite modules"} <= kinds
    # the structured efficiency model has a dead patch
    assert "probe: detection bin with efficiency 0" in kinds
    assert sum(r["lor_kind"] == "random" for r in rows) == 200
    for r in rows:
        # total (petsird.helpers) = product of the components read from the tables
        expected = r["calibration_factor"] * r["detection_bin_efficiency_0"] * r["detection_bin_efficiency_1"] * r["module_pair_efficiency"]
        np.testing.assert_allclose(r["total_efficiency"], expected, rtol=1e-6, atol=1e-12)
        assert r["sgid"] >= 0 and r["detection_bin_0"] >= r["detection_bin_1"]
    fname = tmp_path / "reference.csv"
    write_reference_table(rows, fname)
    with open(fname) as f:
        read = list(csv.DictReader(f))
    assert list(read[0]) == REFERENCE_COLUMNS and len(read) == len(rows)


def test_fan_efficiencies_match_petsird_helpers(tiny_header):
    """vectorized LOR efficiencies (our tables) == petsird.helpers.get_detection_efficiency"""
    scanner = tiny_header.scanner
    type_pair = petsird.TypeOfModulePair((0, 0))
    for a in (0, 17, 50):
        fan = fan_efficiencies(scanner, a)
        for b in range(fan.size):
            if np.isnan(fan[b]):
                continue
            ref = get_detection_efficiency(scanner, type_pair, max(a, b), min(a, b))
            np.testing.assert_allclose(fan[b], ref, rtol=1e-6, atol=1e-12)
        assert np.isnan(fan[a])


def test_plots_run(tiny_header):
    scanner = tiny_header.scanner
    for fig in (
        plot_detection_bin_efficiencies(scanner),
        plot_sgid_luts(scanner),
        plot_module_pair_efficiencies(scanner),
        plot_fans(scanner, [0, 20]),
    ):
        assert len(fig.axes) > 0
        matplotlib.pyplot.close(fig)
