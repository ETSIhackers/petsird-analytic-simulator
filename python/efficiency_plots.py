"""visualization of the detection efficiencies encoded in a PETSIRD file

Generic for any PETSIRD file (only the header is used). All plots are in index space,
i.e. they show the encoded arrays as a reader indexes them (module / detecting element
numbers), which works for any arrangement of the modules. The positions of the detection
bins (ADPs) are part of the reference table.

- plot_detection_bin_efficiencies: detection bin efficiencies, module x detecting element
- plot_sgid_luts: module pair SGID LUTs (module pairs in coincidence, symmetry groups)
- plot_module_pair_efficiencies: module pair efficiency matrix of every SGID,
  values[element of the first module, element of the second module]
- fan_efficiencies / plot_fans: total efficiency of all LORs of a detection bin
- efficiency_reference_table: decoded efficiencies of LORs, computed with the official
  petsird.helpers (independent of the readers of this repository), to compare readers
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import petsird
from petsird.helpers import expand_detection_bin, get_detection_efficiency
from petsird.helpers.geometry import get_ADP

from petsird_reader import (
    detection_bin_tables,
    get_all_detector_centers,
    get_module_pair_efficiencies,
    get_module_pair_sgid_lut,
    module_type_info,
)


def _detection_bin_effs(scanner, module_type, energy_bin):
    """detection bin efficiencies (num_modules, num_det_els) of one energy bin"""
    tables = detection_bin_tables(scanner, get_all_detector_centers(scanner.scanner_geometry))
    num_modules, num_det_els, num_energy_bins = module_type_info(scanner, module_type)
    return tables.effs[module_type].reshape(num_modules, num_det_els, num_energy_bins)[..., energy_bin]


def _index_ticks(set_ticks, n: int, num_ticks: int = 12) -> None:
    """integer ticks (index numbers) at about num_ticks positions of an axis of length n"""
    step = max(1, int(np.ceil(n / num_ticks)))
    set_ticks(np.arange(0, n, step))


def _module_element_image(ax, values, cmap, vmin=None, vmax=None, zero_marker=True):
    """image of values[module, element] (rows: modules, columns: detecting elements)"""
    num_modules, num_det_els = values.shape
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if zero_marker:
        rows, cols = np.nonzero(values == 0)
        if rows.size > 0:
            ax.scatter(cols, rows, marker="x", s=10, c="red", linewidths=0.8, label="efficiency 0")
            ax.legend(loc="upper right", fontsize=8)
    _index_ticks(ax.set_xticks, num_det_els, 18)
    _index_ticks(ax.set_yticks, num_modules, 24)
    ax.set_xlabel("detecting element")
    ax.set_ylabel("module")
    return im


def plot_detection_bin_efficiencies(
    scanner: petsird.ScannerInformation, module_type: int = 0, energy_bin: int = 0
) -> plt.Figure:
    """detection bin efficiencies as image module x detecting element (one energy bin);
    efficiency 0 is marked with a red x"""
    effs = _detection_bin_effs(scanner, module_type, energy_bin)
    fig, ax = plt.subplots(figsize=(13, 4.2), layout="constrained")
    im = _module_element_image(ax, effs, "viridis", vmin=0)
    fig.colorbar(im, ax=ax, label="efficiency")
    ax.set_title(
        f"detection bin efficiencies (module type {module_type}, energy bin {energy_bin}): "
        "detection_bin_efficiencies[type][(module * num_elements + element) * num_energy_bins + energy]",
        fontsize=10,
    )
    return fig


def plot_sgid_luts(scanner: petsird.ScannerInformation) -> plt.Figure:
    """module pair SGID LUTs of all module type pairs (-1 / white: not in coincidence)"""
    num_types = scanner.scanner_geometry.number_of_module_types()
    pairs = [(m1, m2) for m1 in range(num_types) for m2 in range(m1 + 1)]
    fig, axs = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5), layout="constrained", squeeze=False)
    for ax, (m1, m2) in zip(axs[0], pairs):
        lut = get_module_pair_sgid_lut(scanner, m1, m2)
        masked = np.ma.masked_less(lut, 0)
        ax.imshow(masked, cmap="tab20", interpolation="nearest")
        if lut.size <= 900:
            for (i, j), v in np.ndenumerate(lut):
                if v >= 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=7)
        _index_ticks(ax.set_xticks, lut.shape[1], 24)
        _index_ticks(ax.set_yticks, lut.shape[0], 24)
        ax.set_xlabel(f"module (type {m2}) of the second detection bin")
        ax.set_ylabel(f"module (type {m1}) of the first detection bin")
        ax.set_title(f"SGID LUT, module types ({m1}, {m2})\n(white: not in coincidence)", fontsize=10)
    return fig


def plot_module_pair_efficiencies(
    scanner: petsird.ScannerInformation,
    module_types: tuple[int, int] = (0, 0),
    energy_bins: tuple[int, int] = (0, 0),
) -> plt.Figure:
    """module pair efficiency matrices of all SGIDs,
    values[element of the first module, element of the second module] (selected energy bins)"""
    m1, m2 = module_types
    mp = get_module_pair_efficiencies(scanner, m1, m2)
    _, n1, e1 = module_type_info(scanner, m1)
    _, n2, e2 = module_type_info(scanner, m2)
    mp = mp.reshape(-1, n1, e1, n2, e2)[:, :, energy_bins[0], :, energy_bins[1]]
    num_sgids = mp.shape[0]
    ncols = min(4, num_sgids)
    nrows = int(np.ceil(num_sgids / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1, 3.6 * nrows + 0.6), layout="constrained", squeeze=False)
    vmax = np.percentile(mp, 99)
    for k, ax in enumerate(axs.ravel()):
        if k >= num_sgids:
            ax.axis("off")
            continue
        im = ax.imshow(mp[k], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"SGID {k}", fontsize=10)
        _index_ticks(ax.set_xticks, n2)
        _index_ticks(ax.set_yticks, n1)
        ax.tick_params(labelsize=6, length=2)
        ax.tick_params(axis="x", labelrotation=90)
    axs[0, 0].set_ylabel("detecting element of the first module")
    axs[-1, 0].set_xlabel("detecting element of the second module")
    fig.colorbar(im, ax=axs, shrink=0.6, label="module pair efficiency")
    fig.suptitle(
        f"module pair efficiencies values[element of the first module][element of the second module] "
        f"(module types {module_types}, energy bins {energy_bins})"
    )
    return fig


def fan_efficiencies(scanner: petsird.ScannerInformation, detection_bin: int, module_type: int = 0) -> np.ndarray:
    """total efficiency of the LORs of a detection bin with all detection bins of the same
    module type (NaN if not in coincidence), shape (num_detection_bins,)

    The LOR (a, b) is ordered following PETSIRD (first detection bin = the larger one).
    """
    tables = detection_bin_tables(scanner, get_all_detector_centers(scanner.scanner_geometry))
    effs = tables.effs[module_type]
    nbm = tables.bins_per_module[module_type]
    b = np.arange(effs.size)
    a = np.full_like(b, detection_bin)
    d0, d1 = np.maximum(a, b), np.minimum(a, b)
    lut = tables.sgid_luts[(module_type, module_type)]
    sgid = lut[d0 // nbm, d1 // nbm]
    mp = tables.module_pair_effs[(module_type, module_type)]
    total = tables.calibration_factor * effs[d0] * effs[d1] * mp[(np.maximum(sgid, 0) * nbm + d0 % nbm) * nbm + d1 % nbm]
    return np.where((sgid >= 0) & (b != detection_bin), total, np.nan)


def plot_fans(scanner: petsird.ScannerInformation, detection_bins: list[int], module_type: int = 0, energy_bin: int = 0) -> plt.Figure:
    """total efficiency of all LORs of some detection bins, as image module x detecting
    element of the second detection bin (energy bin energy_bin; white: not in coincidence)"""
    num_modules, num_det_els, num_energy_bins = module_type_info(scanner, module_type)
    fig, axs = plt.subplots(len(detection_bins), 1, figsize=(13, 3.6 * len(detection_bins) + 0.4), layout="constrained", squeeze=False)
    for ax, a in zip(axs[:, 0], detection_bins):
        fan = fan_efficiencies(scanner, a, module_type).reshape(num_modules, num_det_els, num_energy_bins)[..., energy_bin]
        im = _module_element_image(ax, fan, "magma", zero_marker=False)
        ma, rest = divmod(a, num_det_els * num_energy_bins)
        el = rest // num_energy_bins
        ax.scatter([el], [ma], marker="*", s=160, c="cyan", edgecolors="k", label=f"detection bin {a} (module {ma}, element {el})")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(im, ax=ax, label="LOR efficiency")
    fig.suptitle(f"total efficiency of all LORs of a detection bin, at the second detection bin (energy bin {energy_bin}; white: not in coincidence)")
    return fig


def _raw_components(scanner, m1, m2, b0, b1):
    """efficiency components of a LOR read directly from the PETSIRD tables (empty / missing
    tables mean 1, see petsird.DetectionEfficiencies)"""
    effs = scanner.detection_efficiencies
    dbe = effs.detection_bin_efficiencies
    e0 = float(dbe[m1][b0]) if dbe and len(dbe[m1]) > 0 else 1.0
    e1 = float(dbe[m2][b1]) if dbe and len(dbe[m2]) > 0 else 1.0
    x0, x1 = expand_detection_bin(scanner, m1, b0), expand_detection_bin(scanner, m2, b1)
    lut = effs.module_pair_sgidlut[m1][m2] if effs.module_pair_sgidlut else []
    sgid = int(lut[x0.module_index][x1.module_index]) if len(lut) > 0 else 0
    mpv = effs.module_pair_efficiencies_vectors[m1][m2] if effs.module_pair_efficiencies_vectors else []
    mp = 1.0
    if sgid < 0:
        mp = 0.0
    elif len(mpv) > 0:
        _, n2, en2 = module_type_info(scanner, m2)
        _, n1, en1 = module_type_info(scanner, m1)
        values = {v.sgid: v.values for v in mpv}[sgid]
        mp = float(np.asarray(values)[b0 % (n1 * en1), b1 % (n2 * en2)])
    return x0, x1, sgid, e0, e1, mp, float(effs.calibration_factor)


REFERENCE_COLUMNS = [
    "lor_kind", "type_0", "type_1", "detection_bin_0", "detection_bin_1",
    "module_0", "element_0", "energy_0", "module_1", "element_1", "energy_1",
    "adp_0_x", "adp_0_y", "adp_0_z", "adp_1_x", "adp_1_y", "adp_1_z",
    "sgid", "detection_bin_efficiency_0", "detection_bin_efficiency_1",
    "module_pair_efficiency", "calibration_factor", "total_efficiency",
]


def efficiency_reference_table(
    scanner: petsird.ScannerInformation, num_random_lors: int = 500, seed: int = 0
) -> list[dict]:
    """decoded efficiencies of probe and random LORs (in coincidence) of all module type
    pairs, computed with the official petsird.helpers (expand_detection_bin, get_ADP,
    get_detection_efficiency) and the components read directly from the PETSIRD tables

    Probe LORs (module types (0, 0)): first and last detection bin each with the first
    detection bin of a module in coincidence, a detection bin with efficiency 0 (if any),
    opposite modules (largest module offset in coincidence).
    """
    rng = np.random.default_rng(seed)
    num_types = scanner.scanner_geometry.number_of_module_types()
    rows = []

    def add(kind, m1, m2, b0, b1):
        # PETSIRD order: first detection bin larger for equal module types
        if m1 == m2 and b0 < b1:
            b0, b1 = b1, b0
        tp = petsird.TypeOfModulePair((m1, m2))
        x0, x1, sgid, e0, e1, mp, cal = _raw_components(scanner, m1, m2, b0, b1)
        if sgid < 0:
            return False
        a0 = get_ADP(scanner, m1, x0).c
        a1 = get_ADP(scanner, m2, x1).c
        total = get_detection_efficiency(scanner, tp, b0, b1)
        rows.append(dict(zip(REFERENCE_COLUMNS, [
            kind, m1, m2, b0, b1, x0.module_index, x0.element_index, x0.energy_index,
            x1.module_index, x1.element_index, x1.energy_index, *map(float, a0), *map(float, a1),
            sgid, e0, e1, mp, cal, float(total)])))
        return True

    lut = get_module_pair_sgid_lut(scanner, 0, 0)
    num_modules, num_det_els, num_energy_bins = module_type_info(scanner, 0)
    nbm = num_det_els * num_energy_bins
    coinc = np.argwhere(np.maximum(lut, lut.T) >= 0)
    if coinc.size > 0:
        for kind, b in (("probe: first detection bin", 0), ("probe: last detection bin", num_modules * nbm - 1)):
            m = b // nbm
            partners = coinc[coinc[:, 0] == m, 1]
            if partners.size > 0:
                add(kind, 0, 0, b, int(partners[0]) * nbm)
        effs = _detection_bin_effs(scanner, 0, 0).ravel()
        dead = np.nonzero(effs == 0)[0]
        if dead.size > 0:
            b = int(dead[0]) * num_energy_bins
            partners = coinc[coinc[:, 0] == b // nbm, 1]
            if partners.size > 0:
                add("probe: detection bin with efficiency 0", 0, 0, b, int(partners[0]) * nbm + nbm // 2)
        offsets = (coinc[:, 0] - coinc[:, 1]) % num_modules
        dist = np.minimum(offsets, num_modules - offsets)
        i, j = coinc[np.argmax(dist)]
        add("probe: opposite modules", 0, 0, int(i) * nbm + nbm // 3, int(j) * nbm + (2 * nbm) // 3)

    pairs = [(m1, m2) for m1 in range(num_types) for m2 in range(m1 + 1)]
    added, attempts = 0, 0
    while added < num_random_lors and attempts < 50 * num_random_lors:
        attempts += 1
        m1, m2 = pairs[rng.integers(len(pairs))]
        n1 = int(np.prod(module_type_info(scanner, m1)))
        n2 = int(np.prod(module_type_info(scanner, m2)))
        added += add("random", m1, m2, int(rng.integers(n1)), int(rng.integers(n2)))
    return rows


def write_reference_table(rows: list[dict], fname: str | Path) -> None:
    with open(fname, "w", newline="", encoding="UTF-8") as f:
        writer = csv.DictWriter(f, fieldnames=REFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
