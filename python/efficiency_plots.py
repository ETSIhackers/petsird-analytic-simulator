"""visualization of the detection efficiencies encoded in a PETSIRD file

Generic for any PETSIRD file (only the header is used). The efficiencies are shown in
geometric coordinates (positions of the detecting elements from their ADPs), such that
errors in the interpretation of the index order (flipped / swapped indices, wrong energy
bin stride, transposed matrices) change the smooth structures of the efficiencies into
visibly scrambled or mirrored patterns.

- detecting_element_layout: angle around the scanner axis, axial position and depth
  layer of every detecting element (from the ADPs)
- plot_detection_bin_efficiencies: "unrolled ring" maps (angle vs axial position) of the
  detection bin efficiencies, one per depth layer
- plot_sgid_luts: module pair SGID LUTs (module pairs in coincidence and their symmetry
  group)
- plot_module_pair_efficiencies: module pair efficiency matrix of every SGID, with the
  detecting elements sorted geometrically (layer, angle, axial position)
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


def detecting_element_layout(
    scanner: petsird.ScannerInformation, module_type: int
) -> dict[str, np.ndarray]:
    """geometric layout of the detecting elements of a module type (from their ADPs)

    Returns arrays of shape (num_modules, num_det_els):
    angle_deg: angle of the ADP around the scanner axis (z), in [-180, 180)
    z: axial position (mm)
    depth: position along the outward (transaxial) direction of the module (mm), relative
        to the module center
    layer: index of the depth layer (detecting elements with the same depth, rounded to
        0.1 mm; a single layer if the depths do not form a few distinct values)
    """
    adps = get_all_detector_centers(scanner.scanner_geometry)[module_type].astype(np.float64)
    angle = np.degrees(np.arctan2(adps[..., 1], adps[..., 0]))
    center = adps.mean(axis=1, keepdims=True)
    normal = center[..., :2] / np.linalg.norm(center[..., :2], axis=-1, keepdims=True)
    depth = np.sum((adps[..., :2] - center[..., :2]) * normal, axis=-1)
    depths = np.round(depth, 1)
    unique = np.unique(depths)
    layer = np.searchsorted(unique, depths) if unique.size <= 8 else np.zeros(depths.shape, int)
    return {"angle_deg": angle, "z": adps[..., 2], "depth": depth, "layer": layer}


def _detection_bin_effs(scanner, module_type, energy_bin):
    """detection bin efficiencies (num_modules, num_det_els) of one energy bin"""
    tables = detection_bin_tables(scanner, get_all_detector_centers(scanner.scanner_geometry))
    num_modules, num_det_els, num_energy_bins = module_type_info(scanner, module_type)
    return tables.effs[module_type].reshape(num_modules, num_det_els, num_energy_bins)[..., energy_bin]


def _module_angles(layout: dict[str, np.ndarray]) -> np.ndarray:
    """mean angle (deg, circular mean) of the detecting elements of every module"""
    rad = np.radians(layout["angle_deg"])
    return np.degrees(np.arctan2(np.sin(rad).mean(axis=1), np.cos(rad).mean(axis=1)))


def _annotate_modules(ax, layout: dict[str, np.ndarray]) -> None:
    """module numbers above the detecting elements of every module"""
    for m, angle in enumerate(_module_angles(layout)):
        ax.annotate(
            f"m{m}", (angle, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
            textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold",
        )


def plot_detection_bin_efficiencies(
    scanner: petsird.ScannerInformation, module_type: int = 0, energy_bin: int = 0
) -> plt.Figure:
    """unrolled ring maps (angle vs axial position) of the detection bin efficiencies,
    one panel per depth layer; dead elements (efficiency 0) are marked with x"""
    layout = detecting_element_layout(scanner, module_type)
    effs = _detection_bin_effs(scanner, module_type, energy_bin)
    layers = np.unique(layout["layer"])
    fig, axs = plt.subplots(len(layers), 1, figsize=(13, 2.6 * len(layers) + 0.8), layout="constrained", squeeze=False)
    vmax = np.percentile(effs[effs > 0], 99) if np.any(effs > 0) else 1
    for ax, lay in zip(axs[:, 0], layers):
        sel = layout["layer"] == lay
        sc = ax.scatter(layout["angle_deg"][sel], layout["z"][sel], c=effs[sel], marker="s", s=14, cmap="viridis", vmin=0, vmax=vmax)
        dead = sel & (effs == 0)
        ax.scatter(layout["angle_deg"][dead], layout["z"][dead], marker="x", s=14, c="red", label="efficiency 0")
        ax.set_title(f"depth layer {lay} (depth {layout['depth'][sel].mean():+.1f} mm from module center)", fontsize=10, pad=14)
        ax.set_xlabel("angle of the ADP around the scanner axis (deg)")
        ax.set_ylabel("z (mm)")
        ax.set_xlim(-180, 180)
        _annotate_modules(ax, layout)
        if np.any(dead):
            ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(sc, ax=ax, label="efficiency")
    fig.suptitle(f"detection bin efficiencies (module type {module_type}, energy bin {energy_bin}) at the positions of the detecting elements")
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
        ax.set_xlabel(f"module (type {m2}) of the second detection bin")
        ax.set_ylabel(f"module (type {m1}) of the first detection bin")
        ax.set_title(f"SGID LUT, module types ({m1}, {m2})\n(white: not in coincidence)", fontsize=10)
    return fig


def _geometric_order(scanner, module_type):
    """order of the detecting elements of a module by (layer, angle within module, z), and
    the layer boundaries in this order (module 0 is used for the angles)"""
    layout = detecting_element_layout(scanner, module_type)
    angle = layout["angle_deg"][0] - layout["angle_deg"][0].mean()
    order = np.lexsort((layout["z"][0], np.round(angle, 3), layout["layer"][0]))
    boundaries = np.nonzero(np.diff(layout["layer"][0][order]))[0] + 0.5
    return order, boundaries


def _element_ticks(set_ticks, order: np.ndarray, num_ticks: int = 12, rotation: int = 0) -> None:
    """ticks with the detecting element numbers at every n-th position of a sorted axis"""
    step = max(1, int(np.ceil(order.size / num_ticks)))
    positions = np.arange(0, order.size, step)
    set_ticks(positions, [str(order[p]) for p in positions], rotation=rotation)


def plot_module_pair_efficiencies(
    scanner: petsird.ScannerInformation,
    module_types: tuple[int, int] = (0, 0),
    energy_bins: tuple[int, int] = (0, 0),
) -> plt.Figure:
    """module pair efficiency matrices of all SGIDs, values[element of the first module,
    element of the second module], elements sorted by (layer, angle, z) (white lines:
    layer boundaries)"""
    m1, m2 = module_types
    mp = get_module_pair_efficiencies(scanner, m1, m2)
    _, n1, e1 = module_type_info(scanner, m1)
    _, n2, e2 = module_type_info(scanner, m2)
    mp = mp.reshape(-1, n1, e1, n2, e2)[:, :, energy_bins[0], :, energy_bins[1]]
    order1, b1 = _geometric_order(scanner, m1)
    order2, b2 = _geometric_order(scanner, m2)
    num_sgids = mp.shape[0]
    ncols = min(4, num_sgids)
    nrows = int(np.ceil(num_sgids / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 1, 3.6 * nrows + 0.6), layout="constrained", squeeze=False)
    vmax = np.percentile(mp, 99)
    for k, ax in enumerate(axs.ravel()):
        if k >= num_sgids:
            ax.axis("off")
            continue
        im = ax.imshow(mp[k][np.ix_(order1, order2)], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        for b in b1:
            ax.axhline(b, color="w", lw=0.8)
        for b in b2:
            ax.axvline(b, color="w", lw=0.8)
        ax.set_title(f"SGID {k}", fontsize=10)
        _element_ticks(ax.set_xticks, order2, rotation=90)
        _element_ticks(ax.set_yticks, order1)
        ax.tick_params(labelsize=6, length=2)
    axs[0, 0].set_ylabel("detecting element of the first module\n(sorted by layer, angle, z)")
    axs[-1, 0].set_xlabel("detecting element of the second module (sorted by layer, angle, z)")
    fig.colorbar(im, ax=axs, shrink=0.6, label="module pair efficiency")
    fig.suptitle(
        f"module pair efficiencies (module types {module_types}, energy bins {energy_bins}); tick labels: "
        "detecting element numbers, white lines: depth layer boundaries (front layer first)"
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
    """total efficiency of all LORs of some detection bins, at the positions of the other
    detection bins (front layer, energy bin energy_bin)"""
    layout = detecting_element_layout(scanner, module_type)
    num_modules, num_det_els, num_energy_bins = module_type_info(scanner, module_type)
    front = layout["layer"] == 0
    fig, axs = plt.subplots(len(detection_bins), 1, figsize=(13, 2.7 * len(detection_bins) + 0.6), layout="constrained", squeeze=False)
    for ax, a in zip(axs[:, 0], detection_bins):
        fan = fan_efficiencies(scanner, a, module_type).reshape(num_modules, num_det_els, num_energy_bins)[..., energy_bin]
        sc = ax.scatter(layout["angle_deg"][front], layout["z"][front], c=fan[front], marker="s", s=14, cmap="magma")
        ma, rest = divmod(a, num_det_els * num_energy_bins)
        el = rest // num_energy_bins
        ax.scatter(layout["angle_deg"][ma, el], layout["z"][ma, el], marker="*", s=160, c="cyan", edgecolors="k", label=f"detection bin {a} (module {ma}, element {el})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(-180, 180)
        _annotate_modules(ax, layout)
        ax.set_xlabel("angle around the scanner axis (deg)")
        ax.set_ylabel("z (mm)")
        fig.colorbar(sc, ax=ax, label="LOR efficiency")
    fig.suptitle(f"total efficiency of all LORs of a detection bin, at the second detection bin (front layer, energy bin {energy_bin}; empty: not in coincidence)")
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
