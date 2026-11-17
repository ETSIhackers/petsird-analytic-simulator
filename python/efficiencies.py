"""detection efficiency model of the simulated scanner

The efficiency model is defined once (EfficiencyModel) and converted into
- the efficiencies of all LORs of the parallelproj LOR descriptor (used for the
  simulation) and
- the PETSIRD DetectionEfficiencies (written to the PETSIRD file)
such that both are consistent by construction.

efficiency(LOR) = det_el_effs[module 1, element 1] * det_el_effs[module 2, element 2]
                  * module_pair_effs[offset(module 1, module 2)][element 1, element 2]

where module 1 is the module with the higher index ("first" module in the PETSIRD
convention of lower-triangular module pairs).
"""

from dataclasses import dataclass

import numpy as np
import petsird
from parallelproj.pet_lors import EqualBlockPETLORDescriptor
from petsird.helpers.create import construct_lower_triangular_matrix

from scanner import ScannerConfig, coincident_module_pairs


def module_offset(i_mod_1: int, i_mod_2: int, num_modules: int) -> int:
    """offset (i_mod_1 - i_mod_2) mod num_modules between the first and second module of a pair

    For our scanner with rotationally symmetric modules, two module pairs with the
    same offset are related by a rotation that maps the first (second) module of one
    pair onto the first (second) module of the other pair. Therefore, all module pairs
    with the same offset have the same module pair efficiencies, i.e. they form a
    PETSIRD symmetry group.
    Note: since PETSIRD stores module pairs (of the same type) in lower-triangular order
    (first module index >= second module index), a rotation can swap the order of the
    modules, which maps offset d to num_modules - d. The module pair efficiencies of
    offset num_modules - d are therefore the transpose of the ones of offset d.
    """
    return (i_mod_1 - i_mod_2) % num_modules


@dataclass
class EfficiencyModel:
    # detecting element efficiencies, shape (num_modules, num_det_els_per_module)
    det_el_effs: np.ndarray
    # module pair efficiencies: offset -> matrix (num_det_els, num_det_els) indexed by
    # [element in first module, element in second module]
    module_pair_effs: dict[int, np.ndarray]

    @property
    def num_modules(self) -> int:
        return self.det_el_effs.shape[0]

    def sgid(self, offset: int) -> int:
        """symmetry group ID of a module offset (one SGID per offset)"""
        return sorted(self.module_pair_effs).index(offset)


# efficiency models of generate_efficiency_model
EFFICIENCY_MODELS = ("structured", "random")


def _normalized_index(n: int) -> np.ndarray:
    """0 ... 1 for n indices (0 for a single index)"""
    return np.arange(n) / (n - 1) if n > 1 else np.zeros(n)


def _structured_det_el_effs(config: ScannerConfig) -> np.ndarray:
    """structured (exaggerated, non-physical) detecting element efficiencies

    eff(m, t, r, a) = gain(m) * tangential(t) * layer(r) * axial(a) * (1 + jitter)

    with the module m, and the tangential / radial (depth layer) / axial index t, r, a of
    the detecting element in its module (block_shape order):
    - gain(m) = 1 + 0.6 sin(2 pi m / M + 0.4), module 2 x 1.8: sector artifacts
    - tangential(t) = 0.5 + 0.8 u_t: linear ramp, not mirror symmetric
    - layer(r) = 1.4 - 0.8 u_r: front layer 1.4, back layer 0.6
    - axial(a) = 1.2 - 2 (u_a - 0.35)^2: off-center peak, low end planes, not symmetric
    - jitter: uniform in [-0.1, 0.1]
    (u: index normalized to [0, 1]) and a "dead" patch (efficiency 0) of up to 3 x 3
    detecting elements (tangential x axial) in the front layer of module 7 / 12 * M.
    None of the components is symmetric under the flips / swaps of indices that a
    reader could get wrong.
    """
    nt, nr, na = config.block_shape
    u_t, u_r, u_a = (_normalized_index(n) for n in (nt, nr, na))
    num_modules = config.num_modules

    gain = 1 + 0.6 * np.sin(2 * np.pi * np.arange(num_modules) / num_modules + 0.4)
    if num_modules > 2:
        gain[2] *= 1.8
    tangential = 0.5 + 0.8 * u_t
    layer = 1.4 - 0.8 * u_r
    axial = 1.2 - 2.0 * (u_a - 0.35) ** 2

    # (module, t, r, a) -> flattened detecting element index = np.ravel_multi_index((t, r, a))
    effs = (
        gain[:, None, None, None]
        * tangential[None, :, None, None]
        * layer[None, None, :, None]
        * axial[None, None, None, :]
    )
    effs = effs * (1 + np.random.uniform(-0.1, 0.1, effs.shape))

    # dead patch (central 3 x 3 tangential x axial elements of the front layer)
    t0, a0 = max(0, nt // 2 - 1), max(0, na // 2 - 1)
    effs[(7 * num_modules) // 12, t0 : t0 + 3, 0, a0 : a0 + 3] = 0
    return effs.reshape(num_modules, -1).astype("float32")


def _structured_module_pair_effs(config: ScannerConfig, d: int) -> np.ndarray:
    """structured (exaggerated) module pair efficiencies of module offset d

    eff[e1, e2] = level(d) * tangential(t1, t2) * axial(a1, a2) * layers(r1, r2) * (1 + jitter)

    for the detecting elements e1 (first module of the pair) and e2 (second module):
    - level(d) = 1 + 0.5 (-1)^d: alternating between module offsets (symmetry groups)
    - tangential(t1, t2) = 1 + 0.5 cos(pi (u_t1 + u_t2)) + 0.3 (u_t1 - u_t2): depends on
      the radial position of the LOR in the module pair (ring artifacts), not symmetric
    - axial(a1, a2) = 1.3 - 0.6 |u_a1 - u_a2|: oblique LORs less efficient
    - layers(r1, r2) = 1 + 0.6 (u_r2 - u_r1): front -> back 1.6, back -> front 0.4, i.e.
      strongly non-symmetric (transposition errors are visible)
    - jitter: uniform in [-0.1, 0.1]
    """
    nt, nr, na = config.block_shape
    t, r, a = np.unravel_index(np.arange(config.num_det_els_per_module), config.block_shape)
    u_t = _normalized_index(nt)[t]
    u_r = _normalized_index(nr)[r]
    u_a = _normalized_index(na)[a]
    tangential = (
        1
        + 0.5 * np.cos(np.pi * (u_t[:, None] + u_t[None, :]))
        + 0.3 * (u_t[:, None] - u_t[None, :])
    )
    axial = 1.3 - 0.6 * np.abs(u_a[:, None] - u_a[None, :])
    layers = 1 + 0.6 * (u_r[None, :] - u_r[:, None])
    m = (1 + 0.5 * (-1) ** d) * tangential * axial * layers
    return m * (1 + np.random.uniform(-0.1, 0.1, m.shape))


def generate_efficiency_model(
    config: ScannerConfig,
    uniform_crystal_eff: bool = False,
    uniform_sg_eff: bool = False,
    model: str = "structured",
) -> EfficiencyModel:
    """non-physical efficiency model (uses the global numpy random state)

    model "structured" (default): exaggerated, structured efficiencies (see
    _structured_det_el_effs and _structured_module_pair_effs), such that ignoring or
    misinterpreting them gives clear, recognizable artifacts (sectors, rings, axial
    bands, a cold streak from the dead patch) and with as few symmetries as possible.

    model "random" (previous model):
    - detecting element efficiencies: uniform in [0.2, 2.2], x3 in the first module,
      /3 in the last module, elements with efficiency < 0.21 are "dead" (0)
    - module pair efficiencies: random non-symmetric matrices with a level alternating
      between module offsets

    In both models, the module pair efficiencies are consistent with the rotational
    symmetry of the scanner (matrix of offset n - d = transpose of offset d, symmetric
    matrix for opposite modules), such that they can be encoded with one SGID per offset.
    """
    if model not in EFFICIENCY_MODELS:
        raise ValueError(f"unknown efficiency model {model}, available: {EFFICIENCY_MODELS}")
    num_det_els = config.num_det_els_per_module

    if uniform_crystal_eff:
        det_el_effs = np.ones((config.num_modules, num_det_els), dtype="float32")
    elif model == "structured":
        det_el_effs = _structured_det_el_effs(config)
    else:
        det_el_effs = 0.2 + 2 * np.astype(
            np.random.rand(config.num_modules, num_det_els), "float32"
        )
        det_el_effs[0, :] *= 3
        det_el_effs[-1, :] /= 3
        det_el_effs[det_el_effs < 0.21] = 0

    offsets = sorted(
        {module_offset(j, i, config.num_modules) for i, j in coincident_module_pairs(config)}
    )
    module_pair_effs = {}
    for d in offsets:
        if uniform_sg_eff:
            module_pair_effs[d] = np.ones((num_det_els, num_det_els), dtype="float32")
        elif (config.num_modules - d) in module_pair_effs:
            module_pair_effs[d] = module_pair_effs[config.num_modules - d].T.copy()
        else:
            if model == "structured":
                m = _structured_module_pair_effs(config, d)
            else:
                # level alternating between offsets (same level for d and num_modules - d)
                m = (1 + 0.5 * (-1) ** d) * np.random.uniform(0.5, 1.5, (num_det_els, num_det_els))
            if 2 * d == config.num_modules:
                # opposite modules: matrix has to be symmetric
                m = 0.5 * (m + m.T)
            module_pair_effs[d] = m.astype("float32")

    return EfficiencyModel(det_el_effs=det_el_effs, module_pair_effs=module_pair_effs)


def lor_efficiencies(
    model: EfficiencyModel, lor_desc: EqualBlockPETLORDescriptor
) -> np.ndarray:
    """efficiencies of all LORs of a parallelproj LOR descriptor

    Returns an array of shape (num_block_pairs, num_lors_per_block_pair).
    The LORs of a block pair (start module < end module) are ordered by
    (start element, end element). The end module is the first module of the pair in
    the PETSIRD convention.
    """
    num_det_els = lor_desc.num_lorendpoints_per_block
    start_el, end_el = np.meshgrid(np.arange(num_det_els), np.arange(num_det_els), indexing="ij")
    start_el, end_el = start_el.ravel(), end_el.ravel()

    effs = np.ones((lor_desc.num_block_pairs, num_det_els**2), dtype="float32")
    for i, (start_mod, end_mod) in enumerate(lor_desc.all_block_pairs):
        mp_eff = model.module_pair_effs[
            module_offset(int(end_mod), int(start_mod), model.num_modules)
        ]
        effs[i, ...] = (
            mp_eff[end_el, start_el]
            * model.det_el_effs[start_mod, start_el]
            * model.det_el_effs[end_mod, end_el]
        )
    return effs


def to_petsird_detection_efficiencies(
    model: EfficiencyModel, config: ScannerConfig, num_energy_bins: int
) -> petsird.DetectionEfficiencies:
    """PETSIRD detection efficiencies of the (only) module type

    - detection bin efficiencies: the detection bin index runs over
      (module, element, energy bin) with the energy bin as the fastest index
    - lower-triangular module pair SGID LUT (-1 for module pairs not in coincidence)
    - one module pair efficiency matrix per SGID:
      values[detection bin in first module][detection bin in second module]
    """
    sgid_lut = construct_lower_triangular_matrix(config.num_modules, dtype=int, value=-1)
    for i, j in coincident_module_pairs(config):
        # lower triangular: first module index >= second module index
        sgid_lut[j][i] = model.sgid(module_offset(j, i, config.num_modules))

    module_pair_efficiencies_vector = [
        petsird.ModulePairEfficiencies(
            values=np.kron(
                model.module_pair_effs[d],
                np.ones((num_energy_bins, num_energy_bins), dtype="float32"),
            ).tolist(),
            sgid=model.sgid(d),
        )
        for d in sorted(model.module_pair_effs)
    ]

    return petsird.DetectionEfficiencies(
        method_description="simulated random crystal and symmetry group efficiencies (petsird-analytic-simulator)",
        calibration_factor=1.0,
        detection_bin_efficiencies=[
            np.repeat(model.det_el_effs.ravel(), num_energy_bins).tolist()
        ],
        module_pair_sgidlut=[[sgid_lut]],
        module_pair_efficiencies_vectors=[[module_pair_efficiencies_vector]],
    )
