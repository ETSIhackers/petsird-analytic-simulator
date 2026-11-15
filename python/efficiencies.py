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


def generate_efficiency_model(
    config: ScannerConfig, uniform_crystal_eff: bool = False, uniform_sg_eff: bool = False
) -> EfficiencyModel:
    """random (non-physical) efficiency model (uses the global numpy random state)

    - detecting element efficiencies: uniform in [0.2, 2.2], x3 in the first module,
      /3 in the last module, elements with efficiency < 0.21 are "dead" (0)
    - module pair efficiencies: random non-symmetric matrices (to be able to detect
      transposition errors when reading the data) consistent with the rotational
      symmetry of the scanner (matrix of offset n - d = transpose of offset d)
    """
    num_det_els = config.num_det_els_per_module

    if uniform_crystal_eff:
        det_el_effs = np.ones((config.num_modules, num_det_els), dtype="float32")
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
            # level alternating between offsets (same level for d and num_modules - d)
            level = 1 + 0.5 * (-1) ** d
            m = np.random.uniform(0.5, 1.5, (num_det_els, num_det_els))
            if 2 * d == config.num_modules:
                # opposite modules: matrix has to be symmetric
                m = 0.5 * (m + m.T)
            module_pair_effs[d] = (level * m).astype("float32")

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
