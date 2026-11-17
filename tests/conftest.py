"""shared fixtures: a tiny scanner (fast tests) and its PETSIRD header"""

import numpy as np
import pytest

from efficiencies import generate_efficiency_model
from scanner import ScannerConfig
from simulation import create_petsird_header, tof_parameters

TINY_IMG_SHAPE = (16, 16, 4)
TINY_VOXEL_SIZE = (4.0, 4.0, 4.0)


@pytest.fixture
def tiny_config() -> ScannerConfig:
    # 2 x 8 modules (4 panels of 2 modules at 2 axial positions) with 3 x 2 x 2 detecting
    # elements, opposite modules are in coincidence
    return ScannerConfig(
        num_transaxial_modules=8,
        num_axial_modules=2,
        radius=60.0,
        block_shape=(3, 2, 2),
        block_spacing=(4.0, 6.0, 4.0),
        min_coincidence_distance=2,
    )


@pytest.fixture
def tiny_efficiency_model(tiny_config):
    np.random.seed(1)
    return generate_efficiency_model(tiny_config)


@pytest.fixture
def tiny_tof_params():
    return tof_parameters(20.0, TINY_IMG_SHAPE, TINY_VOXEL_SIZE)


@pytest.fixture
def tiny_header(tiny_config, tiny_efficiency_model, tiny_tof_params):
    return create_petsird_header(tiny_config, tiny_efficiency_model, tiny_tof_params, 20.0)
