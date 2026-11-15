import copy
import warnings

import numpy as np
import petsird
import pytest

from petsird_reader import UnsupportedPETSIRDFeatureError, check_header_support, get_num_data_bins
from scanner import create_parallelproj_lor_descriptor


def test_simulated_header_is_supported(tiny_header):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_header_support(tiny_header)


@pytest.mark.parametrize(
    "modify",
    [
        lambda h: setattr(h, "contains_weighted_event_time_blocks", True),
        lambda h: setattr(h.scanner, "prompt_event_policy", petsird.CoincidencePolicy.NONE),
    ],
)
def test_unsupported_features_raise(tiny_header, modify):
    h = copy.deepcopy(tiny_header)
    modify(h)
    with pytest.raises(UnsupportedPETSIRDFeatureError):
        check_header_support(h)


def test_ignored_features_warn(tiny_header):
    h = copy.deepcopy(tiny_header)
    h.scanner.gantry_alignment = petsird.RigidTransformation(
        matrix=np.array([[1, 0, 0, 5], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32")
    )
    h.scanner.delayed_event_policy = petsird.CoincidencePolicy.REJECT_HIGHER_MULTIPLES
    with pytest.warns(UserWarning) as record:
        check_header_support(h)
    messages = " ".join(str(r.message) for r in record)
    assert "gantry alignment" in messages and "delayed" in messages


def test_num_data_bins(tiny_config, tiny_header, tiny_tof_params):
    lor_desc = create_parallelproj_lor_descriptor(tiny_config)
    num_lors = lor_desc.num_block_pairs * tiny_config.num_det_els_per_module**2
    assert get_num_data_bins(tiny_header.scanner, tof=False) == num_lors
    assert get_num_data_bins(tiny_header.scanner, tof=True) == num_lors * tiny_tof_params.num_tofbins
