import itertools

import numpy as np
import pytest

from phantoms import PHANTOMS, create_phantom


@pytest.mark.parametrize("name", list(PHANTOMS))
def test_activity_inside_water_body(name):
    activity, attenuation = create_phantom(name, (55, 55, 19), (2.0, 2.0, 2.0))
    assert activity.max() > 0
    assert np.all(attenuation[activity > 0] > 0)


def test_offcenter_cylinder_has_no_symmetry():
    """axis flips and x/y swaps (e.g. in readers) are visible"""
    activity, _ = create_phantom("offcenter_cylinder", (55, 55, 19), (2.0, 2.0, 2.0))
    for swap in (False, True):
        for flips in itertools.product((False, True), repeat=3):
            if not swap and not any(flips):
                continue
            x = activity.transpose(1, 0, 2) if swap else activity
            x = np.flip(x, [i for i, f in enumerate(flips) if f]) if any(flips) else x
            assert np.linalg.norm(x - activity) > 0.5 * np.linalg.norm(activity)
