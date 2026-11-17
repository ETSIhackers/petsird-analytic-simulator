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



def test_uniform_cylinder_text():
    from phantoms import default_letter_z_mm, uniform_cylinder, uniform_cylinder_text

    shape, vs = (55, 55, 21), (2.0, 2.0, 2.0)
    img, water = uniform_cylinder_text(shape, vs, text="PET", contrast=0.15)
    base, base_water = uniform_cylinder(shape, vs)
    np.testing.assert_array_equal(water, base_water)
    letters = img - base
    # letters only inside of the cylinder, contrast at most 0.15
    assert letters.max() == pytest.approx(0.15) and letters.min() == 0
    assert np.all(letters[base == 0] == 0)
    z = (np.arange(shape[2]) - 10) * 2.0
    letter_z = default_letter_z_mm(shape, vs, 3)
    # letters only within 3 mm of their axial centers, in all 3 axial sections
    expected = np.any([np.abs(z - zc) < 3 for zc in letter_z], axis=0)
    np.testing.assert_array_equal((letters > 0.01).any(axis=(0, 1)), expected)


def test_letter_orientation():
    """up = -y: the bowl of the P is at -y (top) and +x (right), the stem at -x"""
    from phantoms import add_letters

    shape, vs = (55, 55, 3), (2.0, 2.0, 2.0)
    zeros, ones = np.zeros(shape, np.float32), np.ones(shape, bool)
    img = add_letters(zeros, ones, vs, text="P", contrast=1.0, letter_z_mm=(0.0,), up="-y")
    letter = img[:, :, 1] > 0.5
    xs, ys = np.nonzero(letter)
    y0, y1 = ys.min(), ys.max()
    top, bottom = ys < y0 + 0.25 * (y1 - y0), ys > y0 + 0.75 * (y1 - y0)
    # the top quarter (towards -y) of the P (bowl) is wider than the bottom quarter (stem)
    assert np.ptp(xs[top]) > 1.5 * np.ptp(xs[bottom])
    # the stem is on the left (-x)
    assert xs[bottom].mean() < xs.mean()
    # up = +y mirrors the letter in y
    img_up = add_letters(zeros, ones, vs, text="P", contrast=1.0, letter_z_mm=(0.0,), up="+y")
    np.testing.assert_allclose(img_up[:, :, 1], img[:, ::-1, 1], atol=1e-6)


@pytest.mark.parametrize("letter", ["O", "P", "R", "A"])
def test_letters_have_empty_counters(letter):
    """the counters (holes) of letters are not filled"""
    from phantoms import add_letters

    shape, vs = (55, 55, 3), (2.0, 2.0, 2.0)
    zeros, ones = np.zeros(shape, np.float32), np.ones(shape, bool)
    img = add_letters(zeros, ones, vs, text=letter, contrast=1.0, letter_z_mm=(0.0,))[:, :, 1]
    letter_mask = img > 0.5
    # fill the letter from the outside: background pixels not reachable from the image
    # border are enclosed by the letter (counter)
    outside = np.zeros_like(letter_mask)
    outside[0, :] = outside[-1, :] = outside[:, 0] = outside[:, -1] = True
    outside &= ~letter_mask
    while True:
        grown = outside.copy()
        grown[1:] |= outside[:-1]
        grown[:-1] |= outside[1:]
        grown[:, 1:] |= outside[:, :-1]
        grown[:, :-1] |= outside[:, 1:]
        grown &= ~letter_mask
        if np.array_equal(grown, outside):
            break
        outside = grown
    counter = ~letter_mask & ~outside
    assert counter.sum() >= 4, f"no empty counter in {letter}"
