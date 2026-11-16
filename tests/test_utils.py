from utils import FWHM_PER_SIGMA, lm_file_fingerprint, load_or_compute


def test_fwhm_per_sigma():
    assert abs(FWHM_PER_SIGMA - 2.3548200450309493) < 1e-12


def test_fingerprint_changes_with_content(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1000)
    fp1 = lm_file_fingerprint(f)
    f.write_bytes(b"x" * 999 + b"y")
    assert lm_file_fingerprint(f) != fp1


def test_load_or_compute(tmp_path):
    path = tmp_path / "value.txt"
    calls = []

    def compute():
        calls.append(1)
        return "42"

    kw = dict(save=lambda p, x: p.write_text(x), load=lambda p: p.read_text())
    assert load_or_compute(path, compute, **kw) == "42"
    assert load_or_compute(path, compute, **kw) == "42"
    assert len(calls) == 1


def test_load_or_compute_recomputes_corrupt_cache(tmp_path):
    """a cache file that can not be loaded (e.g. incomplete) is recomputed and replaced"""
    import numpy as np
    import pytest

    path = tmp_path / "events.npz"
    np.savez_compressed(path, a=np.arange(100000))
    path.write_bytes(path.read_bytes()[:2000])  # truncated (interrupted write)
    kw = dict(save=lambda p, x: np.savez(p, a=x), load=lambda p: np.load(p)["a"])
    with pytest.warns(UserWarning, match="Could not load"):
        result = load_or_compute(path, lambda: np.arange(5), **kw)
    np.testing.assert_array_equal(result, np.arange(5))
    np.testing.assert_array_equal(load_or_compute(path, lambda: None, **kw), np.arange(5))


def test_load_or_compute_interrupted_save_leaves_no_cache(tmp_path):
    """if saving fails (e.g. the run is interrupted), neither the cache file nor a
    temporary file remains"""
    import pytest

    path = tmp_path / "value.npy"

    def save(p, x):
        p.write_bytes(b"incomplete")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        load_or_compute(path, lambda: 1, save=save, load=lambda p: None)
    assert list(tmp_path.iterdir()) == []
