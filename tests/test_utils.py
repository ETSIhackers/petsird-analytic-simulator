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
