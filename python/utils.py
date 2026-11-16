"""small generic helpers"""

import hashlib
import os
import sys
import time
import warnings
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np


def check_petsird_version(min_version: tuple[int, int, int] = (0, 13, 2)) -> None:
    """raise an ImportError if the installed petsird version is too old"""
    petsird_version = tuple(map(int, version("petsird").split(".")))
    if petsird_version < min_version:
        raise ImportError(
            f"petsird version {petsird_version} is not supported, please install "
            f"petsird >= {'.'.join(map(str, min_version))}"
        )


check_petsird_version()

# optional compiled package to read / write the time blocks of PETSIRD files with the
# yardl-generated C++ code (see petsird_fast/ in this repository)
# (from the repository root, the source folder petsird_fast/ can be imported as an empty
# namespace package, hence the check for the compiled classes)
try:
    import petsird_fast

    petsird_fast.TimeBlockReader, petsird_fast.TimeBlockWriter  # noqa: B018
except (ImportError, AttributeError):
    petsird_fast = None

# backends to read / write PETSIRD time blocks: yardl-generated Python code (petsird) or
# yardl-generated C++ code (petsird_fast); "auto": petsird_fast if installed
PETSIRD_BACKENDS = ("auto", "petsird", "petsird_fast")


def petsird_fast_available() -> bool:
    """whether the optional compiled package petsird_fast is installed"""
    return petsird_fast is not None


def resolve_petsird_backend(backend: str = "auto") -> str:
    """'petsird' or 'petsird_fast' for a backend in PETSIRD_BACKENDS"""
    if backend not in PETSIRD_BACKENDS:
        raise ValueError(f"unknown PETSIRD backend {backend}, available: {PETSIRD_BACKENDS}")
    if backend == "auto":
        return "petsird_fast" if petsird_fast_available() else "petsird"
    if backend == "petsird_fast" and not petsird_fast_available():
        raise ImportError(
            "the PETSIRD backend petsird_fast is not installed (see petsird_fast/README.md)"
        )
    return backend


# ratio between the FWHM and the standard deviation of a Gaussian: 2 * sqrt(2 * ln(2))
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def lm_file_fingerprint(fname: str | Path, chunk_size: int = 4 * 1024**2) -> str:
    """short content fingerprint of a (PETSIRD listmode) file

    Used to tag cached / derived files such that they are not reused for a different
    file with the same name (e.g. after re-running the simulation into the same
    directory). To be fast also for large files, only the file size, the first and
    the last chunk of the file are hashed. The first chunk contains the PETSIRD
    header (incl. the detection efficiencies), the last chunk the last events.
    """
    fname = Path(fname)
    size = fname.stat().st_size

    hash_obj = hashlib.sha256(str(size).encode())
    with open(fname, "rb") as f:
        hash_obj.update(f.read(chunk_size))
        if size > chunk_size:
            f.seek(max(chunk_size, size - chunk_size))
            hash_obj.update(f.read(chunk_size))

    return hash_obj.hexdigest()[:8]


def load_or_compute(
    path: str | Path,
    compute: Callable[[], Any],
    save: Callable[[Path, Any], None],
    load: Callable[[Path], Any],
    description: str = "",
) -> Any:
    """load a cached result from path if it exists, otherwise compute and save it

    The result is first saved to a temporary file in the same directory, which is
    renamed to path afterwards, such that an interrupted run never leaves an incomplete
    cache file behind. If a cache file can not be loaded (e.g. an incomplete file of an
    older version), the result is recomputed and the cache file is replaced.
    """
    path = Path(path)
    if path.exists():
        print(f"Loading {description} from {path}")
        try:
            return load(path)
        except Exception as e:  # noqa: BLE001 - any error of a corrupt cache file
            warnings.warn(
                f"Could not load the cached {description} from {path} ({type(e).__name__}: "
                f"{e}), recomputing it.",
                UserWarning,
            )
    print(f"Calculating {description} ...")
    result = compute()
    # same suffix (e.g. .npy / .npz) such that numpy does not append another one
    tmp_path = path.with_name(f"{path.stem}.tmp{os.getpid()}{path.suffix}")
    try:
        save(tmp_path, result)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"Saved {description} to {path}")
    return result


class Progress:
    """single-line progress display (updated in place, at most every 0.2 s)

    Usage:
        progress = Progress("Writing time blocks", total=100)
        for ...:
            progress.update(1, events=num_events_so_far)
        progress.close()

    Without total (e.g. reading a PETSIRD file, whose number of time blocks is unknown
    in advance), the count, the events and the elapsed time are shown.
    """

    def __init__(self, description: str, total: int | None = None, unit: str = "time blocks"):
        self.description = description
        self.total = total
        self.unit = unit
        self.count = 0
        self.events = 0
        self._start = time.perf_counter()
        self._last_print = 0.0

    def _line(self) -> str:
        elapsed = time.perf_counter() - self._start
        count = f"{self.count}/{self.total}" if self.total else f"{self.count}"
        percent = f" ({100 * self.count / self.total:3.0f}%)" if self.total else ""
        rate = f", {self.events / elapsed / 1e6:.1f} M events/s" if elapsed > 0 and self.events else ""
        return (
            f"{self.description}: {count} {self.unit}{percent}, "
            f"{self.events:,} events, {elapsed:.1f} s{rate}"
        )

    def update(self, n: int = 1, events: int | None = None) -> None:
        self.count += n
        if events is not None:
            self.events = events
        now = time.perf_counter()
        if now - self._last_print > 0.2 or (self.total and self.count >= self.total):
            self._last_print = now
            sys.stdout.write("\r" + self._line())
            sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\r" + self._line() + "\n")
        sys.stdout.flush()
