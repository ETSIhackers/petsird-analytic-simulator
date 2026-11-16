"""small generic helpers"""

import hashlib
import os
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
