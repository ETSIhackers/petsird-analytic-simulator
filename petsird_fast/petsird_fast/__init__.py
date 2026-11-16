"""prototype: fast reading / writing of PETSIRD time blocks from / to flat arrays

The header of a PETSIRD file is small and is read with the yardl-generated Python
package (petsird). The time blocks are read with the yardl-generated C++ code
(libpetsird) by a compiled extension, which returns all events as one contiguous array
per field (struct of arrays). Python never handles individual events or time blocks.

Result of read_time_blocks() / of every chunk of iter_time_blocks():

    {
      "prompts":  {(type_1, type_2): coincidences, ...},   # type_2 <= type_1
      "delayeds": {(type_1, type_2): coincidences, ...},
      "event_time_blocks": {"start": uint32[K], "stop": uint32[K]},  # ms
      "singles": {                                          # level ALL only
          "counts": [uint64[Ks, num_detection_bins(type)] for every module type],
          "start": uint32[Ks], "stop": uint32[Ks],          # ms
      },
      "other": {name: count},  # e.g. dead_time_time_blocks, bed_movement_time_blocks,
                               # triple_events (not converted to arrays)
      "num_time_blocks": int,
    }

    coincidences = {
      "det_bin0": uint32[n], "det_bin1": uint32[n],  # PETSIRD detection bins
      "tof_idx": uint32[n],
      "block_offsets": int64[K + 1],  # events of event time block k: [offsets[k], offsets[k+1])
    }

All arrays are NumPy arrays that own memory allocated in C++ (no copies). Their fixed
dtypes and contiguous layout are chosen such that a later version can hand them to other
array frameworks (e.g. cupy / pytorch on GPU via DLPack) or fill buffers provided by
the caller.
"""

import tempfile
from collections.abc import Iterator, Mapping
from os import PathLike
from pathlib import Path

import numpy as np

from ._petsird_fast import TimeBlockReader
from ._petsird_fast import TimeBlockWriter as _TimeBlockWriter

__all__ = ["TimeBlockReader", "TimeBlockWriter", "read_time_blocks", "iter_time_blocks"]


def read_time_blocks(
    fname: str | PathLike, time_range_ms: tuple[int, int] | None = None
) -> dict:
    """read all time blocks of a PETSIRD binary file (see module docstring)

    time_range_ms: (start, stop) in ms, only time blocks (of all types) whose time
    interval lies within [start, stop] are converted, the others are only counted in
    "num_skipped_time_blocks" (all time blocks are still read from the file)
    """
    start, stop = time_range_ms if time_range_ms is not None else (None, None)
    return TimeBlockReader(str(fname)).read(0, start, stop)


def iter_time_blocks(
    fname: str | PathLike,
    max_time_blocks: int,
    time_range_ms: tuple[int, int] | None = None,
) -> Iterator[dict]:
    """read a PETSIRD binary file in chunks of max_time_blocks time blocks

    Every chunk has the format of read_time_blocks() (block_offsets refer to the
    event time blocks of the chunk). time_range_ms: see read_time_blocks().
    """
    if max_time_blocks < 1:
        raise ValueError("max_time_blocks has to be >= 1")
    start, stop = time_range_ms if time_range_ms is not None else (None, None)
    reader = TimeBlockReader(str(fname))
    while not reader.done:
        chunk = reader.read(max_time_blocks, start, stop)
        if chunk["num_time_blocks"] > 0:
            yield chunk


def _uint32(x) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.uint32)


def _coincidences(events: Mapping | None) -> dict | None:
    if events is None:
        return None
    return {
        (int(m1), int(m2)): tuple(_uint32(a) for a in arrays)
        for (m1, m2), arrays in events.items()
    }


class TimeBlockWriter:
    """write a PETSIRD binary file with time blocks given as flat arrays

    The header (a petsird.Header) is written with the yardl-generated Python code into a
    temporary PETSIRD file, from which the compiled extension reads it, such that the
    header types do not have to be bound to Python. All time blocks are written by the
    yardl-generated C++ code.

    Usage:
        with TimeBlockWriter(fname, header) as writer:
            for k in range(num_time_blocks):
                writer.add_event_time_block(start, stop, {(0, 0): (det_bin0, det_bin1, tof_idx)},
                                            delayeds={(0, 0): (...)})   # optional
                writer.add_singles_histogram_time_block(start, stop, [counts])  # optional
                writer.flush()  # writes the added time blocks as one batch

    prompts / delayeds map module type pairs (type_1, type_2 <= type_1) to arrays
    (det_bin0, det_bin1, tof_idx) of PETSIRD detection bins (uint32). Module type pairs
    that are not given have no events in this time block. delayeds=None means no
    delayed events in this time block. counts contains the singles counts of all
    detection bins (one array per module type, singles_histogram_level ALL).
    """

    def __init__(self, fname: str | PathLike, header):
        import petsird

        with tempfile.TemporaryDirectory() as tmp_dir:
            header_fname = str(Path(tmp_dir) / "header.bin")
            with petsird.BinaryPETSIRDWriter(header_fname) as header_writer:
                header_writer.write_header(header)
                header_writer.write_time_blocks([])
            self._writer = _TimeBlockWriter(str(fname), header_fname)

    def add_event_time_block(
        self, start: int, stop: int, prompts: Mapping, delayeds: Mapping | None = None
    ) -> None:
        self._writer.add_event_time_block(
            int(start), int(stop), _coincidences(prompts), _coincidences(delayeds)
        )

    def add_singles_histogram_time_block(self, start: int, stop: int, counts) -> None:
        self._writer.add_singles_histogram_time_block(
            int(start), int(stop), [np.ascontiguousarray(c, dtype=np.uint64) for c in counts]
        )

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "TimeBlockWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
