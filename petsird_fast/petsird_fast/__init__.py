"""prototype: fast reading of PETSIRD time blocks into flat arrays

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

from collections.abc import Iterator
from os import PathLike

from ._petsird_fast import TimeBlockReader

__all__ = ["TimeBlockReader", "read_time_blocks", "iter_time_blocks"]


def read_time_blocks(fname: str | PathLike) -> dict:
    """read all time blocks of a PETSIRD binary file (see module docstring)"""
    return TimeBlockReader(str(fname)).read()


def iter_time_blocks(fname: str | PathLike, max_time_blocks: int) -> Iterator[dict]:
    """read a PETSIRD binary file in chunks of max_time_blocks time blocks

    Every chunk has the format of read_time_blocks() (block_offsets refer to the
    event time blocks of the chunk).
    """
    if max_time_blocks < 1:
        raise ValueError("max_time_blocks has to be >= 1")
    reader = TimeBlockReader(str(fname))
    while not reader.done:
        chunk = reader.read(max_time_blocks)
        if chunk["num_time_blocks"] > 0:
            yield chunk
