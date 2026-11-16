# petsird_fast (prototype)

Fast reading of the time blocks of PETSIRD binary files into flat NumPy arrays, using
the yardl-generated C++ code of PETSIRD (`libpetsird` from conda-forge) through a small
[nanobind](https://github.com/wjakob/nanobind) extension.

**Why:** the yardl-generated Python code creates one Python object per event and needs
about 2 µs per event. The yardl-generated C++ code needs about 30 ns per event.
This prototype keeps the yardl-generated Python package for the (small) header and reads
all time blocks in C++. Python only receives one contiguous array per field
(struct of arrays), never individual events or time blocks.

Status: prototype, kept separate from the simulator / reconstruction code of this
repository. Only the binary format is supported (not NDJSON / HDF5). Supported
platforms are those of `libpetsird` on conda-forge (currently linux-64, osx-arm64).

## Build

```bash
conda env create -f petsird_fast/environment.yml
conda activate petsird-fast
python -m pip install --no-build-isolation ./petsird_fast
```

The `libpetsird` 0.13.2 headers require C++20 (through xtensor), which the CMake
configuration sets.

## Usage

```python
import petsird
import petsird_fast

header = petsird.BinaryPETSIRDReader(fname).read_header()  # yardl Python
data = petsird_fast.read_time_blocks(fname)                 # C++ -> flat arrays

prompts = data["prompts"][(0, 0)]   # module type pair (type_1, type_2), type_2 <= type_1
prompts["det_bin0"], prompts["det_bin1"], prompts["tof_idx"]   # uint32[n]
prompts["block_offsets"]            # int64[K + 1]: events of event time block k
data["delayeds"][(0, 0)]            # same for the delayed coincidences
data["event_time_blocks"]           # {"start", "stop"}: uint32[K] (ms)
data["singles"]                     # {"counts": [uint64[Ks, num_bins] per module type], "start", "stop"}
data["other"]                       # counts of everything that is not converted to arrays

# large files: chunks of time blocks
for chunk in petsird_fast.iter_time_blocks(fname, max_time_blocks=100):
    ...
```

All other time block types (dead time, external signals, bed / gantry movement with
identity or non-identity transforms, weighted events, singles per singles unit) and
single / triple / quadruple events are only counted in `data["other"]`. A reader can
then raise errors or warnings for them, as `python/petsird_reader.py` does.

## Tests and benchmark

From the repository root (so that the installed package, not the source folder, is
imported):

```bash
python -m pytest petsird_fast/tests
python petsird_fast/benchmark.py path/to/file.bin
```

The tests compare all arrays with the events, offsets, time intervals and singles read
with the yardl-generated Python code. They cover 1 and 2 module types, module type
pairs and time blocks without events, missing delayeds in some time blocks, chunked
reading, other time block types and errors.

## Design notes

- **Header in Python, time blocks in C++.** The header is small (≈70 ms in Python, 1 ms in
  C++), and binding the full PETSIRD type hierarchy would be the main maintenance
  burden. The binding only depends on the time block types.
- **Struct of arrays with fixed dtypes.** The arrays are created in C++ and handed over
  without copying (a capsule owns the `std::vector`). The GIL is released while reading.
- **Later: GPU arrays.** Since all outputs are contiguous arrays of fixed dtypes,
  nanobind's `ndarray` (DLPack) can later return them for other frameworks
  (cupy, pytorch), or fill buffers (e.g. pinned or device memory) provided by the caller,
  without changing the structure of the result.
- **Later: writing.** A bulk writer (arrays in, time blocks out) would speed up the
  simulator in the same way.
- **Packaging.** On the long term a conda-forge package, possibly as an additional
  output of the PETSIRD feedstock(s), built against the same `libpetsird` version.
