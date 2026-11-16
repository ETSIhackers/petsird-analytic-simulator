"""benchmark: reading the time blocks of a PETSIRD file with yardl Python vs petsird_fast

1. raw events: yardl-generated Python reader (one Python object per event) vs
   petsird_fast (C++, flat arrays)
2. end to end: python/petsird_reader.read_listmode_events() (LOR coordinates, TOF bins,
   efficiencies, ...) with the yardl Python backend vs the petsird_fast backend

Usage (from the repository root):
    python petsird_fast/scripts/benchmark.py FILE.bin
"""

import sys
import time
from pathlib import Path

import numpy as np
import petsird

import petsird_fast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from petsird_reader import get_all_detector_centers, read_listmode_events  # noqa: E402

fname = sys.argv[1]


def timed(label, func, num_events=None):
    t0 = time.perf_counter()
    result = func()
    dt = time.perf_counter() - t0
    per_event = f", {1e9 * dt / num_events:7.1f} ns/event" if num_events else ""
    print(f"{label:58s} {dt:8.3f} s{per_event}")
    return result, dt


def yardl_raw():
    reader = petsird.BinaryPETSIRDReader(fname)
    reader.read_header()
    raw = {"prompts": [], "delayeds": []}
    for tb in reader.read_time_blocks():
        if isinstance(tb, petsird.TimeBlock.EventTimeBlock):
            for stream, nested in (("prompts", tb.value.prompt_events), ("delayeds", tb.value.delayed_events)):
                if len(nested) > 0:
                    raw[stream].append(
                        np.array([(e.detection_bins[0], e.detection_bins[1], e.tof_idx) for e in nested[0][0]], dtype=np.uint32)
                    )
    return {k: np.concatenate(v) for k, v in raw.items()}


header = petsird.BinaryPETSIRDReader(fname).read_header()
fast, t_fast = timed("petsird_fast.read_time_blocks (C++)", lambda: petsird_fast.read_time_blocks(fname))
n = sum(fast[s][(0, 0)]["det_bin0"].size for s in ("prompts", "delayeds"))
print(f"{n} prompts + delayeds, {fast['num_time_blocks']} time blocks")
print(f"(timing of petsird_fast: {1e9 * t_fast / n:.1f} ns/event)")
raw, t_raw = timed("yardl Python: events -> (n, 3) arrays", yardl_raw, n)
for s in ("prompts", "delayeds"):
    a = fast[s][(0, 0)]
    assert np.array_equal(raw[s], np.stack([a["det_bin0"], a["det_bin1"], a["tof_idx"]], 1)), s
print(f"identical events, speed-up raw reading: {t_raw / t_fast:.0f}x")

# end to end: LOR end points, TOF bins, efficiencies, energy bins, detection bin ids
centers = get_all_detector_centers(header.scanner.scanner_geometry)


def end_to_end_yardl():
    reader = petsird.BinaryPETSIRDReader(fname)
    h = reader.read_header()
    return read_listmode_events(reader, h, centers)


def end_to_end_fast():
    # read_listmode_events uses petsird_fast if the file name is given
    reader = petsird.BinaryPETSIRDReader(fname)
    h = reader.read_header()
    return read_listmode_events(reader, h, centers, fname=fname)


ref, t_ref = timed("end to end: read_listmode_events (yardl Python)", end_to_end_yardl, n)
new, t_new = timed("end to end: read_listmode_events (petsird_fast)", end_to_end_fast, n)
for s in ("prompts", "delayeds"):
    for k, v in new[s].items():
        assert np.array_equal(v, ref[s][k]), (s, k)
print(f"identical converted events, end-to-end speed-up: {t_ref / t_new:.0f}x")
