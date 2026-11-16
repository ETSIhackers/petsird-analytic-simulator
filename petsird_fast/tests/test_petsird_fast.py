"""petsird_fast returns exactly the time blocks read by the yardl-generated Python code"""

import copy
import sys
from pathlib import Path

import numpy as np
import petsird
import pytest

import petsird_fast

# PETSIRD headers of the simulator (python/ of this repository)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from efficiencies import generate_efficiency_model  # noqa: E402
from scanner import ScannerConfig  # noqa: E402
from simulation import create_petsird_header, tof_parameters  # noqa: E402


def make_header(num_module_types: int = 1) -> petsird.Header:
    """small simulated header, optionally with replicated module types (copies)"""
    config = ScannerConfig(
        num_modules=8, radius=60.0, block_shape=(3, 2, 2), block_spacing=(4.0, 6.0, 4.0),
        min_coincidence_distance=2,
    )
    np.random.seed(1)
    header = create_petsird_header(
        config,
        generate_efficiency_model(config),
        tof_parameters(20.0, (16, 16, 4), (4.0, 4.0, 4.0)),
        20.0,
        store_delayeds=True,
        store_singles=True,
    )
    geom = header.scanner.scanner_geometry
    geom.replicated_modules = [copy.deepcopy(geom.replicated_modules[0]) for _ in range(num_module_types)]
    return header


def random_events(rng, n: int) -> list[petsird.CoincidenceEvent]:
    return [
        petsird.CoincidenceEvent(detection_bins=[int(a), int(b)], tof_idx=int(t))
        for a, b, t in zip(rng.integers(0, 96, n), rng.integers(0, 96, n), rng.integers(0, 5, n))
    ]


def lower_triangular(num_types: int, make) -> list[list]:
    return [[make(m1, m2) for m2 in range(m1 + 1)] for m1 in range(num_types)]


def write_file(fname, header, time_blocks):
    with petsird.BinaryPETSIRDWriter(str(fname)) as writer:
        writer.write_header(header)
        writer.write_time_blocks(time_blocks)


def reference(fname) -> dict:
    """events / singles read with the yardl-generated Python code"""
    reader = petsird.BinaryPETSIRDReader(str(fname))
    header = reader.read_header()
    num_types = header.scanner.scanner_geometry.number_of_module_types()
    ref = {s: {(m1, m2): [] for m1 in range(num_types) for m2 in range(m1 + 1)} for s in ("prompts", "delayeds")}
    offsets = {s: {k: [0] for k in ref[s]} for s in ref}
    singles, intervals = [], []
    for tb in reader.read_time_blocks():
        if isinstance(tb, petsird.TimeBlock.EventTimeBlock):
            intervals.append((tb.value.time_interval.start, tb.value.time_interval.stop))
            for s, nested in (("prompts", tb.value.prompt_events), ("delayeds", tb.value.delayed_events)):
                for (m1, m2), events in ref[s].items():
                    if len(nested) > 0:
                        events += [(e.detection_bins[0], e.detection_bins[1], e.tof_idx) for e in nested[m1][m2]]
                    offsets[s][(m1, m2)].append(len(events))
        elif isinstance(tb, petsird.TimeBlock.SinglesHistogramTimeBlock):
            singles.append(tb.value.singles_per_detection_bin_histogram)
    return {"events": ref, "offsets": offsets, "singles": singles, "intervals": intervals}


def assert_equal_to_reference(result: dict, ref: dict):
    for s in ("prompts", "delayeds"):
        assert set(result[s]) == set(ref["events"][s])
        for pair, events in ref["events"][s].items():
            a = result[s][pair]
            expected = np.array(events, dtype=np.uint32).reshape(-1, 3)
            np.testing.assert_array_equal(np.stack([a["det_bin0"], a["det_bin1"], a["tof_idx"]], 1), expected)
            np.testing.assert_array_equal(a["block_offsets"], ref["offsets"][s][pair])
            for name in ("det_bin0", "det_bin1", "tof_idx"):
                assert a[name].dtype == np.uint32 and a[name].flags.c_contiguous
    np.testing.assert_array_equal(
        np.stack([result["event_time_blocks"]["start"], result["event_time_blocks"]["stop"]], 1),
        np.array(ref["intervals"]).reshape(-1, 2),
    )
    for m, counts in enumerate(result["singles"]["counts"]):
        np.testing.assert_array_equal(counts, np.array([h[m] for h in ref["singles"]], dtype=np.uint64).reshape(counts.shape))


@pytest.mark.parametrize("num_module_types", [1, 2])
def test_events_and_singles_match_yardl_python(tmp_path, num_module_types):
    rng = np.random.default_rng(0)
    header = make_header(num_module_types)
    blocks = []
    for k in range(4):
        interval = petsird.TimeInterval(start=10 * k, stop=10 * (k + 1))
        blocks.append(
            petsird.TimeBlock.EventTimeBlock(
                petsird.EventTimeBlock(
                    time_interval=interval,
                    # some module type pairs without events in some time blocks
                    prompt_events=lower_triangular(num_module_types, lambda m1, m2: random_events(rng, int(rng.integers(0, 50)))),
                    # delayeds missing in time block 2
                    delayed_events=[] if k == 2 else lower_triangular(num_module_types, lambda m1, m2: random_events(rng, 7)),
                )
            )
        )
        blocks.append(
            petsird.TimeBlock.SinglesHistogramTimeBlock(
                petsird.SinglesHistogramTimeBlock(
                    time_interval=interval,
                    singles_per_detection_bin_histogram=[
                        rng.integers(0, 2**40, 96).tolist() for _ in range(num_module_types)
                    ],
                )
            )
        )
    fname = tmp_path / "test.bin"
    write_file(fname, header, blocks)

    result = petsird_fast.read_time_blocks(fname)
    assert result["num_time_blocks"] == 8
    assert result["other"] == {}
    assert_equal_to_reference(result, reference(fname))


def test_chunks_concatenate_to_full_read(tmp_path):
    rng = np.random.default_rng(1)
    blocks = [
        petsird.TimeBlock.EventTimeBlock(
            petsird.EventTimeBlock(
                time_interval=petsird.TimeInterval(start=k, stop=k + 1),
                prompt_events=[[random_events(rng, 20 + k)]],
            )
        )
        for k in range(7)
    ]
    fname = tmp_path / "chunks.bin"
    write_file(fname, make_header(), blocks)

    full = petsird_fast.read_time_blocks(fname)
    chunks = list(petsird_fast.iter_time_blocks(fname, 3))
    assert [c["num_time_blocks"] for c in chunks] == [3, 3, 1]
    for name in ("det_bin0", "det_bin1", "tof_idx"):
        np.testing.assert_array_equal(
            np.concatenate([c["prompts"][(0, 0)][name] for c in chunks]), full["prompts"][(0, 0)][name]
        )
    # exactly divisible: no empty last chunk
    assert [c["num_time_blocks"] for c in petsird_fast.iter_time_blocks(fname, 7)] == [7]


def test_other_time_blocks_are_counted(tmp_path):
    rng = np.random.default_rng(2)
    eye = np.eye(4, dtype="float32")[:3]
    shifted = eye.copy()
    shifted[0, 3] = 5
    interval = petsird.TimeInterval(start=0, stop=1)
    blocks = [
        petsird.TimeBlock.EventTimeBlock(
            petsird.EventTimeBlock(time_interval=interval, prompt_events=[[random_events(rng, 5)]])
        ),
        petsird.TimeBlock.BedMovementTimeBlock(
            petsird.BedMovementTimeBlock(time_interval=interval, transform=petsird.RigidTransformation(matrix=eye))
        ),
        petsird.TimeBlock.BedMovementTimeBlock(
            petsird.BedMovementTimeBlock(time_interval=interval, transform=petsird.RigidTransformation(matrix=shifted))
        ),
        petsird.TimeBlock.ExternalSignalTimeBlock(
            petsird.ExternalSignalTimeBlock(time_interval=interval, signal_id=1, signal_values=[1.0])
        ),
    ]
    fname = tmp_path / "other.bin"
    write_file(fname, make_header(), blocks)
    result = petsird_fast.read_time_blocks(fname)
    assert result["other"] == {
        "identity_bed_movement_time_blocks": 1,
        "bed_movement_time_blocks": 1,
        "external_signal_time_blocks": 1,
    }
    assert result["prompts"][(0, 0)]["det_bin0"].size == 5
    assert result["singles"]["counts"][0].shape == (0, 0)


def test_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        petsird_fast.read_time_blocks(tmp_path / "does_not_exist.bin")
