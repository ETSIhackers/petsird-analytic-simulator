"""visualize the detection efficiencies encoded in a PETSIRD file

Generic for any PETSIRD file (only the header is read). Writes next to the listmode file
(<stem>_efficiencies_<fingerprint>_*):
- crystals.png: detection bin efficiencies at the positions of the detecting elements
  (angle around the scanner axis vs axial position, one panel per depth layer)
- sgid_lut.png: module pair SGID LUTs (module pairs in coincidence, symmetry groups)
- module_pairs.png: module pair efficiency matrix of every SGID (elements sorted by
  layer, angle, axial position)
- fans.png: total efficiency of all LORs of some detection bins
- reference.csv: decoded efficiencies (and all their components) of probe and random
  LORs computed with the official petsird.helpers, to compare with other readers

See efficiency_plots.py and the README for what correct decoding looks like.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import petsird

from efficiency_plots import (
    efficiency_reference_table,
    plot_detection_bin_efficiencies,
    plot_fans,
    plot_module_pair_efficiencies,
    plot_sgid_luts,
    write_reference_table,
)
from petsird_reader import module_type_info
from utils import lm_file_fingerprint

parser = argparse.ArgumentParser(
    description="Visualize the detection efficiencies encoded in a PETSIRD file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("fname", type=str, help="Path to the PETSIRD listmode file")
parser.add_argument("--energy_bin", type=int, default=0, help="energy bin to show")
parser.add_argument(
    "--fan_bins",
    type=int,
    nargs="*",
    default=None,
    help="detection bins (module type 0) whose LOR efficiencies are shown "
    "(default: the first detection bin and one in the middle of the second module)",
)
parser.add_argument("--num_random_lors", type=int, default=500, help="random LORs in the reference table")
parser.add_argument("--seed", type=int, default=0, help="random seed of the reference table")
parser.add_argument("--skip_plots", action="store_true", help="do not show the figures")
args = parser.parse_args()

fname = Path(args.fname)
header = petsird.BinaryPETSIRDReader(str(fname)).read_header()
scanner = header.scanner
prefix = fname.with_name(f"{fname.stem}_efficiencies_{lm_file_fingerprint(fname)}")

num_modules, num_det_els, num_energy_bins = module_type_info(scanner, 0)
fan_bins = args.fan_bins
if fan_bins is None:
    bins_per_module = num_det_els * num_energy_bins
    fan_bins = [0, min(num_modules - 1, 1) * bins_per_module + (num_det_els // 2) * num_energy_bins]

figures = {
    "crystals": plot_detection_bin_efficiencies(scanner, 0, args.energy_bin),
    "sgid_lut": plot_sgid_luts(scanner),
    "module_pairs": plot_module_pair_efficiencies(scanner, (0, 0), (args.energy_bin, args.energy_bin)),
    "fans": plot_fans(scanner, fan_bins, 0, args.energy_bin),
}
for name, fig in figures.items():
    path = Path(f"{prefix}_{name}.png")
    fig.savefig(path, dpi=110)
    print(f"Saved {path}")

rows = efficiency_reference_table(scanner, args.num_random_lors, args.seed)
csv_path = Path(f"{prefix}_reference.csv")
write_reference_table(rows, csv_path)
print(f"Saved {csv_path} ({len(rows)} LORs)")

if not args.skip_plots:
    plt.show()
