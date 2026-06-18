"""Step 0 (optional) — Reproduce Fig. 1: baryonic density + spiral arm tracers.

Reads the pre-computed density grid and spiral arm loci from data/ and
generates the two-panel figure:

  Panel (a) — Face-on surface density with Reid+2019 spiral arm loci.
  Panel (b) — Meridional baryonic density cross-section.

Usage
-----
  python scripts/step0_reproduce_fig1.py
  python scripts/step0_reproduce_fig1.py --dpi 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vgrav.figures import plot_fig1

FIGS = ROOT / "figs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dpi", type=int, default=260, help="Figure DPI (default: 260)")
    args = parser.parse_args()

    FIGS.mkdir(exist_ok=True)
    data_dir = ROOT / "data"
    fig_path = FIGS / "fig1_reproduced.png"

    import matplotlib.pyplot as plt
    fig = plot_fig1(data_dir=data_dir, output_path=fig_path, dpi=args.dpi)
    plt.close(fig)
    print("Step 0 complete.")


if __name__ == "__main__":
    main()
