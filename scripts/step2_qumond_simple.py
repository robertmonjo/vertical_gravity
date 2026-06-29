"""Step 2 (QUMOND Simple) — 3D Poisson solver, simple interpolation function (~2h).

Reads: outputs/mc100_baryonic_radial.csv, _vertical.csv
Writes: outputs/model_qumond_simple_radial.csv, _vertical.csv

Usage
-----
  python step2_qumond_simple.py
"""
import subprocess, sys
from pathlib import Path

subprocess.run(
    [sys.executable,
     str(Path(__file__).parent / "step2_fit_all_models.py"),
     "--full", "--models", "qumond_simple"],
    check=True,
)
