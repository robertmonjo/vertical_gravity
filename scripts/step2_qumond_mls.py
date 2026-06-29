"""Step 2 (QUMOND MLS/RAR) — 3D Poisson solver, McGaugh-Lelli-Schombert interpolation (~2h).

Reads: outputs/mc100_baryonic_radial.csv, _vertical.csv
Writes: outputs/model_qumond_mls_radial.csv, _vertical.csv

Usage
-----
  python step2_qumond_mls.py
"""
import subprocess, sys
from pathlib import Path

subprocess.run(
    [sys.executable,
     str(Path(__file__).parent / "step2_fit_all_models.py"),
     "--full", "--models", "qumond_mls"],
    check=True,
)
