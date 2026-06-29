"""Step 2 (STVG) — scalar-tensor-vector gravity, direct summation (~2h).

Reads: outputs/mc100_baryonic_radial.csv, _vertical.csv
Writes: outputs/model_stvg_radial.csv, _vertical.csv

Usage
-----
  python step2_stvg.py
"""
import subprocess, sys
from pathlib import Path

subprocess.run(
    [sys.executable,
     str(Path(__file__).parent / "step2_fit_all_models.py"),
     "--full", "--models", "stvg"],
    check=True,
)
