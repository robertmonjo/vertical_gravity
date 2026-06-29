"""Step 2 (fast models) — fit VEG, HMG, CDM, f(R), ReG to all 100 draws.

Runs all algebraic/optimization models in a single session (~30 min):
  baryonic, veg_fixed, veg_free, hmg_k1, fr_screened, refracted_gravity,
  cdm_nfw, cdm_einasto

QUMOND (3D Poisson, ~2h each) and STVG (~2h) are run separately:
  python step2_qumond_simple.py
  python step2_qumond_standard.py
  python step2_qumond_mls.py
  python step2_stvg.py

Usage
-----
  python step2_fast.py          # verify pre-computed CSVs
  python step2_fast.py --full   # regenerate from scratch
"""
import subprocess, sys
from pathlib import Path

FAST_MODELS = (
    "baryonic,veg_fixed,veg_free,hmg_k1,"
    "fr_screened,refracted_gravity,cdm_nfw,cdm_einasto"
)

step2 = Path(__file__).parent / "step2_fit_all_models.py"

if "--full" in sys.argv or "--n-draws" in sys.argv:
    subprocess.run(
        [sys.executable, str(step2), "--full", "--models", FAST_MODELS]
        + [a for a in sys.argv[1:] if a.startswith("--n-draws")],
        check=True,
    )
else:
    subprocess.run([sys.executable, str(step2)], check=True)
