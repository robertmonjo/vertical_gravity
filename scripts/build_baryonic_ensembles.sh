#!/usr/bin/env bash
# Baryonic ensembles n = 1-4 from scratch: weighted band and 100 realisations.
#
# Ensemble n combines the first n reconstructions (MI, LW, B2, MM). For each n:
#   outputs/nbar<n>/baryon_band_nbar<n>.csv                     rebuild_baryon_band.py
#   outputs/nbar<n>/mc100_baryonic_{radial,vertical}.csv        step1_build_baryonic_mc100.py
#
# Usage:  bash scripts/build_baryonic_ensembles.sh [n ...]     (default: 1 2 3 4)
set -euo pipefail
cd "$(dirname "$0")/.."

NBARS="${*:-1 2 3 4}"
for n in $NBARS; do
  mkdir -p outputs/nbar$n
  python scripts/rebuild_baryon_band.py --nbar $n --sigma-log 0.5 \
    --outfile outputs/nbar$n/baryon_band_nbar$n.csv
done
for n in $NBARS; do
  python scripts/step1_build_baryonic_mc100.py --full --outdir outputs/nbar$n \
    --baryon-band outputs/nbar$n/baryon_band_nbar$n.csv > outputs/nbar$n/build.log 2>&1 &
done
wait
