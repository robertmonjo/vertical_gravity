#!/usr/bin/env bash
# Full-likelihood model comparison (BIC) in two frameworks.
#
#   (a) One maximum-likelihood fit per baryonic reconstruction (MI, LW, B2, MM),
#       thin- and thick-disc factors free with Gaussian priors on the local
#       stellar densities, 13 gravity models, 5 radial data vectors.
#         -> outputs/likelihood_bic/likelihood_bic_fits.csv
#   (b) The band-consistent MC100 ensembles (4 cumulative baryonic ensembles x
#       100 draws) refitted with the same likelihood, 12 models (no STVG),
#       5 radial data vectors.
#         -> outputs/likelihood_bic_mc100/<variant>/bc_nbar<n>/
#
# Requires the MC100 baryonic draws in outputs/nbar{1..4}/ (step1_build_baryonic_mc100.py).
# Runtime on 32 cores: (a) ~5 min, STVG ~20 min, (b) ~10 min.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

WORKERS=${WORKERS:-24}
VARIANTS="all no_eilers no_wang overlap_correlated wang_only"
OUT_B=outputs/likelihood_bic_mc100

mkdir -p outputs/likelihood_bic "$OUT_B"
rm -f outputs/likelihood_bic/likelihood_bic_fits.csv
python scripts/likelihood_bic.py --workers "$WORKERS" > outputs/likelihood_bic/run.log 2>&1
python scripts/likelihood_bic_stvg.py --workers "$WORKERS" > outputs/likelihood_bic/stvg_run.log 2>&1

for v in $VARIANTS; do for n in 1 2 3 4; do echo "$v $n"; done; done |
  xargs -P "$WORKERS" -n 2 sh -c '
    v=$0; n=$1
    python scripts/step2_bc.py --full --no-stvg \
      --baryon-band outputs/nbar$n/baryon_band_nbar$n.csv \
      --baryonic-dir outputs/nbar$n --outdir '"$OUT_B"'/$v/bc_nbar$n \
      --full-likelihood --radial-variant "$v" > '"$OUT_B"'/${v}_bc_nbar$n.log 2>&1'

for v in $VARIANTS; do
  python scripts/likelihood_bic_mc100_summary.py --write-csv "$OUT_B/$v"/bc_nbar{1,2,3,4} > "$OUT_B/summary_$v.txt"
done
python scripts/likelihood_bic.py --summary > outputs/likelihood_bic/summary.txt
