"""Model comparison marginalised over the baryonic model.

The baryonic distribution is a nuisance: for each gravity model M the evidence is
averaged over the baryonic realisations with equal prior weight,

    Z_M = (1/N) sum_i exp(-BIC_{M,i} / 2),    BIC_M^marg = -2 ln Z_M,

with BIC_{M,i} the BIC of M for realisation i (Laplace approximation to the evidence
at fixed baryons). Delta BIC^marg = BIC_M^marg - min_M BIC_M^marg then compares the
models without fixing the baryons. Realisations are
  framework (a): the four reconstructions MI, LW, B2, MM (likelihood_bic_fits.csv);
  framework (b): the 100 draws of each MC100 ensemble (bic_per_draw.csv).
Only the models listed in the tables enter. In (a) the effective number of
reconstructions that carry the evidence, exp(H) with H the entropy of the weights
exp(-BIC_{M,i}/2) / sum_i, is also given.

Usage
-----
  python scripts/likelihood_bic_marginal.py [--variant all]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["mcgaugh_imig", "lian_wang", "de_salas", "mcmillan"]
MODELS_A = ["baryonic", "qumond_simple", "qumond_standard", "qumond_mls", "veg_verlinde", "veg_verlinde_free",
            "fr_bounded", "reg_dms", "reg_free", "stvg_std", "cdm_nfw", "cdm_einasto", "hmg_sp", "hmg_sk2",
            "e2d_k2"]
MODELS_B = ["baryonic", "mond_simple", "mond_standard", "mond_rar", "veg_verlinde", "veg_verlinde_free",
            "fr_bounded", "reg_dms", "stvg_std", "cdm_nfw", "cdm_einasto", "hmg_k1", "hmg_sk2", "e2d_k2"]


def marginal(bics: np.ndarray) -> tuple[float, float]:
    """-2 ln of the mean of exp(-BIC/2) and the effective number of realisations."""
    b0 = float(np.min(bics))
    w = np.exp(-0.5 * (bics - b0))
    p = w / w.sum()
    n_eff = math.exp(-float(np.sum(p[p > 0] * np.log(p[p > 0]))))
    return b0 - 2.0 * math.log(w.mean()), n_eff


def framework_a(variant):
    rows = [r for r in csv.DictReader(open(ROOT / "outputs/likelihood_bic/likelihood_bic_fits.csv"))
            if r["variant"] == variant]
    out = {}
    for m in MODELS_A:
        found = {r["config"]: float(r["BIC"]) for r in rows if r["model"] == m}
        if all(c in found for c in CONFIGS):
            out[m] = marginal(np.array([found[c] for c in CONFIGS]))
    best = min(v[0] for v in out.values())
    print(f"% framework (a), variant {variant}: Delta BIC^marg over MI, LW, B2, MM (effective number)")
    for m, (b, n) in sorted(out.items(), key=lambda kv: kv[1][0]):
        print(f"{m:<20} {b - best:9.1f}   ({n:.2f})")


def framework_b(variant):
    print(f"% framework (b), variant {variant}: Delta BIC^marg over the 100 draws of each ensemble")
    res = {}
    for n in (1, 2, 3, 4):
        rows = list(csv.DictReader(open(ROOT / f"outputs/likelihood_bic_mc100/{variant}/bc_nbar{n}/bic_per_draw.csv")))
        marg = {m: marginal(np.array([float(r[m]) for r in rows]))[0] for m in MODELS_B}
        best = min(marg.values())
        res[n] = {m: v - best for m, v in marg.items()}
    print(f"{'model':<20}" + "".join(f"{'nbar' + str(n):>10}" for n in res))
    for m in sorted(MODELS_B, key=lambda m: res[4][m]):
        print(f"{m:<20}" + "".join(f"{res[n][m]:10.1f}" for n in res))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="all")
    args = parser.parse_args()
    framework_a(args.variant)
    print()
    framework_b(args.variant)


if __name__ == "__main__":
    main()
