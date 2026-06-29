"""Run the full pipeline for all nbar variants and print a combined Table 2.

Single entry point to reproduce all baryonic reconstruction variants in one go.
Each variant's outputs go to outputs/nbar{N}/ to avoid overwriting.

Baryonic reconstruction progression (--nbar N activates the first N reconstructions):
  nbar=1  McGaugh/Imig only        (uniform weight=1)
  nbar=2  + Wang/Lian              (kernel-weighted)
  nbar=3  + deSalas B2             (kernel-weighted)
  nbar=4  + McMillan               (kernel-weighted)  [four-source result used in Fig. 2]

Execution modes (--mode):
  sequential   One process at a time: nbar=1 fully completes before nbar=2 starts.
               Minimal RAM/CPU usage. Default. (~40 h for full run)

  parallel     All nbar variants run simultaneously (4 parallel subprocesses).
               rebuild_baryon_band runs sequentially first (pre-generates one
               baryon_band_nbar{N}.csv per variant); then step1, step2, step3
               all run in parallel for all variants.
               Wall-time = max(T_nbar1, T_nbar2, ...) ~ 10 h.  Uses ~4× RAM.

  max          Nbar variants run sequentially (nbar=1 completes before nbar=2),
               but within each variant the slow models (QUMOND×3 + STVG) are
               launched as parallel subprocesses (up to --workers at a time).
               Wall-time = sum(T_nbar1+T_nbar2+...) ~ 20 h (slower than parallel
               because nbar variants are not overlapped, but needs only ~1-2× RAM
               instead of 4×).  Use this when RAM is limited.

Usage
-----
  # Full sequential run (safest, minimal resources):
  python scripts/run_all_nbar.py --full

  # Full parallel run (fastest, 4× resources):
  python scripts/run_all_nbar.py --full --mode parallel

  # Max parallelism within each nbar (balanced):
  python scripts/run_all_nbar.py --full --mode max
  python scripts/run_all_nbar.py --full --mode max --workers 2

  # Quick validation — 5 draws, algebraic QUMOND proxy:
  python scripts/run_all_nbar.py --full --n-draws 5 --approx --no-stvg

  # Only specific baryonic reconstructions:
  python scripts/run_all_nbar.py --full --nbar-list 1,3

  # Table 2 from existing outputs (no recomputation):
  python scripts/run_all_nbar.py --table-only

Output layout
-------------
  outputs/nbar1/   mc100_baryonic_*.csv + model_*.csv + mc100_chi2_all_models.csv
  outputs/nbar2/   idem
  outputs/nbar3/   idem
  outputs/nbar4/   idem
  outputs/table2_all_nbar.txt   Combined Table 2 for all variants

Parallel mode logs
------------------
  Each background subprocess writes stdout+stderr to a log file in its outdir:
    outputs/nbar1/run_step1.log
    outputs/nbar1/run_step2.log  (or run_step2_qumond_simple.log etc. in max mode)
  Monitor progress with:  Get-Content outputs/nbar1/run_step1.log -Wait
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Force UTF-8 output so labels with → × etc. don't crash on Windows cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT     = Path(__file__).resolve().parents[1]
SCRIPTS  = ROOT / "scripts"
OUT_ROOT = ROOT / "outputs"

STEP1   = SCRIPTS / "step1_build_baryonic_mc100.py"
STEP2   = SCRIPTS / "step2_fit_all_models.py"
STEP3   = SCRIPTS / "step3_figures.py"
REBUILD = SCRIPTS / "rebuild_baryon_band.py"

NBAR_ALL = [1, 2, 3, 4]

# Fast algebraic models (< 10 min each)
_FAST_MODELS = (
    "baryonic,veg_fixed,veg_free,hmg_k1,cdm_nfw,cdm_einasto,fr_screened,refracted_gravity"
)
# Slow Poisson/direct-summation models (~2 h each)
_SLOW_MODELS_QUMOND = ["qumond_simple", "qumond_standard", "qumond_mls"]
_SLOW_MODELS_STVG   = ["stvg"]

# Suppress popup console windows on Windows for background subprocesses
_WIN_FLAGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
)


def _outdir(nbar: int) -> Path:
    return OUT_ROOT / f"nbar{nbar}"


def _band_file(nbar: int) -> Path:
    return ROOT / "data" / f"baryon_band_nbar{nbar}.csv"


# ── subprocess helpers ─────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str) -> None:
    """Sequential subprocess — inherits console, no popup window."""
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True, **_WIN_FLAGS)


def _launch(cmd: list[str], log_path: Path) -> tuple[subprocess.Popen, object]:
    """Launch a background subprocess; stdout+stderr → log_path. Returns (proc, fh)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        [sys.executable] + cmd,
        stdout=fh, stderr=subprocess.STDOUT,
        **_WIN_FLAGS,
    )
    return proc, fh


def _wait_all(procs: list[tuple[subprocess.Popen, object, Path, str]]) -> None:
    """Wait for all (proc, fh, log_path, label); close file handles; raise on failure."""
    print(f"\n  Waiting for {len(procs)} parallel process(es)...", flush=True)
    failed = []
    for proc, fh, log_path, label in procs:
        rc = proc.wait()
        try:
            fh.close()  # type: ignore[union-attr]
        except Exception:
            pass
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"    {status}  {label}  [log: {log_path}]", flush=True)
        if rc != 0:
            failed.append((label, log_path, rc))
    if failed:
        for label, log_path, rc in failed:
            print(f"\n  ERROR in '{label}' (rc={rc}). Last 20 lines of log:")
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                for ln in lines[-20:]:
                    print(f"    {ln}")
            except Exception:
                pass
        raise RuntimeError(
            f"{len(failed)} subprocess(es) failed: {[lbl for lbl, _, _ in failed]}"
        )


# ── Mode: sequential ──────────────────────────────────────────────────────────

def _run_one_sequential(
    nbar: int, n_draws: int | None, approx: bool, no_stvg: bool, sigma_log: float = 0.50,
) -> None:
    """Rebuild → step1 → step2 → step3 for a single nbar (all sequential)."""
    outdir = _outdir(nbar)
    outdir.mkdir(parents=True, exist_ok=True)

    _run([str(REBUILD), "--nbar", str(nbar), "--sigma-log", str(sigma_log)],
         f"[nbar={nbar}] rebuild_baryon_band")

    s1 = [str(STEP1), "--full", "--outdir", str(outdir)]
    if n_draws:
        s1 += ["--n-draws", str(n_draws)]
    _run(s1, f"[nbar={nbar}] step1 → {outdir.name}/")


    s2 = [str(STEP2), "--full", "--outdir", str(outdir)]
    if n_draws:
        s2 += ["--n-draws", str(n_draws)]
    if approx:
        s2.append("--approx")
    if no_stvg:
        s2.append("--no-stvg")
    _run(s2, f"[nbar={nbar}] step2 → {outdir.name}/")


    _run([str(STEP3), "--output-dir", str(outdir), "--no-figure"],
         f"[nbar={nbar}] step3 (Table 2)")


def run_sequential(
    nbar_list: list[int], n_draws: int | None, approx: bool, no_stvg: bool, sigma_log: float = 0.50,
) -> None:
    for nbar in nbar_list:
        print(f"\n{'#'*68}")
        print(f"# SEQUENTIAL  nbar={nbar}")
        print(f"{'#'*68}")
        _run_one_sequential(nbar, n_draws, approx, no_stvg, sigma_log)


# ── Mode: parallel ────────────────────────────────────────────────────────────

def run_parallel(
    nbar_list: list[int], n_draws: int | None, approx: bool, no_stvg: bool, sigma_log: float = 0.50,
) -> None:
    """
    Phase 0: generate all baryon_band_nbar{N}.csv sequentially (fast, < 1 min each).
    Phase 1: step1 × all nbar in parallel  (~2 h, uses separate band files).
    Phase 2: step2 × all nbar in parallel  (~8 h for QUMOND+STVG).
    Phase 3: step3 × all nbar sequentially (< 1 min each).
    Background subprocesses write to outputs/nbar{N}/run_step{1,2}.log.
    """
    print(f"\n{'#'*68}")
    print("# PARALLEL — Phase 0: pre-generating baryon_band files")
    print(f"{'#'*68}")
    for nbar in nbar_list:
        out_band = _band_file(nbar)
        _run([str(REBUILD), "--nbar", str(nbar), "--outfile", str(out_band),
              "--sigma-log", str(sigma_log)],
             f"[nbar={nbar}] rebuild → {out_band.name}")

    print(f"\n{'#'*68}")
    print("# PARALLEL — Phase 1: step1 (baryonic Poisson) × all nbar")
    print(f"{'#'*68}")
    procs1: list[tuple[subprocess.Popen, object, Path, str]] = []
    for nbar in nbar_list:
        outdir = _outdir(nbar)
        outdir.mkdir(parents=True, exist_ok=True)
        s1 = [str(STEP1), "--full", "--outdir", str(outdir),
              "--baryon-band", str(_band_file(nbar))]
        if n_draws:
            s1 += ["--n-draws", str(n_draws)]
        log = outdir / "run_step1.log"
        proc, fh = _launch(s1, log)
        procs1.append((proc, fh, log, f"nbar={nbar} step1"))
        print(f"  Launched nbar={nbar} step1 → {log}")
    _wait_all(procs1)

    print(f"\n{'#'*68}")
    print("# PARALLEL — Phase 2: step2 (all models) × all nbar")
    print(f"{'#'*68}")
    procs2: list[tuple[subprocess.Popen, object, Path, str]] = []
    for nbar in nbar_list:
        outdir = _outdir(nbar)
        s2 = [str(STEP2), "--full", "--outdir", str(outdir),
              "--baryon-band", str(_band_file(nbar))]
        if n_draws:
            s2 += ["--n-draws", str(n_draws)]
        if approx:
            s2.append("--approx")
        if no_stvg:
            s2.append("--no-stvg")
        log = outdir / "run_step2.log"
        proc, fh = _launch(s2, log)
        procs2.append((proc, fh, log, f"nbar={nbar} step2"))
        print(f"  Launched nbar={nbar} step2 → {log}")
    _wait_all(procs2)

    print(f"\n{'#'*68}")
    print("# PARALLEL — Phase 3: step3 (Table 2) × all nbar")
    print(f"{'#'*68}")
    for nbar in nbar_list:
        outdir = _outdir(nbar)
        _run([str(STEP3), "--output-dir", str(outdir), "--no-figure"],
             f"[nbar={nbar}] step3")


# ── Mode: max (resource-aware) ────────────────────────────────────────────────

def run_max(
    nbar_list: list[int], n_draws: int | None, approx: bool, no_stvg: bool,
    workers: int, sigma_log: float = 0.50,
) -> None:
    """
    Nbar variants run sequentially.  Within each variant:
      1. rebuild → data/baryon_band.csv
      2. step1 (sequential, ~2 h)
      3. step2 fast models: all algebraic models in one subprocess (< 10 min)
      4. step2 slow models: QUMOND×3 + STVG as parallel subprocesses (up to
         --workers at a time).  Each writes its own model_{key}_*.csv; no conflict.
      5. step3 (fast, sequential)
    Slow model logs: outputs/nbar{N}/run_step2_{model}.log
    """
    # Determine which slow models to run
    slow_models: list[str] = [] if approx else list(_SLOW_MODELS_QUMOND)
    if not no_stvg:
        slow_models.append("stvg")

    # In approx mode, QUMOND proxy is included in fast; only STVG may be slow
    fast_models = _FAST_MODELS
    if approx:
        fast_models += ",qumond_simple,qumond_standard,qumond_mls"

    for nbar in nbar_list:
        print(f"\n{'#'*68}")
        print(f"# MAX  nbar={nbar}  (workers={workers})")
        print(f"{'#'*68}")
        outdir = _outdir(nbar)
        outdir.mkdir(parents=True, exist_ok=True)

        # 1. rebuild → data/baryon_band.csv (sequential, fast)
        _run([str(REBUILD), "--nbar", str(nbar), "--sigma-log", str(sigma_log)],
             f"[nbar={nbar}] rebuild_baryon_band")

        # 2. step1 (sequential, ~2 h)
        s1 = [str(STEP1), "--full", "--outdir", str(outdir)]
        if n_draws:
            s1 += ["--n-draws", str(n_draws)]
        _run(s1, f"[nbar={nbar}] step1")

        # 3. step2 fast models (< 10 min, sequential)
        s2_fast = [str(STEP2), "--full", "--outdir", str(outdir),
                   "--models", fast_models]
        if n_draws:
            s2_fast += ["--n-draws", str(n_draws)]
        if approx:
            s2_fast.append("--approx")
        _run(s2_fast, f"[nbar={nbar}] step2 fast models ({fast_models[:40]}...)")

        # 4. step2 slow models in parallel (throttled to `workers`)
        if slow_models:
            print(f"\n  Launching {len(slow_models)} slow model(s) "
                  f"(max {workers} parallel): {slow_models}", flush=True)
            queue = list(slow_models)
            running: list[tuple[subprocess.Popen, object, Path, str]] = []

            while queue or running:
                # Fill up to `workers` slots
                while queue and len(running) < workers:
                    model = queue.pop(0)
                    s2_slow = [str(STEP2), "--full", "--outdir", str(outdir),
                               "--models", model]
                    if n_draws:
                        s2_slow += ["--n-draws", str(n_draws)]
                    log = outdir / f"run_step2_{model}.log"
                    proc, fh = _launch(s2_slow, log)
                    running.append((proc, fh, log, f"nbar={nbar}/{model}"))
                    print(f"  Launched {model} → {log}", flush=True)

                # Poll for completions
                time.sleep(10)
                still_running: list[tuple[subprocess.Popen, object, Path, str]] = []
                for entry in running:
                    proc, fh, log, label = entry
                    rc = proc.poll()
                    if rc is None:
                        still_running.append(entry)
                    else:
                        try:
                            fh.close()  # type: ignore[union-attr]
                        except Exception:
                            pass
                        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                        print(f"  {status}  {label}", flush=True)
                        if rc != 0:
                            # Kill remaining before raising
                            for p, f2, _, _ in still_running:
                                try:
                                    p.terminate()
                                    f2.close()  # type: ignore[union-attr]
                                except Exception:
                                    pass
                            raise RuntimeError(
                                f"Subprocess failed: {label} (rc={rc}). "
                                f"See {log}"
                            )
                running = still_running

        # 5. step3 (fast)
        _run([str(STEP3), "--output-dir", str(outdir), "--no-figure"],
             f"[nbar={nbar}] step3 (Table 2)")


# ── Combined Table 2 ──────────────────────────────────────────────────────────

def print_combined_table(nbar_list: list[int]) -> None:
    """Read mc100_chi2_all_models.csv for each nbar and print a side-by-side Table 2."""
    import csv

    data: dict[int, dict[str, list[float]]] = {}
    model_order: list[str] = []
    model_names: dict[str, str] = {}
    model_k: dict[str, int] = {}

    for nbar in nbar_list:
        chi2_csv = _outdir(nbar) / "mc100_chi2_all_models.csv"
        if not chi2_csv.exists():
            print(f"  [nbar={nbar}] {chi2_csv} not found — skipped.")
            continue
        data[nbar] = {}
        with open(chi2_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = row["model_key"]
                if key not in model_order:
                    model_order.append(key)
                    model_names[key] = row.get("model_name", key)
                    model_k[key]     = int(row.get("k", 0))
                data[nbar].setdefault(key, []).append(float(row["chi2_nu"]))

    if not data:
        print("No mc100_chi2_all_models.csv found. Run the pipeline first.")
        return

    nbar_present = [n for n in nbar_list if n in data]
    col_w = 24
    hdr_cols = "".join(f"{'nbar='+str(n):>{col_w}}" for n in nbar_present)

    lines = []
    sep = "=" * (38 + col_w * len(nbar_present))
    lines.append(sep)
    lines.append(f"  {'Model':<32} {'k':>3} " + hdr_cols)
    lines.append(
        f"  {'':<32} {'':>3} "
        + "".join(f"{'p16 / p50 / p84':>{col_w}}" for _ in nbar_present)
    )
    lines.append("-" * (38 + col_w * len(nbar_present)))
    for key in model_order:
        cells = []
        for nbar in nbar_present:
            if key not in data.get(nbar, {}):
                cells.append(f"{'—':>{col_w}}")
            else:
                arr = np.array(data[nbar][key])
                p16, p50, p84 = np.percentile(arr, [16, 50, 84])
                cells.append(f"{p16:6.3f}/{p50:6.3f}/{p84:6.3f}")
        lines.append(f"  {model_names[key]:<32} {model_k[key]:>3} " + "".join(cells))
    lines.append(sep)

    table_txt = "\n".join(lines)
    print("\n" + table_txt)

    out_txt = OUT_ROOT / "table2_all_nbar.txt"
    OUT_ROOT.mkdir(exist_ok=True)
    out_txt.write_text(table_txt + "\n", encoding="utf-8")
    print(f"\nCombined Table 2 written to: {out_txt}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    auto_workers = max(1, min(4, (os.cpu_count() or 4) // 2))

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--nbar-list", default="1,2,3,4", metavar="N,...",
        help="Comma-separated nbar values to run (default: 1,2,3,4).",
    )
    parser.add_argument(
        "--mode", default="sequential",
        choices=["sequential", "parallel", "max"],
        help=(
            "Execution mode: sequential (1 process at a time, default) | "
            "parallel (all nbar simultaneously, ~4× RAM/CPU) | "
            "max (QUMOND+STVG in parallel within each nbar, auto resource-aware)."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help=(
            f"Max simultaneous slow-model subprocesses for --mode max "
            f"(default: auto={auto_workers} from os.cpu_count()//2, capped at 4)."
        ),
    )
    parser.add_argument("--full", action="store_true",
                        help="Run full pipeline (step1+step2+step3) for each variant.")
    parser.add_argument("--n-draws", type=int, default=None, metavar="N",
                        help="Limit to first N draws (for quick validation).")
    parser.add_argument("--approx", action="store_true",
                        help="Use algebraic QUMOND proxy instead of 3D Poisson (fast).")
    parser.add_argument("--no-stvg", action="store_true",
                        help="Skip STVG (~2 h per variant).")
    parser.add_argument("--table-only", action="store_true",
                        help="Skip pipeline; print combined Table 2 from existing outputs.")
    parser.add_argument(
        "--sigma-log", type=float, default=0.50, metavar="SIGMA",
        help="Log-radius kernel width for baryonic weighting (default 0.50). "
             "Use 0 or inf for flat global weights.",
    )
    args = parser.parse_args()

    nbar_list = [int(x.strip()) for x in args.nbar_list.split(",")]
    for n in nbar_list:
        if n not in NBAR_ALL:
            parser.error(f"--nbar-list contains invalid value {n}. Allowed: 1,2,3,4.")

    workers = args.workers if args.workers is not None else auto_workers

    if not args.table_only:
        if not args.full and args.n_draws is None:
            parser.error(
                "Specify --full or --n-draws N to run the pipeline, "
                "or --table-only to print combined Table 2."
            )

        if args.mode == "sequential":
            run_sequential(nbar_list, args.n_draws, args.approx, args.no_stvg, args.sigma_log)

        elif args.mode == "parallel":
            run_parallel(nbar_list, args.n_draws, args.approx, args.no_stvg, args.sigma_log)

        elif args.mode == "max":
            print(f"\n  Resource-aware mode: up to {workers} parallel slow-model workers "
                  f"(cpu_count={os.cpu_count()}, auto={auto_workers})")
            run_max(nbar_list, args.n_draws, args.approx, args.no_stvg, workers, args.sigma_log)

    print_combined_table(nbar_list)


if __name__ == "__main__":
    main()
