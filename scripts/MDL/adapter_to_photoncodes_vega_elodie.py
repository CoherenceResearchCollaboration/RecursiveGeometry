#!/usr/bin/env python3
# scripts/MDL/adapter_to_photoncodes_vega_elodie.py
"""
Adapter: ELODIE Vega spectrum (ASCII: wavelength, flux) -> photons list CSV

Input rows (whitespace separated):
    wavelength   flux

We:
  1) load 2 columns,
  2) smooth flux (moving average),
  3) detect absorption lines as dips (local minima),
  4) convert to frequency (Hz) and log10 nu,
  5) write: wavelength_nm, frequency_hz, log10_nu_hz, flux

Run:
PYTHONPATH=. python -m scripts.MDL.aggregate_vega_elodie_dataset \
  --indir data/stars/vega_elodie \
  --out-csv results/stars/raw_csv/vega_all_photons.csv \
  --unit nm --smooth-win 9 --win-mad 201 \
  --k-sigma-seed 2.5 --k-sigma-final 3.0 \
  --min-sep-px 9 --sigmas-pix 2,3,5,8,12,20,35 \
  --merge-kms 2.0 --one-per-kappa --fallback-simple

Options:
  --unit nm|angstrom      (default: nm)
  --smooth-win N          moving-average window (odd; default 9)
  --min-prom FLOAT        minimum prominence in flux units (default: 0.01)
  --min-sep-px N          minimum index separation (default: 5)
  --max-peaks N           optional cap on number of dips
"""

import argparse
import math
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd

C_M_PER_S = 299_792_458.0


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win < 3 or win % 2 == 0:
        return x
    k = win // 2
    kernel = np.ones(win, dtype=float) / win
    y = np.convolve(x, kernel, mode="same")
    y[:k] = y[k]
    y[-k:] = y[-k-1]
    return y


def find_dips(flux: np.ndarray,
              min_prom: float = 0.01,
              min_sep_px: int = 5,
              max_peaks: Optional[int] = None) -> np.ndarray:
    n = flux.size
    if n < 3:
        return np.array([], dtype=int)

    cand = np.where((flux[1:-1] < flux[:-2]) & (flux[1:-1] < flux[2:]))[0] + 1
    if cand.size == 0:
        return cand

    def local_max_left(i: int) -> float:
        j0 = max(0, i - 50)
        return float(np.max(flux[j0:i])) if i > j0 else flux[i]

    def local_max_right(i: int) -> float:
        j1 = min(n, i + 51)
        return float(np.max(flux[i+1:j1])) if i+1 < j1 else flux[i]

    proms = []
    for i in cand:
        lm = local_max_left(i)
        rm = local_max_right(i)
        proms.append(max(lm, rm) - flux[i])
    proms = np.asarray(proms)

    keep = np.where(proms >= min_prom)[0]
    if keep.size == 0:
        return np.array([], dtype=int)

    cand = cand[keep]
    proms = proms[keep]

    order = np.argsort(proms)[::-1]
    selected = []
    occupied = np.zeros(n, dtype=bool)
    for idx in order:
        i = cand[idx]
        if occupied[max(0, i - min_sep_px):min(n, i + min_sep_px + 1)].any():
            continue
        selected.append(i)
        occupied[max(0, i - min_sep_px):min(n, i + min_sep_px + 1)] = True
        if max_peaks and len(selected) >= max_peaks:
            break

    return np.array(sorted(selected))

def rolling_mad(x, win=401):
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    k = win//2
    pad = np.pad(x, (k,k), mode="edge")
    W = sliding_window_view(pad, win)
    med = np.median(W, axis=1)
    mad = np.median(np.abs(W - med[:,None]), axis=1)
    mad = np.where(mad < 1e-9, 1e-9, mad)
    return med, mad

def find_lines_adaptive(lam_nm, flux, win_smooth=9, win_mad=401,
                        k_sigma=6.0, min_sep_px=7, max_peaks=None):
    fls = moving_average(flux, win_smooth)
    # high-pass residual relative to rolling median
    med, mad = rolling_mad(fls, win=win_mad)
    # candidate minima (strict)
    n = fls.size
    cand = np.where((fls[1:-1] < fls[:-2]) & (fls[1:-1] < fls[2:]))[0] + 1
    if cand.size == 0: return cand
    depth = med[cand] - fls[cand]          # positive for absorption
    keep = depth > (k_sigma * mad[cand])   # adaptive SNR test
    cand = cand[keep]
    if cand.size == 0: return cand
    # min separation by strongest depth first
    order = np.argsort(depth[keep])[::-1]
    sel, occ = [], np.zeros(n, dtype=bool)
    for j in order:
        i = cand[j]
        if occ[max(0,i-min_sep_px):min(n,i+min_sep_px+1)].any(): continue
        sel.append(i)
        occ[max(0,i-min_sep_px):min(n,i+min_sep_px+1)] = True
        if max_peaks and len(sel) >= max_peaks: break
    return np.array(sorted(sel))

def load_elodie(filepath: str, unit: str = "nm") -> Tuple[np.ndarray, np.ndarray]:
    lambdas = []
    fluxes = []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            # must have at least two columns
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                fval = float(parts[1])
            except Exception:
                continue
            lambdas.append(w)
            fluxes.append(fval)

    lam = np.asarray(lambdas, dtype=float)
    flx = np.asarray(fluxes, dtype=float)

    unit = unit.lower()
    if unit in ("angstrom", "ang", "a"):
        lam = lam * 0.1  # Å -> nm

    return lam, flx


def nm_to_frequency_hz(lambda_nm: np.ndarray) -> np.ndarray:
    lam_m = lambda_nm * 1e-9
    return C_M_PER_S / lam_m


def main():
    p = argparse.ArgumentParser(description="Adapter: ELODIE Vega -> photons CSV")
    p.add_argument("--in", dest="infile", required=True, help="Input ELODIE Vega file")
    p.add_argument("--out-csv", dest="out_csv", required=True, help="Output CSV path")
    p.add_argument("--unit", choices=["nm", "angstrom"], default="nm", help="Input wavelength unit")
    p.add_argument("--smooth-win", type=int, default=9, help="Moving-average window (odd)")
    p.add_argument("--min-prom", type=float, default=0.01, help="Minimum dip prominence (flux units)")
    p.add_argument("--win-mad", type=int, default=401, help="Rolling MAD window for weights")
    p.add_argument("--min-sep-px", type=int, default=5, help="Minimum separation between dips (indices)")
    p.add_argument("--max-peaks", type=int, default=None, help="Optional cap on number of dips")
    args = p.parse_args()

    lam_nm, flux = load_elodie(args.infile, unit=args.unit)
    if lam_nm.size == 0:
        print("No numeric rows found.", file=sys.stderr)
        sys.exit(2)

    fl_s = moving_average(flux, args.smooth_win)
    dips_idx = find_dips(fl_s, min_prom=args.min_prom, min_sep_px=args.min_sep_px, max_peaks=args.max_peaks)

    if dips_idx.size == 0:
        step = max(1, lam_nm.size // 5000)
        dips_idx = np.arange(0, lam_nm.size, step, dtype=int)

    lam_sel = lam_nm[dips_idx]
    flx_sel = flux[dips_idx]
    nu_hz = nm_to_frequency_hz(lam_sel)
    log10_nu = np.log10(nu_hz)

    # compute weights locally (safe when running adapter directly)
    fls = moving_average(flux, args.smooth_win)
    med, mad = rolling_mad(fls, win=args.win_mad)
    depth = med[dips_idx] - fls[dips_idx]
    w = depth / np.max(depth) if depth.size else depth
    df = pd.DataFrame({
        "wavelength_nm": lam_sel,
        "frequency_hz": nu_hz,
        "log10_nu_hz": log10_nu,
        "flux": flx_sel,
        "weight": w
    })

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} photons -> {args.out_csv}")


if __name__ == "__main__":
    main()
