#!/usr/bin/env python3
"""
Merge all ELODIE Vega files -> one photon *line list* CSV (dips only).

Run:

python -m scripts.MDL.aggregate_vega_elodie_dataset \
  --indir data/stars/vega_elodie \
  --out-csv results/stars/raw_csv/vega_all_photons.csv \
  --unit nm \
  --smooth-win 7 \
  --win-mad 201 \
  --k-sigma-seed 2.5 --k-sigma-final 3.0 \
  --min-sep-px 9 \
  --sigmas-pix 2,3,5,8,12,20,35 \
  --merge-kms 2.0 \
  --one-per-kappa \
  --debug \
  --fallback-simple

"""
import argparse, glob, os, math
import pandas as pd
import numpy as np

from scripts.MDL.line_eventizer import eventize
try:
    from scripts.MDL.adapter_to_photoncodes_vega_elodie import (
        load_elodie,                                  # ← add this
        moving_average as _ma, rolling_mad as _mad,
        find_lines_adaptive as _find_simple
    )
except Exception:
    _ma = _mad = _find_simple = None

    load_elodie = None  # will set below

if load_elodie is None:
    try:
        # fall back to the adapter’s loader if available under a different symbol
        from scripts.MDL.adapter_to_photoncodes_vega_elodie import load_elodie as _load_elodie_fallback
        load_elodie = _load_elodie_fallback
    except Exception:
        # last resort: very generic two/three-column ASCII loader from the eventizer
        from scripts.MDL.line_eventizer import load_ascii_two_or_three_cols as load_elodie

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--unit", default="auto", choices=["nm","angstrom","auto"])
    ap.add_argument("--smooth-win", type=int, default=9)
    # tighter MAD helps metals in the presence of broad Balmer wings
    ap.add_argument("--win-mad", type=int, default=201)
    ap.add_argument("--k-sigma", type=float, default=4.0)
    ap.add_argument("--min-sep-px", type=int, default=7)
    ap.add_argument("--merge-nm", type=float, default=None)
    ap.add_argument("--merge-kms", type=float, default=2.0,
                    help="If set, merge by |Δλ|/λ * c <= Δv (km/s); overrides --merge-nm when not None.")
    ap.add_argument("--one-per-kappa", action="store_true", default=False)
    ap.add_argument("--k-sigma-seed", type=float, default=None)
    ap.add_argument("--k-sigma-final", type=float, default=None)
    # For narrow metals, default to modest widths; broaden only if you want Balmer wings
    ap.add_argument("--sigmas-pix", type=str, default="2,3,5,8,12,20,35")
    # continuum options
    ap.add_argument("--cont-win", type=int, default=2401, help="rolling quantile window for continuum")
    ap.add_argument("--cont-q", type=float, default=0.995, help="continuum quantile (0.98–0.997)")
    ap.add_argument("--no-continuum", action="store_true", help="skip continuum division (use raw flux)")
    # fallback and debug
    ap.add_argument("--fallback-simple", action="store_true",
                    help="if eventizer returns zero lines, try old adaptive-minima path")
    ap.add_argument("--debug", action="store_true", help="print seed/keep counts per file")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.txt")))
    if not files:
        raise SystemExit(f"No .txt files found in {args.indir}")

    rows = []

    # Rolling quantile continuum (same idea as line_eventizer CLI) — keep the stricter one
    def _rolling_quantile(x: np.ndarray, win: int, q: float) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view
        if win % 2 == 0: win += 1
        k = win // 2
        pad = np.pad(x, (k,k), mode="edge")
        W = sliding_window_view(pad, win)
        return np.quantile(W, q, axis=1)

    for path in files:
        lam_nm, flux = load_elodie(path, unit=args.unit)
        if lam_nm.size == 0:
            continue

        # enforce ascending λ
        order = np.argsort(lam_nm); lam_nm = lam_nm[order]; flux = flux[order]
        # continuum-normalize (optional)
        if args.no_continuum:
            fn = flux.copy()
        else:
            cont = _rolling_quantile(flux, win=max(args.cont_win, args.win_mad*6), q=float(args.cont_q))
            fn = flux / np.maximum(cont, 1e-12)

        # eventize this slice
        sigmas = np.array([float(s) for s in args.sigmas_pix.split(",") if s.strip()], dtype=float)
        df_evt = eventize(
            lam_nm, fn,
            win_smooth=args.smooth_win, win_mad=args.win_mad,
            # easier seed / moderate final for A0: helps broad Balmer + narrow metals
            k_sigma=(args.k_sigma or 4.0),
            k_sigma_seed=(args.k_sigma_seed if args.k_sigma_seed is not None else (args.k_sigma or 4.0)*0.75),
            k_sigma_final=(args.k_sigma_final if args.k_sigma_final is not None else (args.k_sigma or 4.0)),
            sigmas_pix=sigmas
        )
        if args.debug:
            print(f"[vega] {os.path.basename(path)}: eventizer → {len(df_evt)} lines")
        if df_evt.empty and args.fallback_simple and (_find_simple is not None):
            # Simple adaptive minima directly on *normalized* flux
            fls = _ma(fn, args.smooth_win) if _ma else fn
            # rolling MAD for weights on normalized profile
            med, mad = (_mad(fls, win=args.win_mad) if _mad else (np.median(fls)*np.ones_like(fls),
                                                                 np.median(np.abs(fls-np.median(fls)))*np.ones_like(fls)))
            idx = _find_simple(lam_nm, fn, win_smooth=args.smooth_win, win_mad=args.win_mad,
                               k_sigma=(args.k_sigma or 4.0), min_sep_px=args.min_sep_px)
            if idx.size:
                dep = (med[idx] - fls[idx]) if med.size==fls.size else (1.0 - fls[idx])
                w = dep / (np.max(dep) if dep.size else 1.0)
                nu = (299_792_458.0 / (lam_nm[idx]*1e-9))
                df_evt = pd.DataFrame({
                    "wavelength_nm": lam_nm[idx],
                    "frequency_hz": nu,
                    "log10_nu_hz": np.log10(nu),
                    "depth": dep,
                    "weight": w
                })
                if args.debug:
                    print(f"[vega] {os.path.basename(path)}: fallback-simple → {len(df_evt)} lines")
        if not df_evt.empty:
            rows.append(df_evt)

    if not rows:
        raise SystemExit("No lines found with current thresholds.")

    df = pd.concat(rows, ignore_index=True)

    # wavelength de-dup (Δv preferred; else Δλ)  -- single pass only
    df = df.sort_values("wavelength_nm").reset_index(drop=True)
    lam = df["wavelength_nm"].to_numpy()
    groups, cur = [], [0]
    if getattr(args, "merge_kms", None) is not None:
        c_kms = 299792.458
        for i in range(1, len(df)):
            dv = abs(lam[i] - lam[cur[-1]])/lam[i] * c_kms
            if dv <= args.merge_kms: cur.append(i)
            else: groups.append(cur); cur = [i]
        groups.append(cur)
    else:
        merge_nm = getattr(args, "merge_nm", 0.004) or 0.004
        for i in range(1, len(df)):
            if lam[i] - lam[cur[-1]] <= merge_nm: cur.append(i)
            else: groups.append(cur); cur=[i]
        groups.append(cur)

    winners = []
    for g in groups:
        sub = df.iloc[g]
        # choose strongest by weight; fall back to deepest (max depth)
        if "weight" in sub and sub["weight"].notna().any():
            j = sub["weight"].idxmax()
        elif "depth" in sub and sub["depth"].notna().any():
            j = sub["depth"].idxmax()
        else:
            j = sub.index[0]
        winners.append(df.loc[j])

    df = pd.DataFrame(winners).sort_values("wavelength_nm").reset_index(drop=True)

    if args.one_per_kappa:
        beta = math.log10(1/137.035999084)
        y = np.log10(df["frequency_hz"].to_numpy())
        kappa = (y - y.max())/beta
        bin_idx = np.floor(kappa/0.002).astype(int)
        keep_idx = (
            pd.DataFrame({
                "bin": bin_idx,
                "w": df["weight"].fillna(0.0) if "weight" in df else pd.Series(1.0, index=df.index)
            })
            .groupby("bin")["w"].idxmax()
            .to_numpy()
        )
        df = df.loc[keep_idx].sort_values("wavelength_nm").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df):,} Vega lines from {len(files)} files -> {args.out_csv}")


if __name__ == "__main__":
    main()
