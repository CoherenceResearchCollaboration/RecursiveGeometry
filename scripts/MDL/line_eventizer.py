#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line Eventizer: waveform -> physically-meaningful line events

Given a 1D spectrum (wavelength, flux), detect absorption line centers
in an amplitude-agnostic way (adaptive SNR against a rolling MAD),
refine centroids, estimate FWHM via half-depth crossings, compute
equivalent width (EW), and emit an event table with a salience weight.

Outputs a CSV with columns:
  wavelength_nm, frequency_hz, log10_nu_hz, depth, snr, fwhm_nm, ew_nm, weight

Typical usage:
python -m scripts.MDL.line_eventizer \
  --in data/solar/fluxatl/lm0296.txt \
  --unit nm \
  --out-csv results/solar/raw_csv/lm0296_events.csv \
  --win-smooth 9 --win-mad 101 --k-sigma 4.5 --min-sep-px 7 \
  --plot results/solar/plots/lm0296_events_norm.png --invert-x

python -m scripts.MDL.line_eventizer \
  --in data/stars/vega_elodie/elodie_19940830_0009.txt \
  --unit nm \
  --out-csv results/stars/raw_csv/vega_0009_events.csv \
  --win-smooth 9 --win-mad 401 --k-sigma 6.0 --min-sep-px 9 \
  --plot results/stars/plots/vega_0009_events_norm.png --invert-x

Optional debug plot:
  python -m scripts.MDL.line_eventizer \
    --in data/stars/vega_elodie/elodie_19940830_0009.txt \
    --unit nm \
    --out-csv results/stars/raw_csv/vega_0009_events.csv \
    --plot results/stars/plots/vega_0009_events.png --invert-x

Notes
-----
* Detection is amplitude-agnostic (contrast/SNR); amplitude is preserved
  as metadata: depth, SNR, EW, weight (SNR × EW by default).
* The code assumes an absorption spectrum (continuum near a smooth median).
  For emission spectra, flip sign (see TODO).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

C_M_PER_S = 299_792_458.0  # speed of light (m/s)


# ------------------------------- helpers -------------------------------------

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Simple centered moving average with edge cloning; odd win >= 3 recommended."""
    if win < 3 or win % 2 == 0:
        return x
    k = win // 2
    kernel = np.ones(win, dtype=float) / float(win)
    y = np.convolve(x, kernel, mode="same")
    # edge steady padding
    y[:k] = y[k]
    y[-k:] = y[-k - 1]
    return y


def rolling_median_mad(x: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling median and MAD (median absolute deviation).
    Returns arrays aligned to x (same length). Edge handling via steady padding.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if win < 5 or win % 2 == 0:
        win = 401
    k = win // 2
    pad = np.pad(x, (k, k), mode="edge")
    W = sliding_window_view(pad, win)  # shape (N, win)
    med = np.median(W, axis=1)
    mad = np.median(np.abs(W - med[:, None]), axis=1)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    return med, mad


def strict_minima(y: np.ndarray) -> np.ndarray:
    """Indices i (1..N-2) where y[i] < y[i-1] and y[i] < y[i+1]."""
    if y.size < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1


def parabolic_refine(x: np.ndarray, y: np.ndarray, i: int, halfwin: int = 3) -> Tuple[float, float]:
    """
    Fit y ~ a*z^2 + b*z + c in a small window around i (z normalized x),
    return (x_vertex, y_vertex). Falls back to (x[i], y[i]) if ill-conditioned.
    """
    lo = max(0, i - halfwin)
    hi = min(len(x), i + halfwin + 1)
    xv = float(x[i])
    yv = float(y[i])
    if hi - lo >= 3:
        X = x[lo:hi].astype(float)
        Y = y[lo:hi].astype(float)
        x0 = X.mean()
        s = np.std(X) or 1.0
        z = (X - x0) / s
        A = np.c_[z * z, z, np.ones_like(z)]
        try:
            a, b, c = np.linalg.lstsq(A, Y, rcond=None)[0]
            if abs(a) > 1e-20:
                zstar = -b / (2 * a)
                xv = float(x0 + s * zstar)
                yv = float(a * zstar * zstar + b * zstar + c)
        except Exception:
            pass
    return xv, yv


def half_depth_fwhm(x: np.ndarray, y: np.ndarray, i: int, cont: float) -> Tuple[float, float]:
    """
    Half-depth crossings for absorption: target = cont - 0.5*(cont - y[i]).
    Returns (xL, xR) crossings; if not found, returns (x[i], x[i]).
    """
    target = cont - 0.5 * (cont - y[i])
    n = len(x)
    # left crossing
    L = i
    while L > 0 and y[L] <= target:
        L -= 1
    if L == i or L == 0:
        xL = x[i]
    else:
        # linear interpolation across (L, L+1)
        xL = x[L] + (x[L + 1] - x[L]) * ((target - y[L]) / (y[L + 1] - y[L] + 1e-30))
    # right crossing
    R = i
    while R < n - 1 and y[R] <= target:
        R += 1
    if R == i or R == n - 1:
        xR = x[i]
    else:
        xR = x[R - 1] + (x[R] - x[R - 1]) * ((target - y[R - 1]) / (y[R] - y[R - 1] + 1e-30))
    return float(xL), float(xR)

def nm_to_frequency_hz(lambda_nm: np.ndarray) -> np.ndarray:
    lam_m = lambda_nm * 1e-9
    return C_M_PER_S / lam_m

def load_photatl_wn(path: str):
    """
    Robust loader for NSO photatl pages ('wn####' or 'wn####.txt').
    Skips header/footer lines that cause bogus wavenumbers.
    Returns (lambda_nm, solar_flux).
    """
    import numpy as np, re
    wn_list, flux_list = [], []
    float_line = re.compile(r"[0-9.Ee+-]{10,}")  # at least two float fields
    with open(path, "r", errors="ignore") as f:
        for line in f:
            # skip obvious headers / blanks
            if not float_line.search(line):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                wn = float(parts[0])
                fl = float(parts[1])
            except Exception:
                continue
            # accept only plausible photatl wn range
            if 1800.0 <= wn <= 9200.0:
                wn_list.append(wn)
                flux_list.append(fl)
    if not wn_list:
        return np.array([]), np.array([])
    wn_cm1 = np.array(wn_list, dtype=float)
    flux_solar = np.array(flux_list, dtype=float)
    lam_nm = 1e7 / wn_cm1
    order = np.argsort(lam_nm)
    return lam_nm[order], flux_solar[order]

def load_ascii_two_or_three_cols(path: str, unit: str = "nm") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load whitespace ASCII; use first numeric column as wavelength, second as flux.
    Ignores non-numeric lines and extra columns. Converts Å->nm if requested.
    """
    waves, flux = [], []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            # scan for first two numeric tokens
            got = []
            for p in parts:
                try:
                    got.append(float(p))
                    if len(got) == 2:
                        break
                except Exception:
                    continue
            if len(got) == 2:
                w, y = got
                waves.append(w)
                flux.append(y)
    lam = np.asarray(waves, dtype=float)
    flx = np.asarray(flux, dtype=float)
    u = unit.lower()
    if u in ("angstrom", "ang", "a", "å"):
        lam *= 0.1  # Å -> nm
    return lam, flx

def gaussian_kernel(sigma_pix: float, radius_mult: float = 4.0) -> np.ndarray:
    """Normalized 1D Gaussian of width sigma_pix (in pixels)."""
    half = int(max(1, round(radius_mult * sigma_pix)))
    x = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-0.5 * (x / sigma_pix) ** 2)
    g /= g.sum() if g.sum() != 0 else 1.0
    return g

def rolling_mad_centered(x: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling MAD, steady-padded; win must be odd."""
    from numpy.lib.stride_tricks import sliding_window_view
    if win % 2 == 0: win += 1
    k = win // 2
    pad = np.pad(x, (k, k), mode="edge")
    W = sliding_window_view(pad, win)
    med = np.median(W, axis=1)
    mad = np.median(np.abs(W - med[:, None]), axis=1)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    return mad

def local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices of strict local maxima in 1D array x (same as strict_minima but for maxima)."""
    if x.size < 3: return np.array([], dtype=int)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1

def nms_suppress(indices: np.ndarray, scores: np.ndarray, radius_px: np.ndarray) -> np.ndarray:
    """
    Non-maximum suppression: keep highest score in any ±radius region.
    radius_px may be scalar or per-index array (same length as indices).
    """
    order = np.argsort(scores)[::-1]           # high -> low
    kept = []
    occupied = np.zeros_like(scores, dtype=bool)
    # We’ll mark suppressed by marking those indices in a boolean mask over the pixel axis:
    # but to keep O(N log N), do local checks using the list of kept positions.
    pos = indices.astype(int)
    r = radius_px if np.ndim(radius_px) else np.full_like(pos, int(radius_px))
    taken = np.zeros(0, dtype=int)
    bands = []
    for j in order:
        p, rad = pos[j], int(max(1, r[j]))
        # check overlap with already kept positions
        ok = True
        for pk, rk in bands:
            if abs(p - pk) <= max(rad, rk):
                ok = False; break
        if ok:
            kept.append(j)
            bands.append((p, rad))
    kept = np.array(sorted(kept), dtype=int)
    return kept

# ------------------------------ eventizer ------------------------------------

def eventize(
    lambda_nm: np.ndarray,
    flux: np.ndarray,
    win_smooth: int = 9,
    win_mad: int = 401,
    k_sigma: float = 9.0,
    k_sigma_seed: Optional[float] = None,
    k_sigma_final: Optional[float] = None,
    min_sep_px: int = 9,
    max_peaks: Optional[int] = None,
    ew_window_mult: float = 2.5,
    sigmas_pix: Optional[np.ndarray] = None,
    min_fwhm_px: Optional[float] = None,
    min_ew_nm: float = 5e-5,
    debug: bool = False,
    nms_mult: float = 1.2,
) -> pd.DataFrame:
    """
    Core logic:
      - Smooth -> rolling median/MAD
      - Strict minima -> SNR gate (depth/MAD >= k_sigma)
      - Greedy min separation by strongest depth
      - Sub-pixel centroid (parabola)
      - FWHM via half-depth crossings
      - EW by trapezoid of (1 - F/continuum) in a local window
      - weight = SNR * EW (nm)
    """
    N = len(lambda_nm)
    if N < 5 or len(flux) != N:
        return pd.DataFrame(columns=[
            "wavelength_nm", "frequency_hz", "log10_nu_hz",
            "depth", "snr", "fwhm_nm", "ew_nm", "weight"
        ])

    # auto-scale flux if it is not roughly normalized (median >> 10)
    if np.nanmedian(flux) > 10.0:
        flux = flux / np.nanpercentile(flux, 98)

    # thresholds: allow split seed/final; default to k_sigma if not provided
    if k_sigma_seed  is None: k_sigma_seed  = float(k_sigma)
    if k_sigma_final is None: k_sigma_final = float(k_sigma)

    # 1) smooth + baseline (on continuum-normalized flux)

    y_s = moving_average(flux, win_smooth)   # here 'flux' is fn if caller normalized; else raw
    # Work on absorbance so lines are positive features
    A = np.clip(1.0 - y_s, 0.0, None)

    # 2) multiscale matched filtering on absorbance
    #    choose a set of Gaussian widths (in pixels) that covers narrow..moderate lines
    if sigmas_pix is None or sigmas_pix.size == 0:
        sigmas_pix = np.array([2, 3, 5, 8, 12], dtype=float)
    resp_scores = []
    resp_indices = []
    resp_radii = []     # suppression radius per seed

    for s in sigmas_pix:
        g = gaussian_kernel(sigma_pix=s, radius_mult=4.0)
        conv = np.convolve(A, g, mode="same")
        # local z-score of the response to be amplitude-agnostic wrt ripple
        mad_conv = rolling_mad_centered(conv, win=max(101, int(20 * s)))
        z = (conv - np.median(conv)) / mad_conv
        # local maxima in the filtered response
        peaks = local_maxima(conv)

        # keep peaks with z >= k_sigma_seed
        good = peaks[z[peaks] >= float(k_sigma_seed)]
        if good.size:
            resp_indices.append(good)
            resp_scores.append(z[good])
            # NMS radius ~ 1.5 * sigma pixels (merge near-duplicates at this scale)
            resp_radii.append(np.full_like(good, int(round(nms_mult * s))))

    # 3) union seeds from all scales and apply non-maximum suppression
    if len(resp_indices) == 0:
        return pd.DataFrame(columns=[
            "wavelength_nm","frequency_hz","log10_nu_hz",
            "depth","snr","fwhm_nm","ew_nm","weight"
        ])

    all_idx = np.concatenate(resp_indices)
    all_scr = np.concatenate(resp_scores)
    all_rad = np.concatenate(resp_radii)
    keep_ids = nms_suppress(all_idx, all_scr, all_rad)

    cand = all_idx[keep_ids]
    score = all_scr[keep_ids]
    # sort by pixel index for stability
    ordr = np.argsort(cand)
    cand = cand[ordr]; score = score[ordr]

    # 4) compute SNR on the *narrow* residual for kept seeds (final amplitude-agnostic gate)
    if debug:
        print(f"[eventizer] seeds after NMS: {len(cand)}")
    # noise scale from rolling MAD, but depth is vs. normalized continuum (~1.0)
    med_res, mad_res = rolling_median_mad(y_s, win_mad)
    depth = 1.0 - y_s[cand]                # absorption depth after normalization
    snr   = depth / mad_res[cand]
    # optional: uncomment if you want quick telemetry
    # print(f"[eventizer] seeds after NMS: {cand.size}")
    good = np.isfinite(snr) & (snr >= float(k_sigma_final))
    cand = cand[good]; snr = snr[good]; depth = depth[good]
    if debug:
        print(f"[eventizer] after SNR gate (k={k_sigma_final}): {len(cand)}")
    # print(f"[eventizer] after SNR gate (k={k_sigma_final}): {cand.size}")

    if cand.size == 0:
        return pd.DataFrame(columns=[
            "wavelength_nm","frequency_hz","log10_nu_hz",
            "depth","snr","fwhm_nm","ew_nm","weight"
        ])

    # 5) enforce pixel min-sep (final pass)
    order = np.argsort(depth)[::-1]
    sel = []
    occ = np.zeros(len(y_s), dtype=bool)
    for j in order:
        i = int(cand[j])
        if occ[max(0, i - min_sep_px):min(len(y_s), i + min_sep_px + 1)].any():
            continue
        sel.append(i)
        occ[max(0, i - min_sep_px):min(len(y_s), i + min_sep_px + 1)] = True
        if max_peaks and len(sel) >= max_peaks:
            break
    sel = np.array(sorted(sel), dtype=int)
    if sel.size == 0:
        return pd.DataFrame(columns=[
            "wavelength_nm","frequency_hz","log10_nu_hz",
            "depth","snr","fwhm_nm","ew_nm","weight"
        ])

    # 5) per-line refinement and metrics (normalized domain)
    pix_nm = float(np.median(np.diff(lambda_nm)))  # median grid step
    rows = []
    for i in sel:
        # refined centroid on the smoothed, normalized flux
        x_c, y_c = parabolic_refine(lambda_nm, y_s, i, halfwin=3)

        # after continuum normalization, local continuum ~ 1.0
        cont_i = 1.0
        d = float(cont_i - y_c)  # depth = 1 - flux_at_center
        snr_i = float(d / (mad_res[i] if mad_res[i] > 0 else 1e-12))

        # FWHM via half-depth crossings on the smoothed, normalized flux
        xL, xR = half_depth_fwhm(lambda_nm, y_s, i, cont_i)
        fwhm = float(abs(xR - xL))
        if fwhm <= 0:                 # degenerate → give it a small, pixel-aware width
            fwhm = max(2.0 * pix_nm, 1.0 * pix_nm)

        # Build a local integration window for EW
        # If FWHM is zero/degenerate, fall back to a small pixel-based window
        if fwhm <= 0:
            pix_nm = np.median(np.diff(lambda_nm))
            halfw = max(2.5 * pix_nm, 3 * pix_nm)
        else:
            halfw = ew_window_mult * fwhm

        lo_x = x_c - halfw
        hi_x = x_c + halfw
        lo = max(0, int(np.searchsorted(lambda_nm, lo_x)))
        hi = min(N - 1, int(np.searchsorted(lambda_nm, hi_x)))
        if hi - lo < 2:
            lo = max(0, i - 2)
            hi = min(N - 1, i + 2)

        # Local arrays (normalized flux and wavelengths)
        X = lambda_nm[lo:hi + 1]
        Y = y_s[lo:hi + 1]

        # Equivalent width on normalized residual (continuum ~ 1.0)
        norm_deficit = np.clip(1.0 - Y, 0.0, None)
        ew_nm = float(np.trapz(norm_deficit, X))

        # Guardrails (instrument-aware, now CLI-tunable)
        # Kitt Peak FTS sampling is ~3e-4 nm near 296 nm (narrow lines) — keep per-pixel thresholds. :contentReference[oaicite:5]{index=5}
        min_fwhm_nm = (float(min_fwhm_px) * pix_nm) if (min_fwhm_px and min_fwhm_px > 0) else (2.0 * pix_nm)
        min_ew_nm   = float(min_ew_nm)
        if ew_nm < min_ew_nm or fwhm < min_fwhm_nm:
            continue

        # Convert center to frequency (Hz)
        lam_m = x_c * 1e-9
        nu_hz = float(C_M_PER_S / lam_m) if lam_m > 0 else np.nan
        log10_nu = float(np.log10(nu_hz)) if nu_hz > 0 else np.nan

        rows.append({
            "wavelength_nm": float(x_c),
            "frequency_hz": nu_hz,
            "log10_nu_hz": log10_nu,
            "depth": d,
            "snr": snr_i,
            "fwhm_nm": fwhm,
            "ew_nm": ew_nm,
            # default salience weight: SNR × EW (nm)
            "weight": float(snr_i * ew_nm)
        })

    return pd.DataFrame(rows)


# ---------------------------------- CLI --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Eventize a spectrum into line centers with SNR/FWHM/EW weights.")
    ap.add_argument("--in", dest="infile", required=True, help="Input ASCII spectrum (wavelength, flux)")
    ap.add_argument("--unit", choices=["nm", "angstrom"], default="nm", help="Wavelength unit of input")
    ap.add_argument("--out-csv", dest="out_csv", required=True, help="Output CSV path for line events")

    # detection / metrics knobs
    ap.add_argument("--win-smooth", type=int, default=9, help="Moving average window (odd)")
    ap.add_argument("--win-mad", type=int, default=401, help="Rolling MAD/median window (odd)")
    ap.add_argument("--k-sigma", type=float, default=9.0, help="Adaptive SNR threshold (used if seed/final not set)")
    ap.add_argument("--k-sigma-seed", type=float, default=None, help="Seed threshold on multiscale response z (default: --k-sigma)")
    ap.add_argument("--k-sigma-final", type=float, default=None, help="Final SNR gate on residual (default: --k-sigma)")
    ap.add_argument("--min-sep-px", type=int, default=9, help="Min pixel separation between retained lines")
    ap.add_argument("--max-peaks", type=int, default=None, help="Optional cap on number of lines")
    ap.add_argument("--ew-window-mult", type=float, default=2.5, help="EW integration half-window = mult * FWHM")

    # optional debug plot
    ap.add_argument("--plot", type=str, default=None, help="Write a debug PNG with retained lines overlaid")
    ap.add_argument("--invert-x", action="store_true", default=False, help="Invert x-axis (shorter λ to left)")
    ap.add_argument("--cont-win", type=int, default=1201, help="rolling window for continuum quantile")
    ap.add_argument("--cont-quantile", type=float, default=0.98, help="continuum quantile (0.95..0.995)")
    ap.add_argument("--sigmas-pix", type=str, default="2,3,5,8,12,20,35,60",
                    help="Comma-separated Gaussian widths (in pixels) for multiscale detector (pixels)")
    # tunable guardrails for high-resolution solar tiles
    ap.add_argument("--min-fwhm-px", type=float, default=2.0,
                    help="Minimum FWHM in instrument pixels (default 2.0)")
    ap.add_argument("--min-ew-nm", type=float, default=5e-5,
                    help="Minimum equivalent width (nm) to keep a line (default 5e-5)")
    # optional telemetry
    ap.add_argument("--debug", action="store_true", help="Print seed/SNR counts for tuning")
    ap.add_argument("--nms-mult", type=float, default=1.2,
                    help="Non-max suppression radius = nms_mult × sigma_pix (default 1.2)")

    args = ap.parse_args()

    # load
    lam_nm, flx = load_ascii_two_or_three_cols(args.infile, unit=args.unit)
    if lam_nm.size == 0 or flx.size != lam_nm.size:
        raise SystemExit(f"[eventizer] No valid data in {args.infile}")

    # enforce ascending wavelength so searchsorted / FWHM windows behave
    order = np.argsort(lam_nm)
    lam_nm = lam_nm[order]
    flx    = flx[order]

    # ------------------------------------------------------------
    # Continuum normalization and optional broad-line seeding
    # ------------------------------------------------------------
    def rolling_quantile(x: np.ndarray, win: int = 801, q: float = 0.95) -> np.ndarray:
        """High-percentile rolling quantile continuum (pseudo-continuum)."""
        from numpy.lib.stride_tricks import sliding_window_view
        if win % 2 == 0:
            win += 1
        k = win // 2
        pad = np.pad(x, (k, k), mode="edge")
        W = sliding_window_view(pad, win)
        return np.quantile(W, q, axis=1)

    # --- compute continuum (keep it above absorption troughs) ---
    cont = rolling_quantile(flx, win=args.cont_win, q=args.cont_quantile)
    fn = flx / np.maximum(cont, 1e-12)

    # --- run the eventizer on the normalized flux ---
    # parse sigmas-pix
    sigmas_pix = np.array([float(s) for s in args.sigmas_pix.split(",") if s.strip()], dtype=float)

    df = eventize(
        lam_nm, fn,                       # note: fn, not flx
        win_smooth=args.win_smooth,
        win_mad=args.win_mad,
        k_sigma=args.k_sigma,
        k_sigma_seed=(args.k_sigma_seed if args.k_sigma_seed is not None else args.k_sigma),
        k_sigma_final=(args.k_sigma_final if args.k_sigma_final is not None else args.k_sigma),
        min_sep_px=args.min_sep_px,
        max_peaks=args.max_peaks,
        ew_window_mult=args.ew_window_mult,
        # pass the list down
        sigmas_pix=sigmas_pix,
        min_fwhm_px=args.min_fwhm_px,
        min_ew_nm=args.min_ew_nm,
        debug=bool(args.debug),
        nms_mult=float(args.nms_mult)
    )

    # write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[eventizer] lines: {len(df):,} -> {args.out_csv}")

    # optional debug plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            y_s = moving_average(flx, args.win_smooth)
            med, _ = rolling_median_mad(y_s, args.win_mad)

            plt.figure(figsize=(12, 4))
            plt.plot(lam_nm, flx, lw=0.6, label="raw")
            plt.plot(lam_nm, y_s, lw=0.8, alpha=0.7, label=f"smooth (win={args.win_smooth})")
            plt.plot(lam_nm, cont, lw=1.0, alpha=0.7, color="purple",
                    label=f"continuum (q={args.cont_quantile})")
            plt.plot(lam_nm, med, lw=0.8, alpha=0.7, label=f"rolling median (win={args.win_mad})")
            if not df.empty:
                plt.scatter(df["wavelength_nm"], np.interp(df["wavelength_nm"], lam_nm, flx),
                            s=14, c="r", label=f"lines (n={len(df)})", zorder=5)
            if args.invert_x:
                plt.gca().invert_xaxis()
            plt.xlabel(f"wavelength ({args.unit})")
            plt.ylabel("flux (arb.)")
            plt.title(Path(args.infile).name)
            plt.legend()
            Path(Path(args.plot).parent).mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=180)
            print(f"[eventizer] plot: {args.plot}")
        except Exception as e:
            print(f"[eventizer] plot failed: {e}")

if __name__ == "__main__":
    main()
