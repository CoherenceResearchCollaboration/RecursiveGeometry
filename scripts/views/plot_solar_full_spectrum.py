#!/usr/bin/env python3
"""
plot_solar_full_spectrum.py
---------------------------
Make publication-quality full-spectrum plots for the Sun using your
event CSVs (the same inputs used for MDL). The figure shows a
normalized, continuous wavelength view plus optional line overlays.

USAGE EXAMPLES
--------------
# 1) Single, merged events CSV that contains both bands (recommended)
python -m scripts.views.ion_identity.plot_solar_full_spectrum \
  --events-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --outdir results/raw_plots --normalize p98 --binsize 0.10 \
  --overlay-ticks --verbose

# 2) Separate CSVs (one for FluxAtlas, one for PhotAtlas)
python -m scripts.views.ion_identity.plot_solar_full_spectrum \
  --events-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --phot-events-csv results/solar/raw_csv/photatl_all_events.csv \
  --outdir results/raw_plots --normalize p98 --binsize 0.10 \
  --overlay-ticks --verbose

# 3) Restrict to a window (e.g., 390–700 nm)
python -m scripts.views.ion_identity.plot_solar_full_spectrum \
  --events-csv results/solar/raw_csv/fluxatl_all_events.csv \
  --xlim 390,700 --outdir results/raw_plots --overlay-ticks --verbose

OUTPUTS
-------
results/raw_plots/solar_full_fluxatl.png   # FluxAtlas (296–1300 nm)
results/raw_plots/solar_full_photatl.png   # PhotAtlas (IR)
results/raw_plots/solar_full_combined.png  # full-span overlay (if both bands found)
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _msg(s: str, v: bool):
    if v:
        print(s)


def read_events(csv_path: Path, use_weights: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    # wavelength column
    wl = cols.get("wavelength_nm") or cols.get("wavelength") or cols.get("lambda_nm")
    if not wl:
        raise SystemExit(f"[ERROR] No wavelength column in {csv_path}")
    df = df.rename(columns={wl: "wavelength_nm"})
    df["wavelength_nm"] = pd.to_numeric(df["wavelength_nm"], errors="coerce")

    # build 'signal'
    if use_weights:
        # prefer modest-scale columns; avoid SNR by clipping
        picked = None
        for name in ("weight", "intensity", "ew_nm", "depth"):
            col = cols.get(name)
            if col:
                picked = col; break
        if picked:
            sig = pd.to_numeric(df[picked], errors="coerce")
            p999 = np.nanpercentile(sig, 99.9) if np.isfinite(sig).any() else 0
            if np.isfinite(p999) and p999 > 0:
                sig = np.clip(sig, 0, p999)
            df["signal"] = sig.fillna(0.0)
        else:
            df["signal"] = 1.0
    else:
        df["signal"] = 1.0

    # keep source_path if present (for band split)
    if "source_path" not in df.columns and "source_path" in cols:
        df = df.rename(columns={cols["source_path"]: "source_path"})

    return df.dropna(subset=["wavelength_nm"])[
        ["wavelength_nm", "signal", *(["source_path"] if "source_path" in df.columns else [])]
    ]

def band_filter(df: pd.DataFrame, want: Optional[str]) -> pd.DataFrame:
    """
    want: None | 'fluxatl' | 'photatl'
    Primary: split using source_path when available (case-insensitive).
    Fallback: split by wavelength if source_path is absent.
    """
    if want is None:
        return df.copy()

    # 1) Try source_path-based split first
    if "source_path" in df.columns:
        sp = df["source_path"].astype(str).str.lower()
        if want.lower().startswith("flux"):
            mask = sp.str.contains("fluxatl") | sp.str.contains("fluxatlas") | sp.str.contains("flux")
        else:  # photatl
            mask = sp.str.contains("photatl") | sp.str.contains("photatlas") | sp.str.contains("phot")
        # If this actually splits rows, use it:
        if mask.any() and (~mask).any():
            return df.loc[mask].copy()
        # else fall through to wavelength split

    # 2) Wavelength fallback (nm)
    wl = pd.to_numeric(df.get("wavelength_nm", np.nan), errors="coerce")
    if want.lower().startswith("flux"):
        # FluxAtlas ~ 296–1300 nm
        mask = (wl >= 250) & (wl <= 1300)
    else:
        # PhotAtlas ~ 1100 nm and longer (IR)
        mask = (wl >= 1100)
    return df.loc[mask.fillna(False)].copy()


def make_histogram(df: pd.DataFrame,
                   binsize: float,
                   xlim: Optional[Tuple[float, float]],
                   verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted 1D histogram of wavelength_nm using 'signal' as weight.
    Returns (bin_centers, normalized_counts)
    """
    wl = df["wavelength_nm"].to_numpy()
    sig = df["signal"].to_numpy()
    if xlim:
        lo, hi = xlim
        keep = (wl >= lo) & (wl <= hi)
        wl, sig = wl[keep], sig[keep]
    if wl.size == 0:
        return np.array([]), np.array([])

    lo = float(np.nanmin(wl))
    hi = float(np.nanmax(wl))
    # guard for binsize
    if binsize <= 0:
        binsize = (hi - lo) / 2000.0 if hi > lo else 0.1
    nbins = int(max(1, np.ceil((hi - lo) / binsize)))
    edges = np.linspace(lo, hi, nbins + 1)
    counts, _ = np.histogram(wl, bins=edges, weights=sig)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # simple smoothing (3-bin moving avg) to reduce impulsive spikes; optional
    if counts.size >= 3:
        k = np.array([1,2,1], float); k /= k.sum()
        counts = np.convolve(counts, k, mode="same")
    return centers, counts


def normalize_series(y: np.ndarray, mode: str) -> np.ndarray:
    if y.size == 0:
        return y
    y = y.astype(float)
    y = np.clip(y, 0, None)
    if mode == "p98":
        p = np.nanpercentile(y, 98)
        if p > 0:
            return y / p
    elif mode == "max":
        m = np.nanmax(y)
        if m > 0:
            return y / m
    # fallback to linear 0-1
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    rng = (ymax - ymin) if ymax > ymin else 1.0
    return (y - ymin) / rng


def plot_band(ax: plt.Axes,
              x: np.ndarray, y: np.ndarray,
              title: str,
              xlim: Optional[Tuple[float, float]],
              overlay_ticks: Optional[np.ndarray] = None):
    if x.size == 0:
        ax.set_axis_off(); ax.set_title(title); return
    idx = np.argsort(x)
    ax.plot(x[idx], y[idx], lw=0.9)
    if overlay_ticks is not None and overlay_ticks.size:
        ymin, ymax = ax.get_ylim()
        yy0, yy1 = ymin, ymin + 0.12*(ymax - ymin)  # short faint stubs
        ax.vlines(overlay_ticks, yy0, yy1, colors="0.4", alpha=0.25, lw=0.4)
    if xlim: ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalized intensity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-csv", required=True,
                    help="Solar events CSV (merged or FluxAtlas-only).")
    ap.add_argument("--phot-events-csv", default=None,
                    help="Optional PhotAtlas CSV; if omitted we will try to pull photatl from --events-csv via source_path.")
    ap.add_argument("--binsize", type=float, default=0.10,
                    help="Histogram bin width in nm (default 0.10).")
    ap.add_argument("--normalize", choices=["p98","max","linear"], default="p98",
                    help="Normalization mode for the binned curve (default p98).")
    ap.add_argument("--xlim", type=str, default=None,
                    help="Optional 'lo,hi' wavelength window in nm.")
    ap.add_argument("--overlay-ticks", action="store_true",
                    help="Also mark detected line centers as faint ticks.")
    ap.add_argument("--outdir", required=True, help="Output directory for PNGs.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-weights", action="store_true",
                    help="Ignore weight/intensity columns; use 1.0 per event")

    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    xlim = None
    if args.xlim:
        try:
            lo, hi = args.xlim.split(",")
            xlim = (float(lo), float(hi))
        except Exception:
            xlim = None

    # --- Read base events
    df_all = read_events(Path(args.events_csv), use_weights=(not args.no_weights))
    _msg(f"[read] {len(df_all):,} rows from {args.events_csv}", args.verbose)

    # Split by band if possible (or leave as all)
    df_flux = band_filter(df_all, "fluxatl")
    df_phot = band_filter(df_all, "photatl")

    print(f"[debug] total rows: {len(df_all):,}")
    print(f"[debug] fluxatl rows: {len(df_flux):,}")
    print(f"[debug] photatl rows: {len(df_phot):,}")
    if "source_path" in df_all.columns:
        sp = df_all["source_path"].astype(str).str.lower()
        print("[debug] sample source_path values:")
        for s in sp.dropna().unique()[:8]:
            print("   ", s)

    # If user provided a separate phot atlas CSV, load and override
    if args.phot_events_csv:
        df_ph2 = read_events(Path(args.phot_events_csv), use_weights=(not args.no_weights))
        _msg(f"[read] {len(df_ph2):,} rows from {args.phot_events_csv}", args.verbose)
        df_phot = df_ph2

    # If band filter found nothing (e.g., no source_path), interpret df_all as fluxatl
    if df_flux.empty and df_phot.empty:
        _msg("[warn] No source_path filter available; treating --events-csv as FluxAtlas-only.", args.verbose)
        df_flux = df_all.copy()

    # Prepare overlay ticks (optional)
    def ticks_from_df(df: pd.DataFrame, xlim: Optional[Tuple[float,float]]) -> np.ndarray:
        w = df["wavelength_nm"].to_numpy()
        if xlim:
            lo, hi = xlim; w = w[(w >= lo) & (w <= hi)]
        # down-thin to avoid too many ticks
        if w.size > 4000:
            w = np.random.RandomState(123).choice(w, size=4000, replace=False)
        return np.sort(w)

    ticks_flux = ticks_from_df(df_flux, xlim) if (args.overlay_ticks and not df_flux.empty) else None
    ticks_phot = ticks_from_df(df_phot, xlim) if (args.overlay_ticks and not df_phot.empty) else None

    # Build histograms
    def band_hist(df: pd.DataFrame):
        x, y = make_histogram(df, binsize=args.binsize, xlim=xlim, verbose=args.verbose)
        y = normalize_series(y, args.normalize) if y.size else y
        return x, y

    xF, yF = band_hist(df_flux) if not df_flux.empty else (np.array([]), np.array([]))
    xP, yP = band_hist(df_phot) if not df_phot.empty else (np.array([]), np.array([]))

    # Plot FluxAtlas
    if xF.size:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        plot_band(ax, xF, yF, "Sun (FluxAtlas, full spectrum)", xlim, ticks_flux)
        fig.tight_layout()
        fp = outdir / "solar_full_fluxatl.png"
        fig.savefig(fp, dpi=190); plt.close(fig)
        _msg(f"[write] {fp}", args.verbose)
    else:
        _msg("[skip] FluxAtlas band not found.", args.verbose)

    # Plot PhotAtlas
    if xP.size:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        plot_band(ax, xP, yP, "Sun (PhotAtlas, full spectrum)", xlim, ticks_phot)
        fig.tight_layout()
        fp = outdir / "solar_full_photatl.png"
        fig.savefig(fp, dpi=190); plt.close(fig)
        _msg(f"[write] {fp}", args.verbose)
    else:
        _msg("[skip] PhotAtlas band not found.", args.verbose)

    # Combined overlay if both present
    if xF.size and xP.size:
        fig, ax = plt.subplots(figsize=(12.5, 4.0))
        # draw both with slight alpha
        idxF = np.argsort(xF); ax.plot(xF[idxF], yF[idxF], lw=0.9, label="FluxAtlas", alpha=0.9)
        idxP = np.argsort(xP); ax.plot(xP[idxP], yP[idxP], lw=0.9, label="PhotAtlas", alpha=0.9)
        if args.overlay_ticks:
            ymin, ymax = ax.get_ylim()
            yy0, yy1 = ymin, ymin + 0.10*(ymax - ymin)
            if ticks_flux is not None: ax.vlines(ticks_flux, yy0, yy1, colors="0.4", alpha=0.18, lw=0.35)
            if ticks_phot is not None: ax.vlines(ticks_phot, yy0, yy1, colors="0.4", alpha=0.18, lw=0.35)
        if xlim: ax.set_xlim(*xlim)
        ax.set_title("Sun (FluxAtlas + PhotAtlas, full spectrum)")
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("normalized intensity")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fp = outdir / "solar_full_combined.png"
        fig.savefig(fp, dpi=190); plt.close(fig)
        _msg(f"[write] {fp}", args.verbose)

if __name__ == "__main__":
    main()
